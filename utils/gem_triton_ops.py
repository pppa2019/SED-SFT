# Copyright (c) 2025, Ziniu Li.
# The code is modified from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/cross_entropy.py

from typing import Tuple, Optional, Union

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

# `all_gather_into_tensor` and `reduce_scatter_tensor` are new placeholders for
# `_all_gather_base` and `_reduce_scatter_base`. They require the most recent
# version of PyTorch. The following 2 lines are for backward compatibility with
# older PyTorch.
if "all_gather_into_tensor" not in dir(torch.distributed):
    torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base

@triton.jit
def gem_fwd_kernel(
    loss_ptr,  # data ptrs
    lse_ptr,
    z_loss_ptr,
    logits_ptr,
    labels_ptr,
    beta,
    logit_scale,
    lse_square_scale,
    ignore_index,
    total_classes,
    class_start_idx,  # Useful for tensor parallel when each rank only has a subset of classes
    n_cols,  # shapes
    logits_row_stride,  # strides
    BLOCK_SIZE: tl.constexpr,
    # if SPLIT (e.g. tensor parallel), don't include the LSE in the loss since it's not the final LSE
    SPLIT: tl.constexpr,
    PRECOMPUTED_LSE: tl.constexpr,  # If LSE is already computed (also no smoothing and logit_scale == 1.0)
):
    # GEM Loss (h = linear)
    # Loss = -log p(y|x) + sum_y q(y|x) * log p(y|x)
    # q (y|x) = softmax (1 / beta * log p(y|x))
    # Note that the first term is the same as the CE loss. 

    # Prepare for calculating lse.
    row_idx = tl.program_id(0)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)

    if not PRECOMPUTED_LSE:
        # Statistics for online softmax
        m_i = -float("inf")
        l_i = 0.0
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            cols = col_offset + tl.arange(0, BLOCK_SIZE)
            logits = tl.load(logits_ptr + cols, mask=cols < n_cols, other=-float("inf")).to(
                tl.float32
            ) * logit_scale
            m_i_new = tl.maximum(m_i, tl.max(logits))
            l_i = tl.exp(m_i - m_i_new) * l_i + tl.sum(tl.exp(logits - m_i_new))
            m_i = m_i_new
        lse = tl.log(l_i) + m_i
        tl.store(lse_ptr + row_idx, lse)
    else:
        lse = tl.load(lse_ptr + row_idx)
    label_idx = tl.load(labels_ptr + row_idx)

    # Second term: q-regularized loss 
    m_q = -float("inf")  # running max
    s_q = 0.0  # running sum for denominator
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(logits_ptr + cols, mask=cols < n_cols, other=-float("inf")).to(
            tl.float32
        ) * logit_scale
        logits_scaled = logits / beta
        
        # Update running max and rescale previous sum
        m_q_prev = m_q
        m_q = tl.maximum(m_q, tl.max(logits_scaled))
        s_q = tl.exp(m_q_prev - m_q) * s_q
        
        # Add contribution from current block
        numerator = tl.exp(logits_scaled - m_q)
        s_q += tl.sum(tl.where(cols < n_cols, numerator, 0.0))
    
    # Second pass: compute q_loss using final m_q and s_q
    q_loss = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(logits_ptr + cols, mask=cols < n_cols, other=-float("inf")).to(
            tl.float32
        ) * logit_scale
        logits_scaled = logits / beta
        
        # Compute q_probs using final normalization constants
        q_probs = tl.exp(logits_scaled - m_q) / s_q
        q_loss += tl.sum(tl.where(cols < n_cols, q_probs * (logits - lse), 0.0))

    # Compute CE loss term (same as before)
    label_idx = tl.load(labels_ptr + row_idx)
    if label_idx == ignore_index:
        loss = 0.0
        z_loss = 0.0
    else:
        label_idx -= class_start_idx
        if label_idx >= 0 and label_idx < n_cols:
            logits_label = tl.load(logits_ptr + label_idx) * logit_scale
            # GEM loss = CE loss + q_loss
            loss = (lse if not SPLIT else 0.0) - logits_label +  q_loss
        else:
            loss = q_loss
        if not SPLIT:
            z_loss = lse_square_scale * lse * lse
            loss += z_loss
        else:
            z_loss = 0.0
    tl.store(loss_ptr + row_idx, loss)
    if not SPLIT:
        tl.store(z_loss_ptr + row_idx, z_loss)


@triton.jit
def gem_bwd_kernel(
    dlogits_ptr,  # data ptrs
    dloss_ptr,
    logits_ptr,
    lse_ptr,
    labels_ptr,
    beta,
    logit_scale,
    lse_square_scale,
    ignore_index,
    total_classes,
    class_start_idx,  # Useful for tensor parallel when each rank only has a subset of classes
    n_cols,  # shapes
    logits_row_stride,  # strides
    dlogits_row_stride,
    dloss_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    dlogits_ptr = dlogits_ptr + row_idx * dlogits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    if label_idx != ignore_index:
        dloss = tl.load(dloss_ptr + row_idx * dloss_row_stride)
    else:
        dloss = 0.0
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(
        tl.float32
    ) * logit_scale
    lse = tl.load(lse_ptr + row_idx)
    
    # GEM: gradient of y w.r.t. logits = q(y|x) if y != label else q(y|x) - 1.
    # First pass: compute max and sum for proper softmax normalization
    m_q = -float("inf")
    s_q = 0.0
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        logits_block = tl.load(logits_ptr + cols, mask=cols < n_cols, other=-float("inf")).to(tl.float32) * logit_scale
        logits_scaled = logits_block / beta
        m_q = tl.maximum(m_q, tl.max(logits_scaled))

    # Second pass: compute sum with stable numerics
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        logits_block = tl.load(logits_ptr + cols, mask=cols < n_cols, other=-float("inf")).to(tl.float32) * logit_scale
        logits_scaled = logits_block / beta
        numerator = tl.exp(logits_scaled - m_q)
        s_q += tl.sum(tl.where(cols < n_cols, numerator, 0.0))

    # Final pass: compute gradients for current block
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(tl.float32) * logit_scale
    logits_scaled = logits / beta
    probs = tl.exp(logits_scaled - m_q) / s_q

    probs += 2.0 * lse_square_scale * lse * probs
    label_idx -= class_start_idx
    probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    tl.store(dlogits_ptr + col_offsets, (dloss * logit_scale) * probs, mask=col_offsets < n_cols)


class GEMLoss(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        logits,
        labels,
        precomputed_lse=None,
        beta=1.0,
        logit_scale=1.0,
        lse_square_scale=0.0,
        ignore_index=-100,
        inplace_backward=False,
        process_group=None,
    ):
        # For some reason Triton generates wrong code when labels has dtype long and its address
        # is not aligned to 16 bytes. The ld.global.b64 seems to load the wrong label index.
        if labels.dtype == torch.long and labels.data_ptr() % 16 != 0:
            labels = F.pad(labels, (0, 1))[..., :-1]
            assert labels.data_ptr() % 16 == 0
        assert logit_scale > 0.0
        n_rows, n_cols = logits.shape
        assert labels.shape == (n_rows,)
        world_size = 1 if process_group is None else torch.distributed.get_world_size(process_group)
        total_classes = world_size * n_cols
        rank = 0 if process_group is None else torch.distributed.get_rank(process_group)
        class_start_idx = rank * n_cols
        use_precomputed_lse = precomputed_lse is not None and logit_scale == 1.0

        if logits.stride(-1) != 1:
            logits = logits.contiguous()
        MAX_BLOCK_SIZE = 16 * 1024
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), MAX_BLOCK_SIZE)
        num_warps = (
            4
            if BLOCK_SIZE < 2048
            else (8 if BLOCK_SIZE < 8192 else (16 if BLOCK_SIZE < 128 * 1024 else 32))
        )
        losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
        if use_precomputed_lse:
            assert precomputed_lse.shape == (n_rows,)
            lse = precomputed_lse.contiguous()
        else:
            lse = torch.empty(n_rows, dtype=torch.float, device=logits.device)
        z_losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
        # Need this, otherwise Triton tries to launch from cuda:0 and we get
        # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
        with torch.cuda.device(logits.device.index):
            gem_fwd_kernel[(n_rows,)](
                losses,  # data ptrs
                lse,
                z_losses,
                logits,
                labels,
                beta,
                logit_scale,
                lse_square_scale,
                ignore_index,
                total_classes,
                class_start_idx,
                n_cols,  # shapes
                logits.stride(0),  # strides
                BLOCK_SIZE=BLOCK_SIZE,  # constants
                SPLIT=world_size > 1,
                PRECOMPUTED_LSE=use_precomputed_lse,
                num_warps=num_warps,
            )

        if world_size > 1:
            # For GEM loss, if labels are in the vocab of this partition, losses contains
            # - predicted logit + q_loss for this partition, and 0 otherwise.
            # For labels not in the vocab of this partition, losses contains
            # only q_loss for this partition
            lse_allgather = torch.empty(world_size, n_rows, dtype=lse.dtype, device=lse.device)
            torch.distributed.all_gather_into_tensor(lse_allgather, lse, group=process_group)
            handle_losses = torch.distributed.all_reduce(
                losses, op=torch.distributed.ReduceOp.SUM, group=process_group, async_op=True
            )
            lse = torch.logsumexp(lse_allgather, dim=0)
            handle_losses.wait()
            # After the allreduce:
            # 1. For labels in any partition, we have the sum of all q_losses from each partition
            # 2. For the partition containing the label, we also have -predicted_logit
            # We just need to add the global lse to complete the GEM loss
            losses += lse
            if lse_square_scale != 0.0:
                z_losses = lse_square_scale * lse.square()
                z_losses.masked_fill_(labels == ignore_index, 0.0)
                losses += z_losses
            else:
                z_losses = torch.zeros_like(losses)
            losses.masked_fill_(labels == ignore_index, 0.0)

        ctx.save_for_backward(logits, lse, labels)
        ctx.mark_non_differentiable(z_losses)
        ctx.beta = beta
        ctx.logit_scale = logit_scale
        ctx.lse_square_scale = lse_square_scale
        ctx.ignore_index = ignore_index
        ctx.total_classes = total_classes
        ctx.class_start_idx = class_start_idx
        ctx.inplace_backward = inplace_backward
        return losses, z_losses

    @staticmethod
    def backward(ctx, grad_losses, grad_z_losses):
        del grad_z_losses  # z_losses are only for logging.

        logits, lse, labels = ctx.saved_tensors
        dlogits = logits if ctx.inplace_backward else torch.empty_like(logits)
        n_rows, n_cols = logits.shape
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 4 * 1024)
        num_warps = 4 if BLOCK_SIZE < 2048 else (8 if BLOCK_SIZE < 8192 else 16)
        grid = lambda META: (n_rows, triton.cdiv(n_cols, META["BLOCK_SIZE"]))  # noqa
        # Need this, otherwise Triton tries to launch from cuda:0 and we get
        # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
        with torch.cuda.device(logits.device.index):
            gem_bwd_kernel[grid](
                dlogits,  # data ptrs
                grad_losses,
                logits,
                lse,
                labels,
                ctx.beta,
                ctx.logit_scale,
                ctx.lse_square_scale,
                ctx.ignore_index,
                ctx.total_classes,
                ctx.class_start_idx,
                n_cols,  # shapes
                logits.stride(0),  # strides
                dlogits.stride(0),
                grad_losses.stride(0),
                BLOCK_SIZE=BLOCK_SIZE,  # constants
                num_warps=num_warps,
            )
        return dlogits, None, None, None, None, None, None, None, None, None


def gem_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    precomputed_lse: Optional[torch.Tensor] = None,
    beta: float = 1.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    ignore_index=-100,
    inplace_backward: bool = False,
    process_group=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        logits: (batch, vocab_size)
        labels: (batch,)
        beta: float
        logit_scale: float. Multiply logits by this scale before calculating the loss.
        lse_square_scale: float. If > 0, we add lse_square_scale * lse(logits) ^ 2 to the loss.
            This is also referred to as "z-loss".
        ignore_index: int. If labels == ignore_index, the loss is set to 0.0.
        inplace_backward: bool. If True, we do the backward pass in-place by modifying the logits.
            This saves memory.
        process_group: if not None, we're doing Tensor Parallel: each process is responsible for
            one part of the vocab. The loss will be aggregated across processes.
    Returns:
        losses: (batch,), float
        z_losses: (batch,), float
    """
    return GEMLoss.apply(
        logits,
        labels,
        precomputed_lse,
        beta,
        logit_scale,
        lse_square_scale,
        ignore_index,
        inplace_backward,
        process_group,
    )
