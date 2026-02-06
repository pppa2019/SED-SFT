# Copyright (c) 2025, Authors.
# Licensed under the Apache License, Version 2.0.
# Modified from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/losses/cross_entropy.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from .sed_triton_ops import cross_entropy_loss


def entropy(probs: torch.Tensor) -> torch.Tensor:
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)


class SEDLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = "mean",
        beta: float = 1.0,
        logit_scale: float = 1.0,
        lse_square_scale: float = 0.0,
        inplace_backward: bool = False,
        process_group=None,
        return_z_loss: bool = False,
        entropy_penalty_scale: float = 0.2,
        use_static: bool = True,
        use_dynamic: bool = False,
        use_correct_loss_mask: bool = False,
        use_dft_factor: bool = False,
        use_reversed_dft_factor: bool = False,
        use_top_k: bool = False,
        use_full_vocab: bool = False,
        use_context_certain_mask: bool = False,
        top_k: int = 2,
        use_low_entropy_mask: bool = False,
        use_less_low_entropy_mask: bool = False,
        use_low_topk_cumsum_ratio: bool = False,
        cumsum_threshold: float = 0.95,
    ):
        super().__init__()
        if reduction not in ["mean", "none", "sum"]:
            raise NotImplementedError("Only support reduction = 'mean', 'none', or 'sum'")
            
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.beta = beta
        self.logit_scale = logit_scale
        self.lse_square_scale = lse_square_scale
        self.inplace_backward = inplace_backward
        self.process_group = process_group
        self.return_z_loss = return_z_loss
        self.entropy_penalty_scale = entropy_penalty_scale
        self.use_static = use_static
        self.use_top_k = use_top_k
        self.top_k = top_k
        self.use_low_entropy_mask = use_low_entropy_mask
        self.cumsum_threshold = cumsum_threshold
        self.use_low_topk_cumsum_ratio = use_low_topk_cumsum_ratio

    def forward(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor, 
        precomputed_lse: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        assert input.is_cuda and target.is_cuda, "Only CUDA tensors are supported"
        
        # Compute base cross-entropy loss
        loss, z_loss = cross_entropy_loss(
            input,
            target,
            precomputed_lse=precomputed_lse,
            logit_scale=self.logit_scale,
            lse_square_scale=0.0,
            ignore_index=self.ignore_index,
            inplace_backward=self.inplace_backward,
            process_group=self.process_group,
        )
      
        scaled_input = input * self.logit_scale
        probs = F.softmax(scaled_input, dim=-1)

        # Compute top-k cumulative probability
        k = self.top_k
        topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1, largest=True, sorted=True)
        topk_sum_probs = torch.sum(topk_probs, dim=-1)

        # Get probability of target tokens
        batch_size = target.size(0)
        target_probs = probs[torch.arange(batch_size, device=input.device), target]
        
        mask = target != self.ignore_index
        dynamic_terms = torch.zeros_like(target_probs)

        # Compute entropy-based penalty term
        # This encourages learning on uncertain tokens
        dynamic_terms[mask] = target_probs[mask] * (target_probs[mask] - 1) + 0.25

        # Create confidence mask based on cumulative probability threshold
        top_prob_idx = torch.argmax(probs, dim=-1)
        correct_pred_loss_mask = topk_sum_probs < self.cumsum_threshold

        # Compute metrics
        correct_pred_rate = torch.sum((top_prob_idx == target)) / top_prob_idx.size(0)
        mask_rate = 1 - torch.sum(correct_pred_loss_mask) / correct_pred_loss_mask.size(0)
     
        # Apply entropy penalty only to low-confidence tokens
        extra_loss = dynamic_terms * correct_pred_loss_mask * self.entropy_penalty_scale
        weighted_loss = loss + extra_loss
    
        # Apply reduction
        if self.reduction == "mean":
            if mask.sum() > 0:
                weighted_loss = weighted_loss.sum() / mask.sum()
            else:
                weighted_loss = torch.tensor(0.0, device=input.device, dtype=input.dtype)
        elif self.reduction == "sum":
            weighted_loss = weighted_loss.sum()
        
        # Handle z_loss if enabled
        if self.return_z_loss:
            _, z_loss = cross_entropy_loss(
                input,
                target,
                precomputed_lse=precomputed_lse,
                logit_scale=self.logit_scale,
                lse_square_scale=self.lse_square_scale,
                ignore_index=self.ignore_index,
                inplace_backward=self.inplace_backward,
                process_group=self.process_group
            )
            
            if self.reduction == "mean":
                z_loss = z_loss.sum() / (target != self.ignore_index).sum()
            elif self.reduction == "sum":
                z_loss = z_loss.sum()
            
            return weighted_loss, z_loss, correct_pred_rate, mask_rate, extra_loss.mean()
        
        return weighted_loss, correct_pred_rate, mask_rate, extra_loss.mean()