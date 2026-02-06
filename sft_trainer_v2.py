#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2025, Authors.
# Licensed under the Apache License, Version 2.0.
"""
SFT Trainer with multiple loss functions.

This module provides a custom Trainer class that supports various loss functions
for supervised fine-tuning of language models, including:
- Standard Cross-Entropy (CE)
- Generalized Entropy Minimization (GEM)
- Selective Entropy-based Distillation (SED)
- Distribution-weighted Fine-Tuning (DFT)

Modified from HuggingFace Transformers Trainer.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional

from transformers import Trainer
from transformers.trainer import (
    _is_peft_model,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    is_torch_xla_available,
)

from utils.gem_triton_loss import GEMLoss
from utils.sed_triton_loss import SEDLoss
from utils.ce_triton_loss import CrossEntropyLoss


class SFTTrainer(Trainer):
    """
    Custom Trainer for SFT with multiple loss function support.
    
    Args:
        mode: Training mode identifier (default: "sft")
        kl_weight: Weight for KL divergence term (default: 0.1)
        clip_min: Minimum clipping value (default: 0.1)
        clip_max: Maximum clipping value (default: 2.0)
        origin_model: Reference model for distillation (default: None)
        *args, **kwargs: Additional arguments passed to Trainer
    """
    
    def __init__(
        self, 
        mode: str = "sft", 
        kl_weight: float = 0.1, 
        clip_min: float = 0.1, 
        clip_max: float = 2.0, 
        origin_model=None, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.kl_weight = kl_weight
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.origin_model = origin_model
        if origin_model is not None:
            self.origin_model.eval()
        print(f"Training mode: {mode}")

    @torch.no_grad()
    def compute_training_logs(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute training metrics for logging.
        
        Args:
            logits: Model output logits with shape (batch_size, seq_len, vocab_size)
            labels: Target labels with shape (batch_size, seq_len)
            
        Returns:
            Dictionary containing computed metrics
        """
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]

        training_logs = {}
        if self.args.print_entropy:
            entropy = chunked_entropy_from_logits(
                shift_logits,
                batch_size=max(1, shift_logits.size(0) // 4),
            ).mean()
            training_logs["entropy"] = round(entropy.item(), 2)

        return training_logs
    
    def dft_loss(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        num_items_in_batch: Optional[int], 
        ignore_index: int = -100
    ) -> torch.Tensor:
        """
        Compute Distribution-weighted Fine-Tuning loss.
        
        Weights each token's loss by its predicted probability, giving more weight
        to tokens the model is already confident about.
        
        Args:
            logits: Model output logits
            labels: Target labels
            num_items_in_batch: Number of items for loss normalization
            ignore_index: Label index to ignore in loss computation
            
        Returns:
            Computed DFT loss
        """
        ce_loss_func = CrossEntropyLoss(ignore_index=ignore_index, reduction="none")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]

        loss = ce_loss_func(shift_logits, shift_labels)
        
        # Compute probability-based weights
        probs = torch.softmax(shift_logits, dim=-1)
        weights = probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1).detach()
        weighted_losses = loss * weights
        
        if num_items_in_batch is not None:
            return weighted_losses.sum() / num_items_in_batch
        else:
            return weighted_losses.mean()

    def gem_loss(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        num_items_in_batch: Optional[int], 
        beta: float = 0.7, 
        ignore_index: int = -100, 
        h: str = "logsigmoid"
    ) -> torch.Tensor:
        """
        Compute Generalized Entropy Minimization (GEM) loss.
        
        Args:
            logits: Model output logits
            labels: Target labels
            num_items_in_batch: Number of items for loss normalization
            beta: Temperature parameter (0-1), closer to 1 behaves like CE
            ignore_index: Label index to ignore
            h: Weighting function type ("linear" or "logsigmoid")
            
        Returns:
            Computed GEM loss
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]

        with torch.no_grad():
            logits_on_labels = torch.gather(
                shift_logits, dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            logits_diff = shift_logits - logits_on_labels.unsqueeze(-1)
            
            if h == "linear":
                weights = torch.ones_like(logits_diff)
            elif h == "logsigmoid":
                weights = F.sigmoid(0.01 * logits_diff)
            else:
                raise ValueError(f"Unsupported h function: {h}")

        gene_log_probs = F.log_softmax(shift_logits, dim=-1)
        q_probs = torch.exp(F.log_softmax(shift_logits / beta, dim=-1)).detach()
        
        real_log_probs = torch.gather(
            gene_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        )
        probs_diff = real_log_probs - gene_log_probs

        if num_items_in_batch is not None:
            loss = -torch.sum(
                q_probs * weights * probs_diff, dim=-1
            ).sum() / num_items_in_batch
        else:
            loss = -torch.sum(
                q_probs * weights * probs_diff, dim=-1
            ).mean()

        return loss

    def gem_loss_triton(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        num_items_in_batch: Optional[int], 
        beta: float = 0.7, 
        ignore_index: int = -100, 
        h: str = "linear"
    ) -> torch.Tensor:
        """
        Compute GEM loss using Triton-optimized kernel.
        
        Args:
            logits: Model output logits
            labels: Target labels
            num_items_in_batch: Number of items for loss normalization
            beta: Temperature parameter
            ignore_index: Label index to ignore
            h: Weighting function (only "linear" supported for Triton version)
            
        Returns:
            Computed GEM loss
        """
        if h != "linear":
            raise ValueError("Only 'linear' is supported for gem_loss_triton")

        if num_items_in_batch is not None:
            gem_loss_func = GEMLoss(beta=beta, ignore_index=ignore_index, reduction="none")
        else:
            gem_loss_func = GEMLoss(beta=beta, ignore_index=ignore_index, reduction="mean")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]

        loss = gem_loss_func(shift_logits, shift_labels)

        if num_items_in_batch is not None:
            loss = loss.sum() / num_items_in_batch

        return loss

    def sed_loss_triton(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        num_items_in_batch: Optional[int], 
        beta: float = 0.7, 
        ignore_index: int = -100, 
        h: str = "linear",
        entropy_penalty_scale: float = 0.2, 
        use_static: bool = False, 
        top_k: int = 10, 
        cumsum_threshold: float = 0.95,
        use_low_entropy_mask: bool = False, 
        use_low_topk_cumsum_ratio: bool = False
    ):
        """
        Compute Selective Entropy-based Distillation (SED) loss using Triton kernel.
        
        This loss selectively applies entropy penalties based on model confidence,
        measured by top-k cumulative probability.
        
        Args:
            logits: Model output logits
            labels: Target labels
            num_items_in_batch: Number of items for loss normalization
            beta: Temperature parameter
            ignore_index: Label index to ignore
            h: Weighting function (only "linear" supported)
            entropy_penalty_scale: Scale factor for entropy penalty
            use_static: Use static entropy penalty
            top_k: Number of top tokens for cumulative probability
            cumsum_threshold: Threshold for cumulative probability masking
            use_low_entropy_mask: Apply masking based on entropy
            use_low_topk_cumsum_ratio: Apply masking based on top-k cumsum ratio
            
        Returns:
            If using masking: tuple of (loss, correct_pred_rate, mask_rate, extra_loss)
            Otherwise: Computed SED loss
        """
        if h != "linear":
            raise ValueError("Only 'linear' is supported for sed_loss_triton")

        reduction = "none" if num_items_in_batch is not None else "mean"
        
        sed_loss_func = SEDLoss(
            beta=beta, 
            ignore_index=ignore_index, 
            reduction=reduction,
            entropy_penalty_scale=entropy_penalty_scale, 
            use_static=use_static,
            top_k=top_k, 
            cumsum_threshold=cumsum_threshold, 
            use_low_entropy_mask=use_low_entropy_mask, 
            use_low_topk_cumsum_ratio=use_low_topk_cumsum_ratio
        )

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]
        
        loss = sed_loss_func(shift_logits, shift_labels)
        
        if isinstance(loss, tuple):
            train_loss = loss[0]
            correct_pred_rate = loss[1]
            mask_rate = loss[2]
            extra_loss = loss[3]
            if num_items_in_batch is not None:
                train_loss = train_loss.sum() / num_items_in_batch
            return train_loss, correct_pred_rate, mask_rate, extra_loss
    
        if num_items_in_batch is not None:
            loss = loss.sum() / num_items_in_batch

        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the training loss.

        This method is overridden to support multiple loss functions including
        CE, GEM, DFT, and SED variants.
        
        Args:
            model: The model being trained
            inputs: Input batch dictionary
            return_outputs: Whether to return model outputs along with loss
            num_items_in_batch: Number of items for loss normalization
            
        Returns:
            Loss tensor, or tuple of (loss, outputs) if return_outputs=True
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
            
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
            
        outputs = model(**inputs)
        
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
                
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                if self.args.loss == "gem_dynamic_triton":
                    decay_rate = 1.0 + self.state.global_step / self.state.max_steps * 0.5
                loss = self.compute_loss_func(
                    outputs, labels, 
                    num_items_in_batch=num_items_in_batch, 
                    dynamic_weight_decay=decay_rate
                )
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
                
            # Select loss function based on configuration
            if self.args.loss == "ce" or self.control.should_evaluate:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            elif self.args.loss == "dft":
                loss = self.dft_loss(
                    outputs.logits, 
                    inputs["labels"],
                    num_items_in_batch=num_items_in_batch, 
                )
            elif self.args.loss == "gem":
                loss = self.gem_loss(
                    outputs.logits, 
                    inputs["labels"],
                    num_items_in_batch=num_items_in_batch, 
                    beta=self.args.gem_beta, 
                    h=self.args.gem_h
                )
            elif self.args.loss == "gem_triton":
                loss = self.gem_loss_triton(
                    outputs.logits, 
                    inputs["labels"],
                    num_items_in_batch=num_items_in_batch, 
                    beta=self.args.gem_beta, 
                    h=self.args.gem_h
                )
            elif self.args.loss == "sed_triton":
                loss = self.sed_loss_triton(
                    outputs.logits, 
                    inputs["labels"],
                    num_items_in_batch=num_items_in_batch, 
                    beta=self.args.gem_beta, 
                    h=self.args.gem_h,
                    entropy_penalty_scale=self.args.entropy_penalty_scale,
                )
            elif self.args.loss == "sed_with_topk_cumsum_ratio":
                loss = self.sed_loss_triton(
                    outputs.logits, 
                    inputs["labels"],
                    num_items_in_batch=num_items_in_batch, 
                    beta=self.args.gem_beta, 
                    h=self.args.gem_h,
                    entropy_penalty_scale=self.args.entropy_penalty_scale,
                    top_k=self.args.top_k if self.args.top_k is not None else 2, 
                    use_low_topk_cumsum_ratio=True,
                    cumsum_threshold=self.args.cumsum_threshold
                )

        if not isinstance(loss, tuple):
            loss *= self.accelerator.num_processes

            # Add training logs
            if not self.control.should_evaluate:
                self.training_logs = self.compute_training_logs(
                    outputs.logits, inputs["labels"]
                )
                self.training_logs["ce_loss"] = (
                    outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                )
                self.training_logs["ce_loss"] = round(self.training_logs["ce_loss"].item(), 4)

            return (loss, outputs) if return_outputs else loss
        else:
            # Handle tuple loss (from SED with extra metrics)
            correct_pred_rate = loss[1]
            mask_rate = loss[2]
            extra_loss = loss[3]
            loss = loss[0]

            loss *= self.accelerator.num_processes
            
            if not self.control.should_evaluate:
                self.training_logs = self.compute_training_logs(
                    outputs.logits, inputs["labels"]
                )
                self.training_logs["ce_loss"] = (
                    outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                )
                self.training_logs["ce_loss"] = round(self.training_logs["ce_loss"].item(), 4)
                self.training_logs["extra_loss"] = extra_loss.item()
                self.training_logs["correct_pred_rate"] = correct_pred_rate.item()
                self.training_logs["mask_rate"] = mask_rate.item()

            return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(
        self, 
        tr_loss, 
        grad_norm, 
        model, 
        trial, 
        epoch, 
        ignore_keys_for_eval, 
        start_time=None, 
        learning_rate=None
    ):
        """Handle logging, saving checkpoints, and evaluation."""
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # Reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()
            if getattr(self, "training_logs", None):
                logs.update(self.training_logs)

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


def chunked_entropy_from_logits(
    chunk_logits: torch.Tensor, 
    batch_size: Optional[int] = None
) -> torch.Tensor:
    """
    Compute entropy from logits in a memory-efficient manner.
    
    Uses batched processing to avoid memory issues with large tensors.

    Args:
        chunk_logits: Logits tensor of shape (total_samples, num_classes)
        batch_size: Number of samples to process per batch (default: all)

    Returns:
        Entropy tensor of shape (total_samples,)
    """
    total_samples, num_classes = chunk_logits.shape
    entropy_list = []
    if batch_size is None:
        batch_size = total_samples

    # Process logits in batches
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        logits_batch = chunk_logits[start_idx:end_idx]

        # Compute logsumexp for the current batch
        logsumexp_batch = torch.logsumexp(logits_batch, dim=-1, keepdim=False)
        
        # Compute probabilities in log-space without computing softmax
        normalized_logits = logits_batch - logsumexp_batch.unsqueeze(-1)
        exp_normalized_logits = torch.exp(normalized_logits)
        
        # Compute entropy for the batch
        entropy_batch = logsumexp_batch - (logits_batch * exp_normalized_logits).sum(dim=-1)

        entropy_list.append(entropy_batch)

    # Concatenate results from all batches
    if len(entropy_list) > 0:
        return torch.cat(entropy_list, dim=0)
    else:
        return torch.tensor(0.0)