#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2025, Authors.
# Licensed under the Apache License, Version 2.0.
"""
Main training script for SFT with various loss functions.

This script is modified from the HuggingFace example for fine-tuning language models:
https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py

Supports multiple loss functions including CE, GEM, DFT, and SED variants.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import deepspeed
import torch
import torch.distributed as dist
import transformers
from datasets import load_dataset
from packaging import version
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from sft_trainer_v2 import SFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Extended training arguments for SFT with multiple loss functions.
    """
    adam_beta2: float = field(
        default=0.95, 
        metadata={"help": "Beta2 for AdamW optimizer"}
    )
    loss: str = field(
        default="gem", 
        metadata={
            "help": "Loss function to use",
            "choices": ["ce", "gem_triton", "sed_triton", "dft", "sed_with_topk_cumsum_ratio"]
        }
    )
    gem_beta: float = field(
        default=0.7, 
        metadata={
            "help": "Temperature parameter for GEM/SED. Range: 0-1. "
                    "Values close to 1.0 make GEM behave like CE, "
                    "values close to 0.0 preserve more diversity."
        }
    )
    gem_h: str = field(
        default="linear", 
        metadata={
            "help": "Weighting function h in GEM. 'logsigmoid' is more adaptive.",
            "choices": ["logsigmoid", "linear"]
        }
    )
    print_entropy: bool = field(
        default=False, 
        metadata={"help": "Print entropy during training"}
    )
    entropy_penalty_scale: float = field(
        default=1.0, 
        metadata={"help": "Scale factor for entropy penalty term"}
    )
    block_size: int = field(
        default=1024, 
        metadata={"help": "Block size for training"}
    )
    top_k: Optional[int] = field(
        default=None, 
        metadata={"help": "Top-k tokens for cumulative probability calculation in SED"}
    )
    cumsum_threshold: Optional[float] = field(
        default=None,
        metadata={"help": "Threshold for cumulative probability masking"}
    )
    cumsum_ratio: float = field(
        default=0.3, 
        metadata={"help": "Ratio for computing cumsum threshold from validation data"}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to model configuration.
    """
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory to store pretrained models downloaded from huggingface.co"
        },
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether to use Flash Attention"},
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to data loading and preprocessing.
    """
    train_tokenized_file: Optional[str] = field(
        default=None, 
        metadata={"help": "Path to tokenized training data (JSONL format)"}
    )
    test_tokenized_file: Optional[str] = field(
        default=None, 
        metadata={"help": "Path to tokenized test data (JSONL format)"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Truncate training examples to this number (for debugging)"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum sequence length after tokenization. Longer sequences are truncated."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )


class CustomDataset(Dataset):
    """
    Custom dataset for loading pre-tokenized training data.
    
    Args:
        training_args: Training configuration
        data_args: Data configuration
        model_args: Model configuration
        train_tokenized_file: Path to tokenized data file
    """
    
    def __init__(
        self,
        training_args: TrainingArguments,
        data_args: DataArguments,
        model_args: ModelArguments,
        train_tokenized_file: str,
    ):
        self.training_args = training_args
        self.data_args = data_args
        self.model_args = model_args

        raw_datasets = load_dataset(
            "json",
            data_files=[train_tokenized_file],
            cache_dir=self.model_args.cache_dir,
        )
        self.data = raw_datasets["train"]

        if self.data_args.max_train_samples is not None:
            max_samples = min(len(self.data), self.data_args.max_train_samples)
            self.data = self.data.select(range(max_samples))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> dict:
        example = self.data[item]
        assert "input_ids" in example, "Missing 'input_ids' in data"
        assert "labels" in example, "Missing 'labels' in data"
        example = {k: torch.tensor(v, dtype=torch.long) for k, v in example.items()}
        return example


def compute_cumsum_threshold(
    model: AutoModelForCausalLM,
    test_dataset: CustomDataset,
    top_k: int,
    cumsum_ratio: float,
    num_samples: int = 100
) -> float:
    """
    Compute cumulative probability threshold from validation data.
    
    This threshold is used to determine which tokens should receive
    additional entropy-based regularization.
    
    Args:
        model: The language model
        test_dataset: Validation dataset
        top_k: Number of top tokens for cumulative probability
        cumsum_ratio: Ratio for determining threshold percentile
        num_samples: Number of samples to use for computation
        
    Returns:
        Computed threshold value
    """
    cumsum_list = []
    
    with torch.no_grad():
        for input_ids in test_dataset.data['input_ids'][:num_samples]:
            logits = model(
                torch.LongTensor(input_ids).unsqueeze(0).to(model.device)
            ).logits
            probs = torch.softmax(logits, dim=-1)
            topk_probs, _ = torch.topk(probs, k=top_k, dim=-1)
            cumsum = torch.sum(topk_probs, dim=-1).squeeze(0)
            cumsum_list.extend(cumsum.tolist())
    
    cumsum_k = int(cumsum_ratio * len(cumsum_list))
    threshold = sorted(cumsum_list)[-cumsum_k]
    
    return threshold


def main():
    """Main training function."""
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.origin_model_path = model_args.model_name_or_path
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log process information
    global_rank = dist.get_rank()
    logger.warning(
        f"Process rank: {global_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    )
    logger.info(f"Training parameters: {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype="auto",
    )

    # Setup datasets
    train_dataset = CustomDataset(
        training_args, data_args, model_args, 
        data_args.train_tokenized_file
    )
    
    test_dataset = None
    if data_args.test_tokenized_file:
        test_dataset = CustomDataset(
            training_args, data_args, model_args, 
            data_args.test_tokenized_file
        )

    model.to(training_args.device)
    
    # Compute cumsum threshold if needed
    if "cumsum_ratio" in training_args.loss and training_args.cumsum_threshold is None:
        if test_dataset is None:
            raise ValueError(
                "test_tokenized_file is required for computing cumsum_threshold"
            )
        
        top_k = training_args.top_k if training_args.top_k is not None else 10
        training_args.cumsum_threshold = compute_cumsum_threshold(
            model=model,
            test_dataset=test_dataset,
            top_k=top_k,
            cumsum_ratio=training_args.cumsum_ratio,
        )
        logger.info(f"Computed cumsum_threshold: {training_args.cumsum_threshold}")

    # Resize embeddings if necessary
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
        
    if len(tokenizer) > embedding_size:
        logger.warning(
            f"Tokenizer size ({len(tokenizer)}) > embedding size ({embedding_size}). Resizing..."
        )
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    # Enable gradient checkpointing if specified
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, 
            model=model, 
            padding="longest"
        ),
        preprocess_logits_for_metrics=None,
        compute_metrics=None,
    )

    # Start training
    logger.info("*** Starting Training ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)


if __name__ == "__main__":
    main()
