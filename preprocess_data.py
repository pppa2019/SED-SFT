#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2025, Authors.
# Licensed under the Apache License, Version 2.0.
"""
Data preprocessing script for SFT training.

This script tokenizes raw conversation data into a format suitable for
supervised fine-tuning. It handles multi-turn conversations and properly
masks non-assistant turns in the labels.

Usage:
    python preprocess_data.py \
        --dataset_name_or_path /path/to/dataset \
        --tokenizer_name_or_path /path/to/tokenizer \
        --output_file /path/to/output.jsonl
"""

import json
import os
from argparse import ArgumentParser
from multiprocessing import Pool
from typing import Any, Dict, List, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Tokenize dataset for SFT training")
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="HuggingFace dataset name or path to local dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for data selection",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index for data selection (default: all)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        required=True,
        help="Path to tokenizer or HuggingFace model name",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Maximum sequence length after tokenization",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=64,
        help="Number of parallel workers for tokenization",
    )
    return parser.parse_args()


# Global tokenizer and max_seq_length (for multiprocessing)
tokenizer = None
max_seq_length = None


def init_tokenizer(tokenizer_path: str, seq_length: int):
    """Initialize global tokenizer."""
    global tokenizer, max_seq_length
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    max_seq_length = seq_length
    print(f"Loaded tokenizer from {tokenizer_path}")


def convert_to_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert various data formats to standard message format.
    
    Supports the following formats:
    - question/response pairs
    - question/answer pairs
    - Pre-formatted messages list
    
    Args:
        example: Data example dictionary
        
    Returns:
        List of message dictionaries with 'role' and 'content' keys
    """
    if 'question' in example and 'response' in example:
        return [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}"},
            {"role": "user", "content": example["question"]}, 
            {"role": "assistant", "content": example['response']}
        ]
    elif 'question' in example and 'answer' in example:
        return [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}"},
            {"role": "user", "content": example["question"]}, 
            {"role": "assistant", "content": example['answer']}
        ]
    else:
        return example.get("messages", [])


def encode_sft_example(example: Dict[str, Any], verbose: bool = False) -> Dict[str, List[int]]:
    """
    Encode a single example for SFT training.
    
    This function tokenizes the conversation and masks non-assistant turns
    in the labels (setting them to -100) so they don't contribute to the loss.
    
    Args:
        example: Raw data example with conversation data
        verbose: If True, print the formatted chat for debugging
        
    Returns:
        Dictionary containing:
        - input_ids: Token IDs for the full conversation
        - labels: Token IDs with non-assistant turns masked as -100
        - attention_mask: Attention mask (all 1s)
    """
    messages = convert_to_messages(example)
    
    if len(messages) == 0:
        raise ValueError("messages field is empty.")
    
    if verbose:
        chat_messages = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_seq_length,
            add_generation_prompt=False,
        )
        print(f"Chat messages:\n[{chat_messages}]")
    
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=False,
    )
    labels = input_ids.clone()
    
    # Mask non-assistant turns in labels
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            # Calculate start index of this non-assistant message
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer.apply_chat_template(
                    conversation=messages[:message_idx],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            
            # Calculate end index of this non-assistant message
            if (
                message_idx < len(messages) - 1
                and messages[message_idx + 1]["role"] == "assistant"
            ):
                # Include generation prompt in masked region
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[:message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=True,
                ).shape[1]
            else:
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[:message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            
            # Set labels to -100 for non-assistant content
            labels[:, message_start_idx:message_end_idx] = -100
            
            if max_seq_length and message_end_idx >= max_seq_length:
                break
    
    attention_mask = torch.ones_like(input_ids)
    
    return {
        "input_ids": input_ids.flatten().tolist(),
        "labels": labels.flatten().tolist(),
        "attention_mask": attention_mask.flatten().tolist(),
    }


def main():
    """Main preprocessing function."""
    args = parse_args()
    
    # Initialize tokenizer
    init_tokenizer(args.tokenizer_name_or_path, args.max_seq_length)
    
    # Load dataset
    input_data = load_dataset(args.dataset_name_or_path)
    if args.split:
        input_data = input_data[args.split]
    
    # Select data range
    if args.end is None:
        args.end = len(input_data)
    input_data = input_data.select(range(args.start, args.end))
    
    print(
        f"Loaded data from {args.dataset_name_or_path}. "
        f"Processing {len(input_data)} examples (indices {args.start} to {args.end})"
    )
    
    # Show example output
    print("\n=== Example tokenization ===")
    print(encode_sft_example(input_data[0], verbose=True))
    print("=" * 50 + "\n")
    
    # Tokenize data in parallel
    tokenized_data = []
    with Pool(
        processes=args.preprocessing_num_workers,
        initializer=init_tokenizer,
        initargs=(args.tokenizer_name_or_path, args.max_seq_length)
    ) as pool:
        pbar = tqdm(input_data, desc="Tokenizing")
        for tokenized_example in pool.imap(encode_sft_example, pbar):
            dump = json.dumps(tokenized_example)
            tokenized_data.append(dump)
    
    # Write output
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as fw:
        for dump in tokenized_data:
            fw.write(dump + "\n")
    
    print(f"\nSaved {len(tokenized_data)} examples to {args.output_file}")


if __name__ == "__main__":
    main()
