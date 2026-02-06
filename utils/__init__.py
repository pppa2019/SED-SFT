# Copyright (c) 2025, Authors.
# Licensed under the Apache License, Version 2.0.

"""
Triton-optimized loss functions for efficient language model training.

This module provides:
- CrossEntropyLoss: Standard cross-entropy loss with Triton optimization
- GEMLoss: Generalized Entropy Minimization loss
- SEDLoss: Selective Entropy-based Distillation loss
"""

from .ce_triton_loss import CrossEntropyLoss
from .gem_triton_loss import GEMLoss
from .sed_triton_loss import SEDLoss

__all__ = [
    "CrossEntropyLoss",
    "GEMLoss", 
    "SEDLoss",
]
