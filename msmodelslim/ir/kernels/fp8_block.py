#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
FP8 per-block checkpoint kernels (E4M3 weight + block ``weight_scale_inv``).

Used by ``processor/convert`` (FP8_BLOCK → FLOAT) and legacy ``model/*/convert_fp8_to_bf16`` scripts.
"""

from __future__ import annotations

import torch

# Canonical suffix in HuggingFace / DeepSeek-style FP8 checkpoints.
WEIGHT_SCALE_INV_SUFFIX = ".weight_scale_inv"


def weight_dequant(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Dequantize FP8 block weights to bfloat16.

    Args:
        weight: Quantized weight ``(M, N)``.
        scale: Block scale ``(M // block_size, N // block_size)`` (``weight_scale_inv``).
        block_size: Block size from ``quantization_config.weight_block_size`` (default 128).
    """
    m, n = weight.shape
    weight = weight.to(torch.float32)
    scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    scale_expanded = scale_expanded[:m, :n]
    return (weight * scale_expanded).to(torch.bfloat16)
