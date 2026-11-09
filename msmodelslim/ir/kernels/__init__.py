#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Offline weight-transform kernels (convert, model scripts, data-free processors).

These are **stateless tensor ops** on checkpoint layouts. They are intentionally
**not** placed under ``core/convert`` (orchestration only) nor ``ir.api`` (QFuncRegistry
quantize/dequantize dispatch for ``QStorage`` / ``QDType``).

Add new families as separate modules, e.g. ``fp8_block.py``, ``int4_packed.py``.
"""

from msmodelslim.ir.kernels.fp8_block import WEIGHT_SCALE_INV_SUFFIX, weight_dequant

__all__ = [
    "WEIGHT_SCALE_INV_SUFFIX",
    "weight_dequant",
]
