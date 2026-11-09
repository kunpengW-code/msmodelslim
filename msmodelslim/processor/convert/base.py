#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Base class for convert-only processors (data-free, no forward).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import IRKind, LossLevel


class BaseConvertProcessor(ABC):
    """
    Convenience base implementing ``IIRTransformProcessor`` flags.

    Subclasses implement ``transform`` only; register with ``IRRouter.register_processor``.
    """

    name: str
    src_ir: IRKind
    dst_ir: IRKind
    requires_forward: bool = False
    requires_calibration: bool = False
    loss_level: str = LossLevel.LOSSY.value

    @abstractmethod
    def transform(self, module: nn.Module, context: ConvertContext) -> nn.Module: ...
