#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
IR edge: FLOAT -> W8A8_MXFP8.

Reuses ``LinearQuantizer`` + ``AutoFakeQuantLinear.create`` (same stack as ``linear_quant``).
"""

from __future__ import annotations

import torch
from torch import nn

from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import IRKind, LossLevel
from msmodelslim.core.quantizer.linear import LinearQuantizer, LinearQConfig
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.convert.base import BaseConvertProcessor
from msmodelslim.utils.logging import get_logger

logger = get_logger()


def _materialize_linear(module: nn.Module, context: ConvertContext | None = None) -> nn.Linear | None:
    """将 ``ModelFreeLinear`` 或 ``nn.Linear`` 物化为可量化的 ``nn.Linear``。"""
    from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear

    if isinstance(module, ModelFreeLinear):
        if not module.lazy_initialized and context is not None and context.reader is not None:
            # convert 计算全程 CPU；NPU 设备解析交由 group_runner 的 lazy_init 处理。
            module.lazy_init(context.reader, device="cpu")
        weight = getattr(module, "weight", None)
        if weight is None:
            return None
        if weight.ndim != 2:
            logger.warning(
                "Skip MXFP8 for %s: weight shape %s is not 2D (norm/conv layers are left as FLOAT)",
                module.full_name,
                tuple(weight.shape),
            )
            return None
        bias = getattr(module, "bias", None)
        linear = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        linear.weight = nn.Parameter(weight.detach().to(torch.bfloat16), requires_grad=False)
        if bias is not None:
            linear.bias = nn.Parameter(bias.detach().to(torch.bfloat16), requires_grad=False)
        return linear
    if isinstance(module, nn.Linear):
        return module
    return None


class MxFp8QuantProcessor(BaseConvertProcessor):
    name = "MxFp8QuantProcessor"
    src_ir = IRKind.FLOAT
    dst_ir = IRKind.W8A8_MXFP8
    loss_level = LossLevel.LOSSY.value

    def transform(self, module: nn.Module, context: ConvertContext) -> nn.Module:
        linear = _materialize_linear(module, context)
        if linear is None:
            return module
        qconfig = LinearQConfig(
            act=QConfig(
                dtype=QDType.MXFP8,
                scope=QScope.PER_BLOCK,
                symmetric=True,
                method="minmax",
                ext={"axes": -1},
            ),
            weight=QConfig(
                dtype=QDType.MXFP8,
                scope=QScope.PER_BLOCK,
                symmetric=True,
                method="minmax",
                ext={"axes": -1},
            ),
        )
        quantizer = LinearQuantizer(qconfig)
        quantizer.setup(linear)
        # data-free：权重已在 setup → init_weight 中量化，无需 forward
        return quantizer.deploy()
