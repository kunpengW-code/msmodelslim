#!/usr/bin/env python3
"""Main-branch W4A8_DYNAMIC dense Linear scheme, backported to 0.30.0.

Faithful transcription of AscendW4A8DynamicLinearMethod from upstream main
commit 2697e7f (#20469, 2026-11-01), with only these adaptations:
- absolute imports (0.30.0 package layout, same as w4a8_dense_port)
- PreparedLinearInput overlap path omitted (not present in 0.30.0 base)
Everything numeric (weight packing, NZ, bit-encoded scale, GMM call,
scale_bias) is verbatim main-branch behavior.

Importing this module registers ("W4A8_DYNAMIC", "linear").
"""
from typing import Any

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size

from vllm_ascend.quantization.methods.base import AscendLinearScheme
from vllm_ascend.quantization.methods.registry import register_scheme
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD, maybe_trans_nz

_MODELSLIM_SCALE_BIAS_PARTITIONS = 16
_INT4_VALUES_PER_BYTE = 2
_INT4_VALUES_PER_INT32 = 8
_INT4_NZ_OUTPUT_ALIGNMENT = 64
_INT4_NZ_INPUT_ALIGNMENT = 16


@register_scheme("W4A8_DYNAMIC", "linear")
class AscendW4A8DynamicLinearMethod(AscendLinearScheme):
    """ModelSlim symmetric INT4 weights and dynamic per-token INT8 inputs.

    Checkpoints pack two output channels into each INT8 element, giving
    weight [N / 2, K], FP32 scales/offsets [N, 1], and scale_bias [N, 1].
    Row-parallel layers store 16 partial scale_bias sums along K. The
    existing Linear adapter shards these partial sums before post-loading.
    """

    act_quant_type: torch.dtype = torch.int8

    def __init__(self):
        quant_config = get_current_vllm_config().quant_config
        if quant_config.get_name() != ASCEND_QUANTIZATION_METHOD:
            raise ValueError("W4A8 INT Linear currently requires ModelSlim checkpoints.")
        if quant_config.quant_description.get("group_size", 0) > 0:
            raise ValueError("W4A8 INT Linear supports per-channel weights, not per-group weights.")

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        if output_size % _INT4_VALUES_PER_INT32:
            raise ValueError("W4A8 INT Linear output partition size must be divisible by 8.")
        return {
            "weight": torch.empty(output_size // _INT4_VALUES_PER_BYTE, input_size, dtype=torch.int8),
            "_packed_dim": 0,
            "_packed_factor": _INT4_VALUES_PER_BYTE,
        }

    def get_perchannel_param(self, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        return {
            "weight_scale": torch.empty(output_size, 1, dtype=torch.float32),
            "weight_offset": torch.empty(output_size, 1, dtype=torch.float32),
        }

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        if layer_type == "row":
            tp_size = get_tensor_model_parallel_world_size()
            if _MODELSLIM_SCALE_BIAS_PARTITIONS % tp_size:
                raise ValueError("ModelSlim W4A8 INT RowParallel Linear TP size must divide 16.")
            return {
                "scale_bias": torch.empty(
                    output_size, _MODELSLIM_SCALE_BIAS_PARTITIONS // tp_size, dtype=torch.float32
                ),
                "_input_dim": 1,
            }
        return {"scale_bias": torch.empty(output_size, 1, dtype=torch.float32)}

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        input_shape = x.shape
        output_dtype = x.dtype
        if x.numel() == 0:
            return x.new_empty((*input_shape[:-1], layer.weight_scale.numel()))
        quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(
            x.reshape(-1, input_shape[-1]), dst_type=self.act_quant_type
        )

        output_shape = (*input_shape[:-1], layer.weight_scale.numel())
        if quantized_x.numel() == 0:
            return torch.empty(output_shape, dtype=output_dtype, device=quantized_x.device)
        # Use the per-channel A8W4 GMM interface for a single Linear group.
        group_list = torch.full((1,), quantized_x.shape[0], dtype=torch.int64, device=quantized_x.device)
        output = torch_npu.npu_grouped_matmul(
            x=[quantized_x],
            weight=[layer.weight],
            scale=[layer.weight_scale.reshape(1, 1, -1)],
            bias=[layer.scale_bias.reshape(1, -1)],
            per_token_scale=[pertoken_scale],
            group_list=group_list,
            group_list_type=1,
            group_type=0,
            split_item=2,
            output_dtype=output_dtype,
        )[0]
        # The GMM bias compensates the A8W4 activation shift; a model bias is
        # independent of the per-token scale and must be added afterwards.
        if bias is not None:
            output.add_(bias)
        return output.reshape(output_shape)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.weight.device.type == "meta":
            return
        weight_offset = layer.weight_offset.detach().cpu()
        if torch.any(weight_offset != 0).item():
            raise ValueError("W4A8 INT Linear only supports symmetric weight offsets.")
        # Materialize the 3D single-group tensor before NZ conversion. A view
        # retains the 2D storage descriptor and triggers an unsupported TransData.
        packed_weight = layer.weight.data.transpose(0, 1).unsqueeze(0).contiguous()
        # INT4 NZ requires K divisible by 16 and N divisible by 64. Keep ND
        # for smaller TP partitions.
        if (
            packed_weight.shape[1] % _INT4_NZ_INPUT_ALIGNMENT == 0
            and layer.weight_scale.numel() % _INT4_NZ_OUTPUT_ALIGNMENT == 0
        ):
            packed_weight = maybe_trans_nz(packed_weight)
        layer.weight.data = packed_weight.view(torch.int32)
        layer.weight_scale_fp32 = layer.weight_scale.data.flatten().to(torch.float32).contiguous()
        # CANN A8W4 consumes the FP32 scale bits in the low half of INT64.
        layer.weight_scale.data = layer.weight_scale_fp32.view(torch.int32).to(torch.int64)
        layer.weight_offset.data = layer.weight_offset.data.flatten()
        layer.scale_bias.data = layer.scale_bias.data.sum(dim=1).contiguous()
