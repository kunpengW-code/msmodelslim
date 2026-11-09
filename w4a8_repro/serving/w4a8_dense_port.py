#!/usr/bin/env python3
"""Dense W4A8_DYNAMIC linear scheme for vllm-ascend 0.30.0rc2.

Ported from the archived dense W4A8/C8 adaptation
(/home/y00884608/inferPerformance/Qwen3.6-27B/env/w4a8c8-dense-adapt-20261015/
vllm-ascend/vllm_ascend/quantization/methods/w4a8/w4a8_dense.py, vllm-ascend
v0.27.1rc1 base, author y00884608) to the v0.30.0rc2 API.

Purpose: vllm-ascend 0.30.0rc2 only registers W4A8_DYNAMIC for MoE layers, so
msmodelslim dense W4A8 exports (symmetric per-channel int4 + per-token int8,
group_size=0) fail engine init with
"doesn't support quant_type=W4A8_DYNAMIC for layer_type=linear".

Importing this module registers the missing scheme at runtime. No installed
package files are modified. The kernel unpacks signed int4 to sign-extended
int8 (lossless) and uses npu_quant_matmul (INT8xINT8).

Performance (corrected 2026-10-22): this is the SAME algorithm the 10007
serving tree uses (env/w4a8c8-opt-20261019 w4a8_dense.py — verified by file
diff; only Triton tuning differed). Their measured full C-Eval with it:
196.5s vs W8A8 159.5s (1.23x) on a prefill-dominated workload (answer-only
prompts, 1-5 output tokens). The earlier "~10x slower decode" note here was
an unmeasured estimate, not a benchmark. Tuning now matches the opt tree
exactly (int32 intermediate, '|' compose, BLOCK 8192). The remaining
theoretical cost vs a fused INT4 kernel is the per-apply unpack pass
(~1.5x extra weight traffic), which only matters for long decode-heavy
workloads. A fused dense INT4 GEMM (like the MoE path's fused_experts)
would remove it but is a separate project with numeric-equivalence risk.

Adaptations for 0.30.0rc2:
- absolute imports instead of package-relative
- get_pergroup_param layer_type now receives "others" (not "column") for
  non-row layers (method_adapters.py convention)
- defensive pertoken_scale dim handling mirrored from
  AscendW8A8DynamicLinearMethod
"""
from typing import Any

import torch
import torch_npu
import triton
import triton.language as tl
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size

from vllm_ascend.quantization.methods.base import AscendLinearScheme
from vllm_ascend.quantization.methods.registry import register_scheme


@triton.jit
def _unpack_signed_int4(packed, output, count: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    byte = tl.load(packed + index, index < count, 0).to(tl.int32)
    # Store both signed INT8 values as one contiguous little-endian INT16
    # word. Avoid duplicated/strided byte addressing, which is expensive on
    # the Ascend Triton backend. The arithmetic remains INT8, not INT16.
    low_byte = ((byte & 7) - (byte & 8)) & 255
    high_signed = byte >> 4
    word = low_byte | (high_signed << 8)
    tl.store(output + index, word.to(tl.int16), index < count)


def unpack_signed_int4(packed: torch.Tensor) -> torch.Tensor:
    output = torch.empty((packed.shape[0], packed.shape[1] * 2), device=packed.device, dtype=torch.int8)
    count = packed.numel()
    _unpack_signed_int4[(triton.cdiv(count, 8192),)](packed, output.view(torch.int16), count, 8192)
    return output


@register_scheme("W4A8_DYNAMIC", "linear")
class AscendW4A8DynamicLinearMethod(AscendLinearScheme):
    """msModelSlim symmetric per-channel W4 / per-token signed INT8.

    Hardware arithmetic is INT8 x INT8 after lossless sign extension of
    the packed W4 operand. This is not a fused INT4 x INT8 kernel.
    """

    act_quant_type = torch.int8

    def __init__(self):
        description = get_current_vllm_config().quant_config.quant_description
        if description.get("group_size", 0) != 0:
            raise ValueError("Dense W4A8 supports per-channel group_size=0 only")

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        if output_size % 2:
            raise ValueError("Packed W4 output partition must be even")
        return {
            "weight": torch.empty(output_size // 2, input_size, dtype=torch.int8),
            "_packed_dim": 0,
            "_packed_factor": 2,
        }

    def get_perchannel_param(self, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        return {name: torch.empty(output_size, 1, dtype=torch.float32) for name in ("weight_scale", "weight_offset")}

    def get_pergroup_param(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        layer_type: str = "others",
    ) -> dict[str, Any]:
        tp = get_tensor_model_parallel_world_size()
        if layer_type == "row":
            if 16 % tp:
                raise ValueError("scale_bias requires TP to divide 16")
            return {"scale_bias": torch.empty(output_size, 16 // tp, dtype=torch.float32), "_input_dim": 1}
        return {"scale_bias": torch.empty(output_size, 1, dtype=torch.float32)}

    def process_weights_after_loading(self, layer):
        if torch.any(layer.weight_offset != 0).item():
            raise ValueError("Dense W4A8 currently requires symmetric weight_offset=0")
        if not torch.isfinite(layer.weight_scale).all().item():
            raise ValueError("Non-finite W4 scale")
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_offset.data = layer.weight_offset.data.flatten()
        # scale_bias compensates the native GMM's x-8 transform. Signed
        # integer matmul does not perform that transform, so needs no bias.

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        shape = x.shape
        quantized_x, token_scale = torch_npu.npu_dynamic_quant(x.reshape(-1, shape[-1]), dst_type=self.act_quant_type)
        if token_scale.dim() == 2:
            quantized_x = quantized_x.squeeze(dim=1)
            token_scale = token_scale.squeeze(dim=1)
        weight = unpack_signed_int4(layer.weight)
        output = torch_npu.npu_quant_matmul(
            quantized_x, weight, layer.weight_scale, pertoken_scale=token_scale, bias=bias, output_dtype=x.dtype
        )
        return output.reshape(*shape[:-1], output.shape[-1])
