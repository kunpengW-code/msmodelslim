#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import os.path
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Dict, Generator, List, Optional, Tuple, cast
from unittest.mock import patch

import torch
from safetensors import safe_open
from transformers.cache_utils import DynamicCache
from torch import distributed as dist, nn
from tqdm import tqdm

from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.core.quant_service.modelslim_v1.save.interface import (
    AscendV1SaveInterface,
)
from msmodelslim.model.common.layer_wise_forward import (
    TransformersForwardBreak,
    generated_decoder_layer_visit_func,
)
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.security import (
    MAX_READ_FILE_SIZE_32G,
    get_valid_read_path,
    json_safe_load,
)
from msmodelslim.utils.security.model import SafeGenerator

from ..common.transformers import TransformersModel
from ..interface_hub import (
    FlexSmoothQuantInterface,
    IterSmoothInterface,
    ModelSlimPipelineInterfaceV1,
    QuaRotInterface,
)


_logger = get_logger()

WEIGHT_SCALE_INV_SUFFIX = ".weight_scale_inv"
FP8_BLOCK_SIZE_DEFAULT = 128
LAZY_LOAD_ATTR = "_msmodelslim_is_loaded"


@contextmanager
def default_dtype(dtype):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def _weight_dequant_block(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = FP8_BLOCK_SIZE_DEFAULT,
) -> torch.Tensor:
    """Dequantize an FP8 (e4m3) weight using block-wise scales to bfloat16.

    Mirrors deepseek_v3.convert_fp8_to_bf16.weight_dequant.
    """
    m, n = weight.shape
    weight_fp32 = weight.to(torch.float32)
    scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    scale_expanded = scale_expanded[:m, :n]
    weight_fp32 = weight_fp32 * scale_expanded
    return weight_fp32.to(torch.bfloat16)


@logger_setter("msmodelslim.model.minimax_m2")
class MiniMaxM2ModelAdapter(
    TransformersModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    FlexSmoothQuantInterface,
    IterSmoothInterface,
    QuaRotInterface,
    AscendV1SaveInterface,
):  # pylint: disable=too-many-ancestors
    """Model adapter for MiniMax-M2 quantization.

    Architecture summary:
    - 62 standard decoder layers (no MTP integration in this adapter).
    - Per layer: self_attn (q/k/v/o_proj, optional q_norm/k_norm) + block_sparse_moe
      (gate Linear, 256 experts each with w1/w2/w3, e_score_correction_bias buffer)
      + input_layernorm + post_attention_layernorm.
    - Checkpoint stores attention linears and expert w1/w2/w3 in FP8 (e4m3, 128x128
      block scale); gate / e_score_correction_bias / lm_head are not quantized.
    - Weights are loaded layer-by-layer to CPU to bound peak memory.
    - block_sparse_moe.gate.weight and block_sparse_moe.e_score_correction_bias
      are kept in float32 throughout.
    """

    def get_model_pedigree(self) -> str:
        return "minimax_m2"

    def get_model_type(self) -> str:
        return self.model_type

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device=device, padding=False)

    # ------------------------------------------------------------------ #
    # dtype preservation helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_minimax_moe_block(module: nn.Module) -> bool:
        return module.__class__.__name__ == "MiniMaxM2SparseMoeBlock"

    @staticmethod
    def _register_router_bias_parameter(moe: nn.Module) -> None:
        """Promote ``e_score_correction_bias`` from a buffer to an fp32 Parameter.

        ``MiniMaxM2SparseMoeBlock`` registers ``e_score_correction_bias`` via
        ``register_buffer``. modelslim's checkpoint writer only persists
        ``named_parameters``, so the buffer would be silently dropped from the
        saved weights. Re-registering it as ``nn.Parameter`` ensures it is part
        of ``state_dict`` and survives the save pipeline. Already-Parameter
        tensors are only re-cast to fp32 in place.
        """
        bias = getattr(moe, "e_score_correction_bias", None)
        if not isinstance(bias, torch.Tensor):
            return

        is_already_parameter = "e_score_correction_bias" in dict(moe.named_parameters(recurse=False))
        new_bias = bias.detach().to(dtype=torch.float32)

        if is_already_parameter:
            # Replace the existing Parameter to keep dtype/storage consistent.
            moe.e_score_correction_bias = nn.Parameter(new_bias, requires_grad=bias.requires_grad)
            return

        # Drop the buffer registration before adding the Parameter, otherwise
        # ``register_parameter`` raises because the name is already taken.
        try:
            delattr(moe, "e_score_correction_bias")
        except AttributeError:
            pass
        moe.register_parameter(
            "e_score_correction_bias",
            nn.Parameter(new_bias, requires_grad=bias.requires_grad),
        )

    def _preserve_router_fp32(self, root: nn.Module) -> None:
        """Cast block_sparse_moe.gate and e_score_correction_bias to float32.

        Done before ``load_state_dict`` so that bf16 tensors from the checkpoint
        are upcast in place via ``Tensor.copy_``. The bias keeps its original
        registration (buffer or Parameter) here; promotion to ``nn.Parameter``
        happens at save time in ``ascendv1_save_module_preprocess``.
        """
        for _, moe in root.named_modules():
            if not self._is_minimax_moe_block(moe):
                continue
            if hasattr(moe, "gate") and isinstance(moe.gate, nn.Linear):
                moe.gate.to(dtype=torch.float32)
            bias = getattr(moe, "e_score_correction_bias", None)
            if isinstance(bias, torch.Tensor) and bias.dtype != torch.float32:
                # Re-assignment to an existing buffer/parameter name preserves
                # the registration kind (buffer stays buffer, parameter stays
                # parameter) while updating the underlying tensor dtype.
                moe.e_score_correction_bias = bias.detach().to(dtype=torch.float32)

    @staticmethod
    def _get_transformer_body(model: nn.Module) -> nn.Module:
        return cast(nn.Module, getattr(model, "model"))

    @staticmethod
    def _set_decoder_loaded(decoder: nn.Module, loaded: bool) -> None:
        setattr(decoder, LAZY_LOAD_ATTR, loaded)

    @staticmethod
    def _is_decoder_loaded(decoder: nn.Module) -> bool:
        return bool(getattr(decoder, LAZY_LOAD_ATTR, True))

    def _build_decoder_skeleton(self, template: nn.Module, idx: int) -> nn.Module:
        with torch.device("meta"):
            decoder = template.__class__(config=self.config, layer_idx=idx)
        self._set_decoder_loaded(decoder, False)
        return decoder

    def _populate_decoder_skeletons(self, model: nn.Module, num_layers: int) -> None:
        model_body = self._get_transformer_body(model)
        module_list = cast(nn.ModuleList, getattr(model_body, "layers"))
        template = module_list[0]
        self._set_decoder_loaded(template, True)

        for idx in range(1, num_layers):
            if len(module_list) > idx:
                self._set_decoder_loaded(module_list[idx], False)
                continue
            module_list.append(self._build_decoder_skeleton(template, idx))

    # ------------------------------------------------------------------ #
    # model loading
    # ------------------------------------------------------------------ #

    def _normalize_config_for_forward(self) -> None:
        """Strip runtime-quant settings so HF can build a non-quant fwd model."""
        self.config.use_cache = False
        if getattr(self.config, "_attn_implementation", None) in (None, "eager"):
            self.config._attn_implementation = "sdpa"
        # Drop FP8 quantization metadata so transformers does not try to load
        # a quantized model. We dequantize manually in _get_state_dict.
        if hasattr(self.config, "quantization_config"):
            try:
                delattr(self.config, "quantization_config")
            except AttributeError:
                self.config.quantization_config = None

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        with default_dtype(torch.bfloat16):
            if not hasattr(self.config, "num_hidden_layers"):
                raise InvalidModelError(
                    "Config missing 'num_hidden_layers' attribute",
                    action="Please ensure the model config is valid for MiniMax-M2",
                )

            origin_layers = self.config.num_hidden_layers
            if _logger:
                _logger.info("MiniMax-M2 model with %s layers totally", origin_layers)

            self._normalize_config_for_forward()

            # Load only one layer as template; remaining layers are materialized lazily.
            self.config.num_hidden_layers = 1
            model = SafeGenerator.get_model_from_pretrained(
                model_path=str(self.model_path),
                config=self.config,
                trust_remote_code=self.trust_remote_code,
                device_map="cpu",
                torch_dtype="auto",
            )

            # Restore real layer count for downstream traversal.
            self.config.num_hidden_layers = origin_layers

            # Preserve fp32 dtype on router pieces in the template before loading weights.
            self._preserve_router_fp32(model)

            # Load weights for the template (layer 0 + non-layer params).
            # Use assign=True so that the dequantized bf16 (and fp32 router)
            # tensors fully replace the parameter slots created by
            # from_pretrained. Without assign=True, layer 0's attention and
            # expert weights stay in their native float8_e4m3fn dtype because
            # default load_state_dict goes through ``param.data.copy_`` which
            # preserves the existing parameter dtype.
            state_dict = self._get_state_dict(model)
            model.load_state_dict(state_dict, assign=True)

            # Re-apply router fp32 enforcement in case ``assign=True`` swapped
            # the buffer/parameter back to a non-fp32 tensor for any router.
            self._preserve_router_fp32(model)
            self._populate_decoder_skeletons(model, origin_layers)

            model.eval()
            if _logger:
                _logger.info(
                    "Created MiniMax-M2 template with 1/%s loaded layers; "
                    "remaining decoder shells will load weights on-demand.",
                    origin_layers,
                )
            return model

    # ------------------------------------------------------------------ #
    # decoder traversal
    # ------------------------------------------------------------------ #

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        return generated_decoder_layer_visit_func(
            model,
            transformer_blocks=self._generate_decoder_layers(model),
        )

    def _generate_decoder_layers(self, model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
        for idx in range(self.config.num_hidden_layers):
            name = f"model.layers.{idx}"
            decoder = self._load_decoder_if_not_exist(model, name=name, idx=idx)
            yield name, decoder

    def _load_decoder_if_not_exist(self, model: nn.Module, name: str, idx: int) -> nn.Module:
        decoder: Optional[nn.Module] = None
        try:
            decoder = model.get_submodule(name)
            if self._is_decoder_loaded(decoder):
                return decoder
        except AttributeError:
            decoder = None

        with (
            patch.object(nn.Linear, "reset_parameters", lambda self: None),
            default_dtype(torch.bfloat16),
        ):
            if _logger:
                _logger.info("Creating decoder layer %s", idx)

            model_body = self._get_transformer_body(model)
            module_list = cast(nn.ModuleList, getattr(model_body, "layers"))
            if decoder is None:
                template = module_list[0]
                decoder = template.__class__(config=self.config, layer_idx=idx)
                if len(module_list) <= idx:
                    module_list.append(decoder)
                else:
                    module_list[idx] = decoder

            self._preserve_router_fp32(decoder)

            state_dict = self._get_state_dict(decoder, prefix=name)
            decoder.load_state_dict(state_dict, assign=True)
            self._preserve_router_fp32(decoder)
            decoder.eval()
            self._set_decoder_loaded(decoder, True)

            if _logger:
                _logger.info("Created decoder layer %s successfully", idx)

        return decoder

    # ------------------------------------------------------------------ #
    # forward pipeline
    # ------------------------------------------------------------------ #

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        first_block_input: Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]] = None

        def break_hook(_module, hook_args, hook_kwargs):
            nonlocal first_block_input
            first_block_input = (hook_args, hook_kwargs)
            raise TransformersForwardBreak()

        model_body = self._get_transformer_body(model)
        layer_0 = cast(nn.ModuleList, getattr(model_body, "layers"))[0]
        remove_handler = layer_0.register_forward_pre_hook(break_hook, with_kwargs=True, prepend=True)

        try:
            if isinstance(inputs, (list, tuple)):
                model(inputs[0])
            elif isinstance(inputs, dict):
                model(**inputs)
            else:
                model(inputs)
        except TransformersForwardBreak:
            pass
        except Exception as e:
            if _logger:
                _logger.error("Error during MiniMax-M2 forward pass: %s", e)
            raise
        finally:
            remove_handler.remove()

        if first_block_input is None:
            raise InvalidModelError(
                "Cannot get first block input.",
                action="Please check the model and input",
            )

        if dist.is_initialized():
            dist.barrier()

        args, kwargs = first_block_input
        kwargs = dict(kwargs)

        use_cache = bool(kwargs.pop("use_cache", False))
        if use_cache and kwargs.get("past_key_values") is None:
            kwargs["past_key_values"] = DynamicCache()

        current_inputs = (args, kwargs)
        for name, block in self._generate_decoder_layers(model):
            args, kwargs = current_inputs
            hidden_states = yield ProcessRequest(name, block, args, kwargs)
            current_inputs = ((hidden_states,), kwargs)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        def pre_forward_hook(_module, args, kwargs):
            kwargs["use_cache"] = need_kv_cache
            return args, kwargs

        model_body = self._get_transformer_body(model)
        model_body.register_forward_pre_hook(pre_forward_hook, with_kwargs=True)

    # ------------------------------------------------------------------ #
    # subgraph mapping
    # ------------------------------------------------------------------ #

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        """Build smoothing/fusion subgraphs from the MiniMaxM2 module layout.

        Per-layer mappings (gate / e_score_correction_bias are skipped because
        they are not quantized and must remain fp32):
        - input_layernorm -> q_proj, k_proj, v_proj                 (norm-linear)
        - v_proj -> o_proj                                          (ov)
        - per expert: w3 -> w2                                      (up-down)
        """
        adapter_config: List[AdapterConfig] = []

        num_layers = getattr(self.config, "num_hidden_layers", 62)
        num_experts = getattr(self.config, "num_local_experts", 256)

        for layer_idx in range(num_layers):
            input_norm_mapping = MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",
                targets=[
                    f"model.layers.{layer_idx}.self_attn.q_proj",
                    f"model.layers.{layer_idx}.self_attn.k_proj",
                    f"model.layers.{layer_idx}.self_attn.v_proj",
                ],
            )
            adapter_config.append(AdapterConfig(subgraph_type="norm-linear", mapping=input_norm_mapping))

            ov_mapping = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.v_proj",
                targets=[f"model.layers.{layer_idx}.self_attn.o_proj"],
            )
            adapter_config.append(
                AdapterConfig(
                    subgraph_type="ov",
                    mapping=ov_mapping,
                    extra_config={"group_method": "max"},
                )
            )

            for expert_idx in range(num_experts):
                # MiniMaxM2MLP: out = w2(act(w1(x)) * w3(x))
                # Treat w3 as the up-projection and w2 as the down-projection.
                up_down_mapping = MappingConfig(
                    source=f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3",
                    targets=[f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2"],
                )
                adapter_config.append(AdapterConfig(subgraph_type="up-down", mapping=up_down_mapping))

        return adapter_config

    # ------------------------------------------------------------------ #
    # QuaRot mapping
    # ------------------------------------------------------------------ #

    def get_ln_fuse_map(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        return {}, self._get_quarot_ln_fuse_map()

    def get_bake_names(self) -> Tuple[List[str], List[str]]:
        return [], []

    def get_rotate_map(
        self,
        block_size: int,
    ) -> Tuple[List[QuaRotInterface.RotatePair], List[QuaRotInterface.RotatePair]]:
        hidden_size = getattr(self.config, "hidden_size")
        head_dim = self._get_head_dim()

        rot = QuaRotInterface.get_rotate_command(
            size=hidden_size,
            block_size=block_size,
            mode=QuaRotInterface.QuaRotMode.HADAMARD,
        )
        rot_uv = QuaRotInterface.get_rotate_command(
            size=head_dim,
            block_size=block_size,
            mode=QuaRotInterface.QuaRotMode.HADAMARD,
        )

        pre_run = QuaRotInterface.RotatePair(
            left_rot={},
            right_rot={"model.embed_tokens": rot},
        )

        left_rot: Dict[str, torch.Tensor] = {}
        right_rot: Dict[str, torch.Tensor] = {"lm_head": rot}
        left_rot_uv: Dict[str, torch.Tensor] = {}
        right_rot_uv: Dict[str, torch.Tensor] = {}

        num_layers = getattr(self.config, "num_hidden_layers", 62)
        num_experts = getattr(self.config, "num_local_experts", 256)

        for layer_idx in range(num_layers):
            layer_prefix = f"model.layers.{layer_idx}"
            right_rot.update(
                {
                    f"{layer_prefix}.self_attn.q_proj": rot,
                    f"{layer_prefix}.self_attn.k_proj": rot,
                    f"{layer_prefix}.self_attn.v_proj": rot,
                    f"{layer_prefix}.block_sparse_moe.gate": rot,
                }
            )
            left_rot[f"{layer_prefix}.self_attn.o_proj"] = rot

            for expert_idx in range(num_experts):
                expert_prefix = f"{layer_prefix}.block_sparse_moe.experts.{expert_idx}"
                right_rot[f"{expert_prefix}.w1"] = rot
                right_rot[f"{expert_prefix}.w3"] = rot
                left_rot[f"{expert_prefix}.w2"] = rot

            left_rot_uv[f"{layer_prefix}.self_attn.v_proj"] = rot_uv
            right_rot_uv[f"{layer_prefix}.self_attn.o_proj"] = rot_uv

        rotate_pairs = [
            QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot),
            QuaRotInterface.RotatePair(left_rot=left_rot_uv, right_rot=right_rot_uv),
        ]
        return [pre_run], rotate_pairs

    def _get_quarot_ln_fuse_map(self) -> Dict[str, List[str]]:
        ln_linear_map: Dict[str, List[str]] = {}
        num_layers = getattr(self.config, "num_hidden_layers", 62)
        num_experts = getattr(self.config, "num_local_experts", 256)

        for layer_idx in range(num_layers):
            layer_prefix = f"model.layers.{layer_idx}"
            ln_linear_map[f"{layer_prefix}.input_layernorm"] = [
                f"{layer_prefix}.self_attn.q_proj",
                f"{layer_prefix}.self_attn.k_proj",
                f"{layer_prefix}.self_attn.v_proj",
            ]

            moe_targets = [f"{layer_prefix}.block_sparse_moe.gate"]
            for expert_idx in range(num_experts):
                expert_prefix = f"{layer_prefix}.block_sparse_moe.experts.{expert_idx}"
                moe_targets.extend([f"{expert_prefix}.w1", f"{expert_prefix}.w3"])
            ln_linear_map[f"{layer_prefix}.post_attention_layernorm"] = moe_targets

        ln_linear_map["model.norm"] = ["lm_head"]
        return ln_linear_map

    def _get_head_dim(self) -> int:
        if hasattr(self.config, "head_dim"):
            return int(self.config.head_dim)
        return int(self.config.hidden_size // self.config.num_attention_heads)

    # ------------------------------------------------------------------ #
    # weight loading (prefix-scoped, FP8 -> BF16 dequant)
    # ------------------------------------------------------------------ #

    @lru_cache(maxsize=1)
    def _get_weight_map(self) -> Dict[str, str]:
        model_index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        model_index = json_safe_load(model_index_path)
        return model_index.get("weight_map", {})

    @lru_cache(maxsize=1)
    def _get_fp8_block_size(self) -> int:
        # Read once from the original config.json since we strip quantization_config
        # off the in-memory PretrainedConfig in _normalize_config_for_forward.
        config_path = os.path.join(self.model_path, "config.json")
        try:
            raw = json_safe_load(config_path)
        except Exception:  # pragma: no cover - defensive
            return FP8_BLOCK_SIZE_DEFAULT
        qcfg = raw.get("quantization_config") or {}
        block = qcfg.get("weight_block_size")
        if isinstance(block, (list, tuple)) and len(block) >= 1:
            # Square block (e.g. [128, 128]); use first dim.
            return int(block[0])
        return FP8_BLOCK_SIZE_DEFAULT

    def _get_state_dict(
        self,
        module: nn.Module,
        prefix: str = "",
    ) -> Dict[str, torch.Tensor]:
        """Load a prefix-scoped state dict, dequantizing FP8 weights to bf16.

        For any parameter whose ``<full_name>.weight_scale_inv`` exists in the
        safetensors index, the raw e4m3 weight is multiplied by the per-block
        scale and converted to bf16.
        """
        weight_map = self._get_weight_map()
        block_size = self._get_fp8_block_size()

        names = [name for name, _ in module.named_parameters()]
        names += [name for name, _ in module.named_buffers()]

        # Group required tensors by source safetensors file. Each entry is a
        # list of (local_name, full_name, scale_full_name_or_None) triples.
        groups: Dict[str, List[Tuple[str, str, Optional[str]]]] = defaultdict(list)
        scale_groups: Dict[str, List[str]] = defaultdict(list)

        for name in names:
            full_name = f"{prefix}.{name}" if prefix else name
            if full_name not in weight_map:
                continue
            file_name = weight_map[full_name]
            scale_full_name: Optional[str] = None
            if name.endswith(".weight") or name == "weight":
                candidate = full_name + "_scale_inv"
                # weight_scale_inv key is sibling of .weight, not appended after .weight
                # i.e. "...w1.weight" -> scale "...w1.weight_scale_inv"
                if candidate in weight_map:
                    scale_full_name = candidate
                    scale_file = weight_map[candidate]
                    scale_groups[scale_file].append(scale_full_name)
            groups[file_name].append((name, full_name, scale_full_name))

        # Pre-load every needed scale tensor (small) once per file.
        scales: Dict[str, torch.Tensor] = {}
        for file_name, scale_keys in scale_groups.items():
            file_path = os.path.join(self.model_path, file_name)
            file_path = get_valid_read_path(
                file_path,
                extensions="safetensors",
                size_max=MAX_READ_FILE_SIZE_32G,
            )
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in scale_keys:
                    scales[key] = f.get_tensor(key)

        state_dict: Dict[str, torch.Tensor] = {}
        for file_name, items in tqdm(groups.items(), desc=f"Loading {prefix or 'model'}"):
            file_path = os.path.join(self.model_path, file_name)
            file_path = get_valid_read_path(
                file_path,
                extensions="safetensors",
                size_max=MAX_READ_FILE_SIZE_32G,
            )
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for local_name, full_name, scale_full_name in items:
                    tensor = f.get_tensor(full_name)
                    if scale_full_name is not None:
                        scale = scales.get(scale_full_name)
                        if scale is None:
                            # Fallback: scale lives in same shard.
                            scale = f.get_tensor(scale_full_name)
                        tensor = _weight_dequant_block(tensor, scale, block_size=block_size)
                    state_dict[local_name] = tensor

        return state_dict

    # ------------------------------------------------------------------ #
    # save-time hooks
    # ------------------------------------------------------------------ #

    def ascendv1_save_module_preprocess(
        self,
        prefix: str,
        module: nn.Module,
        model: nn.Module,
    ) -> Tuple[str, nn.Module]:
        """Ensure router pieces stay float32 in the saved checkpoint.

        Also promotes ``e_score_correction_bias`` from buffer to ``nn.Parameter``
        so the modelslim save path (parameters-only) does not drop it.
        """
        if not self._is_minimax_moe_block(module):
            return prefix, module

        gate = getattr(module, "gate", None)
        gate_needs_cast = isinstance(gate, nn.Linear) and gate.weight.dtype != torch.float32
        bias = getattr(module, "e_score_correction_bias", None)
        bias_is_buffer = isinstance(bias, torch.Tensor) and "e_score_correction_bias" not in dict(
            module.named_parameters(recurse=False)
        )
        bias_needs_cast = isinstance(bias, torch.Tensor) and bias.dtype != torch.float32

        if not gate_needs_cast and not bias_needs_cast and not bias_is_buffer:
            return prefix, module

        if gate_needs_cast:
            cast(nn.Linear, getattr(module, "gate")).to(dtype=torch.float32)
        if bias_needs_cast or bias_is_buffer:
            self._register_router_bias_parameter(module)
        return prefix, module
