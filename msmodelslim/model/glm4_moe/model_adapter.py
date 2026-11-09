#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

"""
参考 https://www.modelscope.cn/models/Eco-Tech/GLM-4.7-W8A8 中采用V0框架的量化方法，
可以发现GLM4.7 MOE采用的模型适配器与Qwen3 MOE实现一致，因此GLM-4.7 MOE 按V1框架实现的模型适配器与qwen_moe/model_adapter.py 一致

"""

from typing import List, Any, Generator
from torch import nn
from transformers import PreTrainedTokenizerBase
from msmodelslim.core.const import DeviceType
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.processor.quarot import QuaRotInterface
from msmodelslim.model.interface_hub import (ModelSlimPipelineInterfaceV1, ModelInfoInterface,
    IterSmoothInterface, FlexSmoothQuantInterface)
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func, transformers_generated_forward_func
from msmodelslim.model.common.transformers import TransformersModel
from msmodelslim.utils.security.model import SafeGenerator
from msmodelslim.utils.logging import logger_setter


@logger_setter()
class GLM4MoeModelAdapter(
    TransformersModel,  # 继承自BaseModelAdapter，基于Transformers模型通用特性和行为简化接口实现
    ModelSlimPipelineInterfaceV1,  # 必要，服务于量化调度
    ModelInfoInterface,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    QuaRotInterface,
):
    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'glm4_moe'

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:  # 描述校准集转化为批量输入
        return self._get_tokenized_data(dataset, device)  # TransformersModel已基于Transformers模型特点给出默认实现

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:  # 描述如何初始化模型
        return self._load_model(device)  # TransformersModel已基于Transformers模型特点给出默认实现

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:  # 描述如何将模型分段，必须与模型结构匹配
        # msmodelslim/model/common/layer_wise_forward.py已给出基于DecoderLayer类分段的默认实现
        yield from generated_decoder_layer_visit_func(model)

    def generate_model_forward(self, model: nn.Module, inputs: Any,
                               ) -> Generator[ProcessRequest, Any, None]:  # 描述如何将模型前向过程分段，必须与模型前向过程匹配
        # msmodelslim/model/common/layer_wise_forward.py已给出基于DecoderLayer类前向过程的默认实现
        yield from transformers_generated_forward_func(model,
                                                       inputs)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:  # 描述是否禁用 KVCache，可减少显存
        return self._enable_kv_cache(model, need_kv_cache)  #

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        for layer_idx in range(self.config.num_hidden_layers):
            # Norm-Linear融合的映射配置：输入层归一化到QKV投影
            norm_linear_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",  # 第一个LayerNorm
                targets=[f"model.layers.{layer_idx}.self_attn.k_proj",
                         f"model.layers.{layer_idx}.self_attn.q_proj",
                         f"model.layers.{layer_idx}.self_attn.v_proj"]  # 注意力层的QKV投影
            )

            # OV融合的映射配置（QKV到输出投影）
            ov_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.v_proj",  # V投影层
                targets=[f"model.layers.{layer_idx}.self_attn.o_proj"]  # 输出投影层
            )

            # 为当前layer添加2个配置
            adapter_config.extend([
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=norm_linear_mapping_config
                ),
                AdapterConfig(
                    subgraph_type="ov",
                    mapping=ov_mapping_config,
                    extra_config={
                        'group_method': 'max'
                    }
                ),
            ])
        return adapter_config

    def get_ln_fuse_map(self):
        return {}, glm4_moe_get_ln_fuse_map(self.config)

    def get_bake_names(self):
        return [], []

    def get_rotate_map(self, block_size):
        pre_run, rot_pairs, _, _ = glm4_moe_get_rotate_map(self.config, block_size)
        return [pre_run], [pair for pair in rot_pairs.values()]

    def _load_tokenizer(self, trust_remote_code=False) -> PreTrainedTokenizerBase:
        return SafeGenerator.get_tokenizer_from_pretrained(
            model_path=str(self.model_path),
            legacy=False,
            trust_remote_code=trust_remote_code)


def glm4_moe_get_ln_fuse_map(config):
    # for quarot rotate interface
    ln_linear_map = {}
    for layer_idx in range(config.num_hidden_layers):
        ln_linear_map[f"model.layers.{layer_idx}.input_layernorm"] = [
            f"model.layers.{layer_idx}.self_attn.q_proj",
            f"model.layers.{layer_idx}.self_attn.k_proj",
            f"model.layers.{layer_idx}.self_attn.v_proj"
        ]

        # routed experts
        ln_linear_map[f"model.layers.{layer_idx}.post_attention_layernorm"] = [
            f"model.layers.{layer_idx}.mlp.experts.{i}.{proj}"
            for proj in ["gate_proj", "up_proj"]
            for i in range(config.num_experts)
        ]
        # expert gate
        ln_linear_map[f"model.layers.{layer_idx}.post_attention_layernorm"] += [
            f"model.layers.{layer_idx}.mlp.gate"
        ]
    ln_linear_map["model.norm"] = ['lm_head']
    return ln_linear_map


def glm4_moe_get_rotate_map(config, block_size):
    rot = QuaRotInterface.get_rotate_command(
        size=config.hidden_size,
        mode=QuaRotInterface.QuaRotMode.HADAMARD,
        block_size=block_size,
    )
    rot_uv = QuaRotInterface.get_rotate_command(
        size=config.head_dim,
        mode=QuaRotInterface.QuaRotMode.BLOCK_HADAMARD_SHIFTED,
        block_size=block_size,
    )
    # pre run
    left_rot = {}
    right_rot = {}
    # embedding weight is transposed, right is output channel
    right_rot[f"model.embed_tokens"] = rot
    pre_run = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)
    rot_pairs = {}
    # rot
    left_rot = {}
    right_rot = {}
    right_rot[f"lm_head"] = rot
    for layer_idx in range(config.num_hidden_layers):
        right_rot[f"model.layers.{layer_idx}.self_attn.q_proj"] = rot
        right_rot[f"model.layers.{layer_idx}.self_attn.k_proj"] = rot
        right_rot[f"model.layers.{layer_idx}.self_attn.v_proj"] = rot
        left_rot[f"model.layers.{layer_idx}.self_attn.o_proj"] = rot
        # routed experts
        for i in range(config.num_experts):
            right_rot[f"model.layers.{layer_idx}.mlp.experts.{i}.gate_proj"] = rot
            right_rot[f"model.layers.{layer_idx}.mlp.experts.{i}.up_proj"] = rot
            left_rot[f"model.layers.{layer_idx}.mlp.experts.{i}.down_proj"] = rot
        # expert gate
        right_rot[f"model.layers.{layer_idx}.mlp.gate"] = rot
    rot_pairs['rot'] = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)

    # rot_uv
    left_rot_uv = {}
    right_rot_uv = {}
    for layer_idx in range(config.num_hidden_layers):
        left_rot_uv[f"model.layers.{layer_idx}.self_attn.v_proj"] = rot_uv
        right_rot_uv[f"model.layers.{layer_idx}.self_attn.o_proj"] = rot_uv
    rot_pairs["rot_uv"] = QuaRotInterface.RotatePair(left_rot=left_rot_uv, right_rot=right_rot_uv)

    return pre_run, rot_pairs, rot, rot_uv