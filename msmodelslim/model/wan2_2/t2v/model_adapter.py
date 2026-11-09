#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.
Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

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

# pylint: disable=consider-using-from-import,attribute-defined-outside-init,relative-beyond-top-level

import logging
from typing import Any, Dict, List, Literal, Optional

import torch.nn as nn
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from msmodelslim.core.const import DeviceType
from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VlmCalibSample
from msmodelslim.utils.exception import SchemaValidateError

from ..base_model_adapter import Wan2_2BaseModelAdapter  # pylint: disable=relative-beyond-top-level


class Wan2_2T2VModelAdapter(Wan2_2BaseModelAdapter):  # pylint: disable=too-many-ancestors
    """Wan2.2 text-to-video model adapter (Wan2.2-T2V-A14B)."""

    scene_task = "t2v-A14B"

    class Wan2_2T2VInferenceConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")
        size: Optional[str] = "1280*720"
        frame_num: Optional[int] = 81
        sample_shift: Optional[float] = 12.0
        sample_solver: Optional[Literal["unipc", "dpm++"]] = "unipc"
        sample_steps: Optional[int] = 40
        # 与 generate.py --sample_guide_scale (type=float) 一致；省略则由 WAN_CONFIGS 回填双专家默认值。
        sample_guide_scale: Optional[float] = None
        base_seed: Optional[int] = None
        offload_model: Optional[bool] = None
        task: Optional[str] = "t2v-A14B"
        convert_model_dtype: Optional[bool] = None

        @field_validator("sample_guide_scale", mode="before")
        @classmethod
        def _reject_non_cli_guide_scale(cls, value: Any) -> Any:
            if isinstance(value, (tuple, list)):
                raise ValueError(
                    "sample_guide_scale must be a single float in inference_config "
                    "(same as generate.py --sample_guide_scale). "
                    "Omit this field to use WAN_CONFIGS defaults for low/high noise experts.",
                )
            return value

        @model_validator(mode='before')
        @classmethod
        def _reject_mismatched_task(cls, data: Any) -> Any:
            expected = "t2v-A14B"
            if isinstance(data, dict) and data.get("task") not in (None, expected):
                raise ValueError(
                    f"task {data['task']!r} does not match this adapter (expected {expected!r}). "
                    "Use model_type Wan2.2-T2V-A14B instead of setting task in YAML.",
                )
            return data

    def get_inference_config_class(self):
        return self.Wan2_2T2VInferenceConfig

    def validate_calib_samples(self, samples: List[VlmCalibSample]) -> List[VlmCalibSample]:
        for idx, sample in enumerate(samples):
            if not isinstance(sample.text, str) or not sample.text.strip():
                raise SchemaValidateError(
                    f"wan2_2 t2v sample[{idx}] requires non-empty text",
                    action="Provide text in dataset entries (index.jsonl / VlmCalibSample.text).",
                )
            if sample.image is not None:
                raise SchemaValidateError(
                    f"wan2_2 t2v sample[{idx}] must not include image",
                    action="Remove image field from dataset entries for T2V.",
                )
        return samples

    def init_model(self, device: DeviceType = DeviceType.NPU) -> Dict[str, nn.Module]:
        _ = device
        self._load_pipeline()
        experts = {
            "low_noise_model": self.low_noise_model,
            "high_noise_model": self.high_noise_model,
        }
        self._bind_expert_sub_adapters(experts)
        return experts

    def quantization_context(self):
        # 双专家 DiT：进入 low/high 的 autocast + no_sync 上下文
        return self._quantization_context_with_no_sync(self.low_noise_model, self.high_noise_model)

    def _generate_video(
        self,
        prompt: str,
        image_path: Optional[str],
        inference_config: Any = None,
    ) -> None:
        from wan.configs import SIZE_CONFIGS

        # T2V 固定纯文本；image 由 validate_calib_samples 保证不存在。
        if image_path is not None:
            raise SchemaValidateError(
                "T2V calibration does not accept image input.",
                action="Use text-only VlmCalibSample entries for Wan2.2-T2V-A14B.",
            )

        self.wan_t2v.generate(
            prompt,
            size=SIZE_CONFIGS[self._runtime_value(inference_config, "size")],
            frame_num=self._runtime_value(inference_config, "frame_num"),
            shift=self._runtime_value(inference_config, "sample_shift"),
            sample_solver=self._runtime_value(inference_config, "sample_solver"),
            sampling_steps=self._runtime_value(inference_config, "sample_steps"),
            guide_scale=self._runtime_value(inference_config, "sample_guide_scale"),
            seed=self._runtime_value(inference_config, "base_seed"),
            offload_model=self._runtime_value(inference_config, "offload_model"),
        )

    def _build_wan_pipeline(self, args, cfg, device, rank) -> None:
        import wan

        logging.info("Creating WanT2V pipeline.")
        self.wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            quant_dit_path=args.quant_dit_path,
            device_id=device,
            rank=rank,
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        self.low_noise_model = self.wan_t2v.low_noise_model
        self.high_noise_model = self.wan_t2v.high_noise_model
        self._setup_wan_dit_runtime(args, self.low_noise_model, self.high_noise_model)
