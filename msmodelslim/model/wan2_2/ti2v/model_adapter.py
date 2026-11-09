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

# pylint: disable=consider-using-from-import,attribute-defined-outside-init,relative-beyond-top-level,signature-differs

import logging
from typing import Any, Dict, List, Literal, Optional

import torch.nn as nn
from pydantic import BaseModel, ConfigDict, model_validator

from msmodelslim.core.const import DeviceType
from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VlmCalibSample
from msmodelslim.utils.exception import SchemaValidateError

from ..base_model_adapter import Wan2_2BaseModelAdapter  # pylint: disable=relative-beyond-top-level


class Wan2_2TI2VModelAdapter(Wan2_2BaseModelAdapter):  # pylint: disable=too-many-ancestors
    """
    Wan2.2 统一 TI2V 模型适配器（Wan2.2-TI2V-5B）。

    与推理仓 WanTI2V 一致：单模型内按是否提供 image 分流——
    - 无图（默认）：WanTI2V.t2v()，仅需 dataset 中的 text；
    - 有图：WanTI2V.i2v()，image 来自 dataset/index.jsonl，不写入 inference_config。
    """

    scene_task = "ti2v-5B"

    class Wan2_2TI2VInferenceConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")
        size: Optional[str] = "1280*704"
        frame_num: Optional[int] = 81
        sample_shift: Optional[float] = 5.0
        sample_solver: Optional[Literal["unipc", "dpm++"]] = "unipc"
        sample_steps: Optional[int] = 50
        sample_guide_scale: Optional[float] = 5.0
        base_seed: Optional[int] = None
        offload_model: Optional[bool] = None
        task: Optional[str] = "ti2v-5B"
        convert_model_dtype: Optional[bool] = None

        @model_validator(mode='before')
        @classmethod
        def _reject_mismatched_task(cls, data: Any) -> Any:
            expected = "ti2v-5B"
            if isinstance(data, dict) and data.get("task") not in (None, expected):
                raise ValueError(
                    f"task {data['task']!r} does not match this adapter (expected {expected!r}). "
                    "Use model_type Wan2.2-TI2V-5B instead of setting task in YAML.",
                )
            return data

    def get_inference_config_class(self):
        return self.Wan2_2TI2VInferenceConfig

    def validate_calib_samples(self, samples: List[VlmCalibSample]) -> List[VlmCalibSample]:
        for idx, sample in enumerate(samples):
            if not isinstance(sample.text, str) or not sample.text.strip():
                raise SchemaValidateError(
                    f"wan2_2 ti2v sample[{idx}] requires non-empty text",
                    action="Provide text in dataset entries (index.jsonl / VlmCalibSample.text).",
                )
            # image 可选：缺省走 T2V 校准；若提供则须为非空路径字符串（走 I2V 分支）
            if sample.image is not None and (not isinstance(sample.image, str) or not sample.image.strip()):
                raise SchemaValidateError(
                    f"wan2_2 ti2v sample[{idx}] image must be a non-empty path when set",
                    action="Omit image for T2V-style calibration, or provide a valid image path.",
                )
        return samples

    def init_model(self, device: DeviceType = DeviceType.NPU) -> Dict[str, nn.Module]:
        _ = device
        self._load_pipeline()
        experts = {"": self.transformer}
        self._bind_expert_sub_adapters(experts)
        return experts

    def quantization_context(self):
        # 单 DiT：仅包装 transformer
        return self._quantization_context_with_no_sync(self.transformer)

    def _generate_video(
        self,
        prompt: str,
        image_path: Optional[str],
        inference_config: Any = None,
    ) -> None:
        from wan.configs import SIZE_CONFIGS, MAX_AREA_CONFIGS
        from PIL import Image

        # 与推理仓一致：img=None 走 t2v()，有图走 i2v()。
        img = None
        if image_path is not None and str(image_path).strip():
            img = Image.open(image_path).convert("RGB")

        self.wan_ti2v.generate(
            prompt,
            img,
            size=SIZE_CONFIGS[self._runtime_value(inference_config, "size")],
            max_area=MAX_AREA_CONFIGS[self._runtime_value(inference_config, "size")],
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

        logging.info("Creating WanTI2V pipeline.")
        self.wan_ti2v = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            quant_dit_path=args.quant_dit_path,
            device_id=device,
            rank=rank,
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        self.transformer = self.wan_ti2v.model
        self._setup_wan_dit_runtime(args, self.transformer)
