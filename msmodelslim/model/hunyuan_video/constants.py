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

# 仅用于 hyvideo parse_args 占位；校准 prompt 来自 dataset（VlmCalibSample.text）。
PLACEHOLDER_PROMPT = "msmodelslim calibration placeholder"

TASK_TYPE = "hunyuanvideo"

# hyvideo 默认 model_resolution=540p 但 video_size 默认 (720,1280) 仅属 720p；与权重目录 t2v-720p 对齐。
DEFAULT_MODEL_RESOLUTION = "720p"
DEFAULT_VIDEO_SIZE = (720, 1280)

# HYVIDEO_CLI_LIST_FIELDS：与 hyvideo.config 中 nargs="+" 的 CLI 参数对齐。
# inference_config 中的 list/tuple 仅当字段名在此集合内才会展开为 argv（如 --video-size 720 1280）。
# 其它 list/tuple/dict 在 _namespace_to_argv 中跳过，与推理侧 CLI 能力一致。
HYVIDEO_CLI_LIST_FIELDS = frozenset({"video_size"})

DIT_WEIGHT_REL = (
    "hunyuan-video-t2v-720p",
    "transformers",
    "mp_rank_00_model_states.pt",
)

VAE_PATH_REL = ("hunyuan-video-t2v-720p", "vae")
TEXT_ENCODER_PATH_REL = ("text_encoder",)
TEXT_ENCODER_2_PATH_REL = ("clip-vit-large-patch14",)
