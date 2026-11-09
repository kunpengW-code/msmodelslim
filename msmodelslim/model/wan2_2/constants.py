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

EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
        "The fluffy-furred feline gazes directly at the camera with a relaxed expression.",
        "image": "examples/i2v_input.JPG",
    },
    # 与 MindIE generate.py 一致：默认无 image，parse_args / 初始化走 T2V；有图时走 I2V。
    "ti2v-5B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
}

TASK_TYPES = {
    "t2v-A14B": "t2v",
    "i2v-A14B": "i2v",
    "ti2v-5B": "ti2v",
}

# 双专家 DiT 场景（low_noise_model + high_noise_model）；须绑定子适配器，禁止 get_expert_adapter 回退父适配器。
DUAL_EXPERT_SCENE_TASKS = frozenset({"t2v-A14B", "i2v-A14B"})

# generate.py --size 默认 1280*720；ti2v-5B 仅支持 704*1280 / 1280*704，初始化 parse_args 须显式指定。
DEFAULT_SIZE = {
    "t2v-A14B": "1280*720",
    "i2v-A14B": "1280*720",
    "ti2v-5B": "1280*704",
}
