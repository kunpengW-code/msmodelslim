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
flux1 模型测试共享 fixture 与辅助类
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from msmodelslim.model.flux1.model_adapter import FLUX1ModelAdapter

TEST_MODEL_PATH = "."


@pytest.fixture(autouse=True)
def mock_diffusers_modules():
    """统一模拟所有 diffusers 相关模块"""
    mock_modules = ["diffusers", "diffusers.FluxPipeline"]
    original_modules = {mod: sys.modules.get(mod) for mod in mock_modules}
    for module_path in mock_modules:
        sys.modules[module_path] = MagicMock()
    diffusers_main = sys.modules["diffusers"]
    mock_flux_pipeline = MagicMock()
    mock_transformer = MagicMock()
    mock_transformer.config = MagicMock()
    mock_transformer.config.num_layers = 19
    mock_transformer.config.num_single_layers = 38
    mock_flux_pipeline.transformer = mock_transformer
    mock_flux_pipeline.enable_model_cpu_offload = MagicMock()
    mock_flux_pipeline.images = [MagicMock()]
    diffusers_main.FluxPipeline.from_pretrained.return_value = mock_flux_pipeline
    yield
    for mod, original in original_modules.items():
        if original is not None:
            sys.modules[mod] = original
        elif mod in sys.modules:
            del sys.modules[mod]


@pytest.fixture(name="flux1_adapter")
def _flux1_adapter():
    """创建已 mock 默认参数的 FLUX1ModelAdapter 实例"""
    with patch("msmodelslim.model.flux1.model_adapter.FLUX1ModelAdapter._get_default_model_args"):
        return FLUX1ModelAdapter("flux1", Path(TEST_MODEL_PATH))
