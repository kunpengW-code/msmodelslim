#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
msmodelslim.core.quant_service.modelslim_convert.config_detect 模块的单元测试
"""

from pathlib import Path

from msmodelslim.core.quant_service.modelslim_convert.config_detect import is_modelslim_convert_config


class TestIsModelslimConvertConfig:
    """测试 is_modelslim_convert_config 函数"""

    def test_is_modelslim_convert_config_return_true_when_apiversion_matches(self, tmp_path: Path):
        cfg = tmp_path / "convert.yaml"
        cfg.write_text("apiversion: modelslim_convert\nspec: {}\n", encoding="utf-8")
        assert is_modelslim_convert_config(cfg) is True

    def test_is_modelslim_convert_config_return_false_when_apiversion_other(self, tmp_path: Path):
        cfg = tmp_path / "quant.yaml"
        cfg.write_text("apiversion: modelslim_v1\nspec: {}\n", encoding="utf-8")
        assert is_modelslim_convert_config(cfg) is False
