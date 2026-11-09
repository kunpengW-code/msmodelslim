#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
msmodelslim.core.quant_service.modelslim_convert.impl.save_adapter 模块的单元测试
"""

from unittest.mock import MagicMock, patch

import pytest
from torch import nn

from msmodelslim.core.convert.config import ConvertConfig
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.quant_service.modelslim_convert.impl.save_adapter import SaveProcessorAdapter
from msmodelslim.processor.save.processor import QuantSaveProcessor


class TestSaveProcessorAdapter:
    """测试 SaveProcessorAdapter 类"""

    def test_save_call_ascendv1_when_dst_format_ascendv1(self):
        tree = nn.Module()
        config = ConvertConfig(model_path="/m", save_path="/out", dst_format="ascendv1")
        context = ConvertContext(config=config)
        context.reader = MagicMock()

        with patch(
            "msmodelslim.core.quant_service.modelslim_convert.impl.save_adapter.SaveProcessorAdapter._save_ascendv1",
        ) as mock_save:
            SaveProcessorAdapter().save(context, tree)
            mock_save.assert_called_once()

    def test_save_call_compressed_tensors_when_dst_format_hf(self):
        tree = nn.Module()
        config = ConvertConfig(model_path="/m", save_path="/out", dst_format="huggingface")
        context = ConvertContext(config=config)
        context.reader = MagicMock()

        with patch(
            "msmodelslim.core.quant_service.modelslim_convert.impl.save_adapter.SaveProcessorAdapter._save_compressed_tensors",
        ) as mock_save:
            SaveProcessorAdapter().save(context, tree)
            mock_save.assert_called_once()

    def test_save_raise_error_when_dst_format_unsupported(self):
        tree = nn.Module()
        config = ConvertConfig(model_path="/m", save_path="/out", dst_format="unknown_fmt")
        context = ConvertContext(config=config)
        with pytest.raises(ValueError, match="Unsupported dst_format"):
            SaveProcessorAdapter().save(context, tree)

    def test_save_compressed_tensors_builds_valid_quant_save_config(self):
        from msmodelslim.format.compressed_tensors_format.compressed_tensors import (
            CompressedTensorsQuantFormatConfig,
        )

        tree = nn.Module()
        config = ConvertConfig(model_path="/m", save_path="/out", dst_format="huggingface")
        context = ConvertContext(config=config)
        context.reader = MagicMock()

        with (
            patch.object(QuantSaveProcessor, "pre_run"),
            patch.object(QuantSaveProcessor, "postprocess"),
            patch.object(QuantSaveProcessor, "post_run"),
            patch(
                "msmodelslim.core.quant_service.modelslim_convert.impl.save_adapter.QuantSaveProcessor",
            ) as mock_cls,
        ):
            SaveProcessorAdapter().save(context, tree)
            cfg = mock_cls.call_args[0][1]
            assert isinstance(cfg.format, CompressedTensorsQuantFormatConfig)
            assert cfg.format.part_file_size == 4
