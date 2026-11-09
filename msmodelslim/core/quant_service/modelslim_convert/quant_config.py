#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
modelslim_convert 量化任务配置（apiversion + spec）。
"""

from __future__ import annotations

from typing_extensions import Self

from msmodelslim.core.quant_service.interface import BaseQuantConfig
from .config_mapper import ModelslimConvertServiceConfig, load_specific_config


class ModelslimConvertQuantConfig(BaseQuantConfig):
    spec: ModelslimConvertServiceConfig

    @classmethod
    def from_base(cls, quant_config: BaseQuantConfig) -> Self:
        return cls(
            apiversion=quant_config.apiversion,
            spec=load_specific_config(quant_config.spec),
        )
