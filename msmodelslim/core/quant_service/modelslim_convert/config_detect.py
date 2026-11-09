#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""检测 YAML 是否为 modelslim_convert 任务配置。"""

from __future__ import annotations

from pathlib import Path

from msmodelslim.utils.security import yaml_safe_load

MODELSLIM_CONVERT_APIVERSION = "modelslim_convert"


def is_modelslim_convert_config(config_path: str | Path) -> bool:
    """``config_path`` 指向的 YAML 是否 ``apiversion: modelslim_convert``。"""
    raw = yaml_safe_load(str(config_path))
    if not isinstance(raw, dict):
        return False
    return raw.get("apiversion") == MODELSLIM_CONVERT_APIVERSION
