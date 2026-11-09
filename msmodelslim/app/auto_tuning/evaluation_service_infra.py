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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from msmodelslim.core.tune_strategy import EvaluateResult
from msmodelslim.core.const import DeviceType
from msmodelslim.utils.plugin import TypedConfig

EVALUATE_PLUGIN_PATH = "msmodelslim.evaluation.plugins"


class EvaluateContext(BaseModel):
    evaluate_id: str
    device: DeviceType = DeviceType.NPU
    device_indices: Optional[List[int]] = None
    working_dir: Path


@TypedConfig.plugin_entry(entry_point_group=EVALUATE_PLUGIN_PATH)
class EvaluateServiceConfig(TypedConfig):
    type: TypedConfig.TypeField


class EvaluateServiceInfra(ABC):
    @abstractmethod
    def evaluate(self,
                 context: EvaluateContext,
                 evaluate_config: EvaluateServiceConfig,
                 model_path: Path,
                 ) -> EvaluateResult:
        ...
