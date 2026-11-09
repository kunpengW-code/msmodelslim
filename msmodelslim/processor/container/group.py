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

from typing import Optional, Literal

from torch import nn

from msmodelslim.ir.qal import QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.base import AutoSessionProcessor, AutoProcessorConfig, AutoProcessorConfigList


class GroupProcessorConfig(AutoProcessorConfig):
    type: Literal['group']
    configs: AutoProcessorConfigList


@QABCRegistry.register(dispatch_key=GroupProcessorConfig, abc_class=AutoSessionProcessor)
class GroupProcessor(AutoSessionProcessor):
    """
    前向量化处理器合并器，用于将多个前向量化处理器合并为一个处理器，用于减少模型推理的次数。
    """

    def __init__(
            self,
            model: nn.Module,
            config: GroupProcessorConfig,
            adapter: Optional[object] = None,
    ):
        super().__init__(model)
        self.processors = [AutoSessionProcessor.from_config(model, cfg, adapter) for cfg in config.configs]
        self.processor_names = [processor.__class__.__name__ for processor in self.processors]

    def __repr__(self) -> str:
        return f"GroupProcessor(processors={self.processor_names})"

    def preprocess(self, request: BatchProcessRequest) -> None:
        for processor in self.processors:
            processor.preprocess(request)

    def postprocess(self, request: BatchProcessRequest) -> None:
        for processor in self.processors:
            processor.postprocess(request)

    def pre_run(self) -> None:
        for processor in self.processors:
            processor.pre_run()

    def post_run(self) -> None:
        for processor in self.processors:
            processor.post_run()

    def is_data_free(self) -> bool:
        """
        判断处理器是否需要数据。
        """
        return all(processor.is_data_free() for processor in self.processors)

    def need_kv_cache(self):
        return any(processor.need_kv_cache() for processor in self.processors)
    
    def support_distributed(self) -> bool:
        return any(processor.support_distributed() for processor in self.processors)
