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

from typing import Any, List, Type, ClassVar, Union, Set, Literal, get_origin, get_args

from pydantic import BaseModel, BeforeValidator, TypeAdapter, Field, model_validator, SerializeAsAny
from torch import nn
from typing_extensions import Annotated
from typing_extensions import Self

from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.core.base.processor import BaseProcessor
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.validation.validation_state import add_validation_error, set_validation_context


class AutoProcessorConfig(BaseModel):
    type: str

    _registry: ClassVar[Set[Type[Self]]] = set()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        if 'type' not in cls.model_fields:
            raise TypeError(f"Must provide a type field for {cls.__bases__}'s subclass")

        cls._registry.add(cls)
        get_logger().debug("Add subclass %s to registry", cls.__name__)

        return super().__pydantic_init_subclass__(**kwargs)

    @model_validator(mode='wrap')
    @classmethod
    def _validate_subclass(cls: Type['AutoProcessorConfig'], value: Any, handler: Any) -> 'AutoProcessorConfig':
        union_type = TypeAdapter(Annotated[Union[tuple(cls._registry)], Field(discriminator='type')])
        # 检查 cls 的 type 字段是否是 Literal 且值以 _ 开头
        type_field = cls.model_fields.get('type')
        is_literal_with_underscore = False
        if type_field is not None:
            type_annotation = type_field.annotation
            if get_origin(type_annotation) is Literal:
                literal_args = get_args(type_annotation)
                is_literal_with_underscore = any(isinstance(arg, str) and arg.startswith('_') for arg in literal_args)
        if is_literal_with_underscore or cls not in cls._registry:
            # 排除 type 字段以 _ 开头的配置
            return union_type.validate_python(value)
        return handler(value)


def validate_auto_processor_config_list(v: Any) -> List['AutoProcessorConfig']:
    if isinstance(v, list):
        # spec 是所有架构的统一前缀，在 BeforeValidator 中设置
        set_validation_context(path_prefix="spec.")
        validated_configs = []
        for i, item in enumerate(v):
            if isinstance(item, AutoProcessorConfig):
                validated_configs.append(item)
            elif isinstance(item, dict):
                validated_configs.append(item)
            else:
                add_validation_error(
                    f"process.{i}: Invalid config item: expected dict with 'type' field, "
                    f"got {type(item).__name__}={item!r}. "
                    f"Check YAML indentation - item may be incorrectly nested under another field."
                )
        return validated_configs
    raise ValueError("Expected a list of AutoProcessorConfig or dict")


AutoProcessorConfigList = Annotated[
    List[SerializeAsAny[AutoProcessorConfig]], BeforeValidator(validate_auto_processor_config_list)
]


@QABCRegistry.register_abc(dispatch_key=Type[AutoProcessorConfig])
class AutoSessionProcessor(BaseProcessor):
    """
    会话基础处理器。
    """

    def __repr__(self):
        return self.__class__.__name__

    @classmethod
    def from_config(cls, model: nn.Module, config: AutoProcessorConfig, adapter: object, *args, **kwargs) -> Self:
        return QABCRegistry.create(
            AutoSessionProcessor,
            type(config),
            *(model, config, adapter, *args),
            **kwargs,
        )

    def support_distributed(self) -> bool:
        return False

    def is_data_free(self) -> bool:
        return False

    def need_kv_cache(self):
        return False

    def process(self, request: BatchProcessRequest) -> None:
        if self.is_data_free():
            return

        super().process(request)
