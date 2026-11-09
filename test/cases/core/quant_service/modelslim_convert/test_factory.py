#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
msmodelslim.core.quant_service.modelslim_convert.factory 模块的单元测试
"""

from msmodelslim.core.quant_service.modelslim_convert.application import ConvertApplication
from msmodelslim.core.quant_service.modelslim_convert.factory import create_convert_application


class TestCreateConvertApplication:
    """测试 create_convert_application 工厂函数"""

    def test_create_convert_application_return_application_when_called(self):
        app = create_convert_application()
        assert isinstance(app, ConvertApplication)
        assert app._router is not None
        assert app._executor._router is app._router
