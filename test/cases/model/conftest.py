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

"""
Pytest config for model tests.

复用全局 mock 工具，避免在导入 msmodelslim 期间真正初始化配置文件和安全路径检查。
同时通过 fixture 方式按会话级别 mock 缺失的第三方依赖（如 wcmatch），
避免对其他测试任务造成长期污染。
"""

import sys
from unittest.mock import MagicMock

import pytest
import torch

from testing_utils.mock import mock_security_library, mock_kia_library, mock_init_config


mock_init_config()
mock_kia_library()
mock_security_library()


# fixup: replace torch.npu with a clean stub. Some tests (e.g. flux1)
# overwrite torch.npu with a bare Mock and don't clean up, causing
# subsequent tests to see Mock-tainted current_device(). An autouse
# fixture ensures the stub is restored before every test.
class _NpuStub:
    """Safe NPU stub — not a Mock, so no auto-created child mocks."""
    def is_available(self):
        return False

    def current_device(self):
        return 0


@pytest.fixture(autouse=True)
def _reset_npu_stub():
    """Restore torch.npu stub before each test, preventing cross-test Mock leakage."""
    torch.npu = _NpuStub()  # type: ignore[attr-defined]
    yield

# 记录 wcmatch 的原始状态，用于 pytest_unconfigure 中清理
_wcmatch_original = None
_wcmatch_mock_used = False

try:
    import wcmatch
    _wcmatch_original = sys.modules.get("wcmatch")
except ImportError:
    # 如果导入失败，创建 mock
    _wcmatch_mock_used = True
    _wcmatch_original = None
    sys.modules["wcmatch"] = MagicMock()
    sys.modules["wcmatch"].fnmatch = MagicMock()


def pytest_unconfigure(config):
    """
    在测试结束后清理 wcmatch mock，恢复原始模块（如果存在）。
    仅清理在本 conftest 中新创建的 mock，避免影响外部真实依赖。
    """
    global _wcmatch_original, _wcmatch_mock_used
    
    if _wcmatch_mock_used:
        if "wcmatch" in sys.modules:
            del sys.modules["wcmatch"]
        if _wcmatch_original is not None:
            sys.modules["wcmatch"] = _wcmatch_original
