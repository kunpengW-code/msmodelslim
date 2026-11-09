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

msmodelslim.utils.security.shell 模块的单元测试
"""

import signal
import subprocess
from unittest.mock import MagicMock, Mock, patch

import pytest

from msmodelslim.utils.exception import SecurityError
from msmodelslim.utils.security import shell as shell_module


class TestValidateSafeIdentifier:
    """测试 validate_safe_identifier 命令标识符校验"""

    def test_validate_safe_identifier_raises_when_empty(self):
        """空标识符时应抛出 SecurityError"""
        with pytest.raises(SecurityError, match="cannot be empty"):
            shell_module.validate_safe_identifier("", "binary")

    def test_validate_safe_identifier_raises_when_invalid_chars(self):
        """含命令注入风险字符时应抛出 SecurityError"""
        with pytest.raises(SecurityError, match="invalid characters"):
            shell_module.validate_safe_identifier("foo;rm", "binary")

    def test_validate_safe_identifier_return_value_when_valid(self):
        """合法标识符应原样返回"""
        assert shell_module.validate_safe_identifier("my-binary_v1", "binary") == "my-binary_v1"


class TestSanitizeExtraArgs:
    """测试 sanitize_extra_args 额外参数清洗"""

    def test_sanitize_extra_args_return_empty_when_none(self):
        """无额外参数时应返回空列表"""
        assert not shell_module.sanitize_extra_args(None)

    def test_sanitize_extra_args_return_list_when_string(self):
        """字符串输入应经 shlex 分割并校验后返回列表"""
        result = shell_module.sanitize_extra_args('--foo bar')
        assert result == ["--foo", "bar"]

    def test_sanitize_extra_args_return_list_when_list_input(self):
        """列表输入应逐项校验后返回"""
        result = shell_module.sanitize_extra_args(["--opt", "value"])
        assert result == ["--opt", "value"]

    def test_sanitize_extra_args_skip_empty_parts_when_string_has_spaces(self):
        """仅空白字符串时应忽略并返回空列表"""
        result = shell_module.sanitize_extra_args('  ')
        assert not result


class TestBuildSafeCommand:
    """测试 build_safe_command 安全命令列表构建"""

    def test_build_safe_command_return_parts_when_valid(self):
        """合法 binary 与参数应组装为命令列表"""
        cmd = shell_module.build_safe_command("python", "script.py", extra_args=["--flag"])
        assert cmd == ["python", "script.py", "--flag"]

    def test_build_safe_command_raises_when_binary_invalid(self):
        """binary 含非法字符时应抛出 SecurityError"""
        with pytest.raises(SecurityError, match="invalid characters"):
            shell_module.build_safe_command("bad|cmd")


class TestBuildSafeCommandWithOptions:
    """测试 build_safe_command_with_options 带选项的命令构建"""

    def test_build_safe_cmd_with_options_return_parts_when_valid(self):
        """合法 options 与 extra_args 应正确展开到命令列表"""
        cmd = shell_module.build_safe_command_with_options("run.sh", {"--models": "model_a"}, extra_args="--verbose")
        assert cmd == ["run.sh", "--models", "model_a", "--verbose"]

    def test_build_safe_cmd_with_options_raises_when_option_name_empty(self):
        """选项名为空时应抛出 SecurityError"""
        with pytest.raises(SecurityError, match="option_name cannot be empty"):
            shell_module.build_safe_command_with_options("bin", {"": "v"})

    def test_build_safe_cmd_with_options_raises_when_option_name_invalid(self):
        """选项名含非法字符时应抛出 SecurityError"""
        with pytest.raises(SecurityError, match="option_name contains invalid"):
            shell_module.build_safe_command_with_options("bin", {"--bad name": "v"})

    def test_build_safe_cmd_with_options_skip_value_when_none(self):
        """选项值为 None 时应只追加选项名、不追加值"""
        cmd = shell_module.build_safe_command_with_options("bin", {"-v": None})
        assert cmd == ["bin", "-v"]


class TestShellRunner:
    """测试 ShellRunner.run_safe_cmd 命令执行"""

    def test_shell_runner_return_success_when_command_ok(self):
        """命令成功时应返回 success=True 及 stdout"""
        mock_result = MagicMock(returncode=0, stdout="ok", stderr="")
        with patch("subprocess.run", return_value=mock_result):
            ok, out, err = shell_module.ShellRunner.run_safe_cmd("echo", options={"-n": "1"})
        assert ok is True
        assert out == "ok"
        assert err == ""

    def test_shell_runner_return_failure_when_nonzero_exit(self):
        """非零退出码时应返回 success=False"""
        mock_result = MagicMock(returncode=1, stdout="", stderr="fail")
        with patch("subprocess.run", return_value=mock_result):
            ok, out, err = shell_module.ShellRunner.run_safe_cmd("false")
        assert ok is False
        assert err == "fail"

    def test_shell_runner_return_timeout_when_expired(self):
        """超时时应返回 success=False 且 stderr 为 Timeout"""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 1)):
            ok, out, err = shell_module.ShellRunner.run_safe_cmd("sleep", timeout=1)
        assert ok is False
        assert err == "Timeout"

    def test_shell_runner_use_plain_command_when_no_options(self):
        """未传 options 时应走 build_safe_command 分支"""
        with patch.object(shell_module, "build_safe_command", return_value=["bin"]) as mock_build:
            with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="", stderr="")):
                shell_module.ShellRunner.run_safe_cmd("bin")
        mock_build.assert_called_once()


class TestAsyncProcess:
    """测试 AsyncProcess 异步子进程管理"""

    def test_async_process_start_popen_when_called(self, tmp_path):
        """start 应通过 Popen 拉起进程并记录 process 句柄"""
        log_file = str(tmp_path / "svc.log")
        proc = shell_module.AsyncProcess("server", log_file, options={"--port": "8000"})
        mock_popen = MagicMock()
        mock_popen.pid = 12345
        with patch("subprocess.Popen", return_value=mock_popen) as mock_pop:
            proc.start()
        mock_pop.assert_called_once()
        assert proc.process is mock_popen

    def test_async_process_merge_env_when_env_provided(self, tmp_path):
        """传入 env 时应合并进子进程环境变量且值转为字符串"""
        log_file = str(tmp_path / "svc.log")
        proc = shell_module.AsyncProcess("server", log_file, env={"EXTRA": 42})
        with patch("subprocess.Popen") as mock_pop:
            proc.start()
        env_passed = mock_pop.call_args[1]["env"]
        assert env_passed["EXTRA"] == "42"

    def test_async_process_stop_killpg_when_running(self, tmp_path):
        """stop 应向进程组发送 SIGTERM"""
        log_file = str(tmp_path / "svc.log")
        proc = shell_module.AsyncProcess("server", log_file)
        proc.process = MagicMock()
        proc.process.pid = 999
        proc.process.wait = Mock()
        with patch("os.getpgid", return_value=999):
            with patch("os.killpg") as mock_kill:
                proc.stop()
        mock_kill.assert_called_with(999, signal.SIGTERM)

    def test_async_process_stop_force_kill_when_term_fails(self, tmp_path):
        """SIGTERM 等待失败时应降级为 SIGKILL"""
        log_file = str(tmp_path / "svc.log")
        proc = shell_module.AsyncProcess("server", log_file)
        proc.process = MagicMock()
        proc.process.pid = 1000
        proc.process.wait = Mock(side_effect=Exception("wait failed"))
        with patch("os.getpgid", return_value=1000):
            with patch("os.killpg") as mock_kill:
                proc.stop()
        assert mock_kill.call_count >= 2
        assert mock_kill.call_args_list[-1][0][1] == signal.SIGKILL
