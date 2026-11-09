#!/usr/bin/env bash
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

# 用例名（用于打屏和指定输出路径）
CASE_NAME=analyze_qwen3

# 无论在哪个目录执行脚本，SCRIPT_DIR 始终指向当前 .sh 文件所在的绝对目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# 引入公共模块
source ${SCRIPT_DIR}/../../utils/common_utils.sh

# 安装依赖
pip install -r ${SCRIPT_DIR}/requirements.txt

msmodelslim analyze \
  --model_type Qwen3-14B \
  --model_path ${MODEL_RESOURCE_PATH}/Qwen3-14B \
  --trust_remote_code True

if [ $? -eq 0 ]; then
  echo "$CASE_NAME: Success"
else
  echo "$CASE_NAME: Failed"
  run_ok=$ret_failed
fi

exit $run_ok
