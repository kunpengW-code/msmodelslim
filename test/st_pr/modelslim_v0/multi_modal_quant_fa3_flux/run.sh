#!/usr/bin/env bash
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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
CASE_NAME=multi_modal_quant_fa3_flux

# 无论在哪个目录执行脚本，SCRIPT_DIR 始终指向当前 .sh 文件所在的绝对目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd) # "
OUTPUT_DIR="${SCRIPT_DIR}/output_${CASE_NAME}"
# 直接拼接「脚本目录 + 配置文件名」，确保无论在哪执行都能找到配置文件
CONFIG_FILE="${SCRIPT_DIR}/dense-w8a8-ssz-v1.yaml"

# 引入公共模块
source ${SCRIPT_DIR}/../../utils/common_utils.sh

MSMODELSLIM_SOURCE_DIR=${SCRIPT_DIR}/../../../..
echo "MSMODELSLIM_SOURCE_DIR: ${MSMODELSLIM_SOURCE_DIR}"
chmod 640 ${MSMODELSLIM_SOURCE_DIR}/example/multimodal_sd/Flux/calib_prompts.txt

# 获取依赖工程仓
cd ${SCRIPT_DIR}
git clone https://modelers.cn/MindIE/FLUX.1-dev.git
FLUX_REPO_DIR=${SCRIPT_DIR}/FLUX.1-dev

# 切换指定版本与依赖
cd ${FLUX_REPO_DIR}
git checkout 12e09174353b1bd57bf7fcb80386f59b09fbbefe
pip install -r requirements.txt

export CPU_AFFINITY_CONF=2
export TASK_QUEUE_ENABLE=2

# 单卡32G Flux 等价优化推理
python ${MSMODELSLIM_SOURCE_DIR}/example/multimodal_sd/Flux/inference_flux.py \
  --path ${MODEL_RESOURCE_PATH}/Flux.1-dev \
  --save_path ${OUTPUT_DIR} \
  --device_id 0 \
  --device "npu" \
  --prompt_path "${MSMODELSLIM_SOURCE_DIR}/example/multimodal_sd/Flux/calib_prompts.txt" \
  --width 1024 \
  --height 1024 \
  --infer_steps 2 \
  --seed 42 \
  --use_cache \
  --device_type "A2-32g-single" \
  --batch_size 1 \
  --max_num_prompt 2 \
  --do_quant \
  --quant_weight_save_folder "${OUTPUT_DIR}/safetensors" \
  --quant_dump_calib_folder "${OUTPUT_DIR}/cache" \
  --quant_type "w8a8_dynamic_fa3"

if [ $? -eq 0 ]; then
  echo ${CASE_NAME}: Success
else
  echo ${CASE_NAME}: Failed
  run_ok=$ret_failed
fi

exit $run_ok
