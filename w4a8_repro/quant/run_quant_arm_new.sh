#!/bin/bash
# W4A8 arm quantization runner (inside wkp-vllm container).
# usage: run_quant_arm_new.sh <config_yaml_name> <out_name> <device_id>
set -u
CFG=$1; OUT=$2; DEV=$3
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=$DEV
mkdir -p /home/w30022218/w4a8_tuning/logs /home/w30022218/w4a8_tuning/quant_out
echo "[$(date '+%F %T')] start arm=$OUT config=$CFG device=$DEV"
exec msmodelslim quant \
  --model_path /home/models/Qwen3.6-27B \
  --save_path /home/w30022218/w4a8_tuning/quant_out/$OUT \
  --device npu \
  --model_type Qwen3.6-27B \
  --config_path /home/w30022218/w4a8_tuning/configs/$CFG \
  --trust_remote_code True
