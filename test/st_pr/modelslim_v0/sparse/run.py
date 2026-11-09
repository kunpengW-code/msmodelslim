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

import os
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from transformers import AutoTokenizer, AutoModel

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator as SparseQuantCalibrator
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig as SparseQuantConfig

# -------------------------- 获取脚本自身所在目录（不受执行目录影响） --------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

model_resource_path = os.environ.get("MODEL_RESOURCE_PATH")
if not model_resource_path:
    raise Exception("获取不到模型路径，请先检查环境变量 MODEL_RESOURCE_PATH")

fp16_path = os.path.join(model_resource_path, "chatglm2-6b")
save_path = os.path.join(script_dir, "output_sparse")
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=fp16_path,
    trust_remote_code=True,
    local_files_only=True)

model = AutoModel.from_pretrained(
    pretrained_model_name_or_path=fp16_path,
    torch_dtype='auto',
    device_map='auto',
    trust_remote_code=True,
    local_files_only=True).eval()

calib_list = ["Where is the capital of China?",
              "Please make a poem:"]


def get_calib_dataset(tokenizers, calib_lists, device='cpu'):
    dataset_all = []
    for calib_data in calib_lists:
        inputs = tokenizers(
            calib_data,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=512
        ).to(device)
        inputs.data['position_ids'] = torch.arange(inputs["input_ids"].shape[1], device=device).unsqueeze(0)
        dataset_tmp = [inputs.data['input_ids'], inputs.data['position_ids'], inputs.data['attention_mask'], None, None,
                       None, None, None,
                       None, None, True]
        dataset_all.append(dataset_tmp)
    return dataset_all


dataset_calib = get_calib_dataset(tokenizer, calib_list, device=model.device)

quant_config = SparseQuantConfig(w_bit=4,
                                 disable_names=['transformer.encoder.layers.0.self_attention.query_key_value',
                                                'transformer.encoder.layers.0.self_attention.dense',
                                                'transformer.encoder.layers.0.mlp.dense_h_to_4h',
                                                'transformer.encoder.layers.0.mlp.dense_4h_to_h',
                                                'transformer.output_layer'],
                                 dev_type='npu',
                                 dev_id=model.device.index,
                                 act_method=3,
                                 pr=2.0,
                                 fraction=0.011,
                                 nonuniform=False,
                                 mm_tensor=False,
                                 co_sparse=True)

calibrator = SparseQuantCalibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run(int_infer=False)
calibrator.save(save_path, save_type=["safe_tensor"])
print('Save quant weight success!')
