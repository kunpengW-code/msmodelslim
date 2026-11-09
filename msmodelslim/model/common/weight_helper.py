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
"""

import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Union, List

from torch import nn
from safetensors import safe_open
from tqdm import tqdm

from msmodelslim.utils.security import get_valid_read_path, json_safe_load, MAX_READ_FILE_SIZE_512G


@lru_cache(maxsize=1)
def get_weight_map(model_path: Union[str, Path]):
    model_index_path = os.path.join(model_path, "model.safetensors.index.json")
    model_index = json_safe_load(model_index_path)
    weight_map = model_index['weight_map']
    return weight_map


def get_state_dict(model_path: Union[str, Path], module: nn.Module, prefix: str = "", exclude: List[str] = None):
    if exclude is None:
        exclude = []
    weight_map = get_weight_map(model_path)
    names = map(lambda x: x[0], module.named_parameters())

    groups = defaultdict(list)
    for name in names:
        if name in exclude:
            continue
        weight_key = f'{prefix}.{name}' if prefix else name
        file_name = weight_map[weight_key]
        groups[file_name].append(name)

    state_dict = {}
    for file_name in tqdm(groups, desc=f'Loading {prefix}'):
        file_path = os.path.join(model_path, file_name)
        file_path = get_valid_read_path(file_path, extensions='safetensors', size_max=MAX_READ_FILE_SIZE_512G)
        with safe_open(file_path, framework='pt', device='cpu') as f:
            for name in tqdm(groups[file_name], desc=f'Loading {file_path}'):
                state_dict[name] = f.get_tensor(f'{prefix}.{name}' if prefix else name)
    return state_dict
