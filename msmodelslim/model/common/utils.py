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

import torch.distributed as dist

from msmodelslim.utils.exception import SchemaValidateError


def _get_expert_range(config):
    if hasattr(config, 'num_experts') and isinstance(
        config.num_experts, int
    ):  # Mock时hasattr一直为True，返回Mock类型无法遍历
        expert_num = config.num_experts
    elif (
        hasattr(config, 'n_routed_experts')
        and hasattr(config, 'n_shared_experts')
        and isinstance(config.n_routed_experts, int)
    ):
        expert_num = config.n_routed_experts
    else:
        expert_num = 0
    if expert_num == 0 or not dist.is_initialized() or dist.get_world_size() <= 1:
        return 0, expert_num  # 单卡：返回全部 expert
    world_size = dist.get_world_size()
    if expert_num % world_size != 0:
        raise SchemaValidateError(
            f"The total number of experts ({expert_num}) must be divisible by the world size ({world_size})."
        )
    n_local_experts = expert_num // world_size
    rank = dist.get_rank()
    start = rank * n_local_experts
    end = start + n_local_experts
    return start, end  # 多卡：只返回本 rank 的 expert 范围
