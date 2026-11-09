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
__all__ = [
    'FakeQuantizedLinear',
    'FlatQuantizedLinear',
    'FlatNormWrapper',
    'ForwardMode',
    'FakeQuantizedLinearConfig',
    'WeightQuantizer',
    'ActivationQuantizer',
    'asym_quant',
    'asym_dequant',
    'GeneralMatrixTrans',
    'SVDSingleTransMatrix',
    'InvSingleTransMatrix',
    'DiagonalTransMatrix'
]


from .fake_clip_quantizer import (
    WeightQuantizer,
    ActivationQuantizer,
    asym_quant,
    asym_dequant,
)
from .flat_fake_quant_linear import (
    FlatFakeQuantLinear,
    FlatNormWrapper,
    FlatFakeQuantLinearConfig,
    ForwardMode,
)
from .flat_quant_manager import FlatQuantLayerManager

from .structure_pair import (
    StructurePair,
    NormLinearPair,
    MLPLinearLinearPair,
    MLPNormLinearPair,
    AttnLinearLinearPair,
    AttnNormLinearPair,
)
from .trans_matrix import (
    GeneralMatrixTrans,
    SVDSingleTransMatrix,
    InvSingleTransMatrix,
    DiagonalTransMatrix
)

from .utils import (
    get_decompose_dim,
    get_init_scale,
    get_init_weight,
    get_n_set_parameters_byname,
    set_require_grad_all,
    stat_input_hook,
    stat_tensor,
    get_trainable_parameters,
    get_para_names,
    match_pattern,
    move_tensors_to_device,
    empty_cache,
    get_module_by_name,
    set_module_by_name,
    clone_module_hooks,
    remove_after_substring,
)