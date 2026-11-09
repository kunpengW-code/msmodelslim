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

from ascend_utils.common.security.type import (
    check_type,
    check_number,
    check_int,
    check_element_type,
    check_character,
    check_dict_character,
    check_dict_element,
)
from ascend_utils.common.security.path import (
    MAX_READ_FILE_SIZE_4G,
    MAX_READ_FILE_SIZE_32G,
    MAX_READ_FILE_SIZE_512G,
    get_valid_path,
    get_valid_read_path,
    get_valid_write_path,
    check_write_directory,
    get_write_directory,
    json_safe_load,
    json_safe_dump,
    yaml_safe_load,
    yaml_safe_dump,
    file_safe_write,
    safe_delete_path_if_exists,
    safe_copy_file,
    SafeWriteUmask,
    set_file_stat
)
