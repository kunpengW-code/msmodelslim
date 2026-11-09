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

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set

from msmodelslim.utils.exception import InvalidDatasetError
from msmodelslim.utils.security import get_valid_read_path

from .jsonl_dataset_loader import JsonlDatasetLoader
from .vlm_dataset_loader import VlmCalibSample


class IndexedDirectoryDatasetLoader(JsonlDatasetLoader):
    """
    IndexedDirectoryDatasetLoader

    Supports:
    - dataset_name is a directory that contains exactly one of:
    - index.json
    - index.jsonl

    If both exist, raise error (strict).

    Loading reuses JsonlDatasetLoader parsing through inheritance.
    """
    def __init__(
        self,
        default_text: Optional[str] = None,
    ):
        super().__init__(
            default_text=default_text,
        )

    def is_support(self, resolved_path: Path) -> bool:
        if not resolved_path.is_dir():
            return False
        index_json = resolved_path / "index.json"
        index_jsonl = resolved_path / "index.jsonl"
        return index_json.exists() or index_jsonl.exists()

    def load_data(self, resolved_path: Path) -> List[VlmCalibSample]:
        resolved_path = Path(get_valid_read_path(str(resolved_path), is_dir=True, check_user_stat=True))

        index_json = resolved_path / "index.json"
        index_jsonl = resolved_path / "index.jsonl"

        if index_json.exists() and index_jsonl.exists():
            raise InvalidDatasetError(
                f"Both index.json and index.jsonl exist in directory: {resolved_path}",
                action="Please keep only one of index.json or index.jsonl.",
            )

        if index_json.exists():
            return super().load_data(index_json)
        if index_jsonl.exists():
            return super().load_data(index_jsonl)

        raise InvalidDatasetError(
            f"Directory does not contain index.json or index.jsonl: {resolved_path}",
            action="Please add one of index.json/index.jsonl, or use another dataset format.",
        )
