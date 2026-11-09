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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .vlm_dataset_loader import VlmCalibSample


class BaseDatasetLoader(ABC):
    """
    Base class for VLM dataset loaders in chain-of-responsibility.

    Concrete loaders must implement:
    - is_support: whether the loader can handle the resolved path.
    - load_data: parse the path and return VLM calibration samples.
    """

    DEFAULT_TEXT = "Describe this image in detail."
    def __init__(
        self,
        default_text: Optional[str] = None,
    ):
        self.supported_image_extensions = {}
        self.supported_audio_extensions = {}
        self.supported_video_extensions = {}
        self.default_text = default_text or self.DEFAULT_TEXT

    @abstractmethod
    def is_support(self, resolved_path: Path) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load_data(self, resolved_path: Path) -> List["VlmCalibSample"]:
        raise NotImplementedError
