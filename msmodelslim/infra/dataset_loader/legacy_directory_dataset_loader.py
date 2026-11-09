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

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from msmodelslim.utils.exception import InvalidDatasetError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.security import get_valid_read_path

from .base_dataset_loader import BaseDatasetLoader
from .vlm_dataset_loader import VlmCalibSample


class LegacyDirectoryDatasetLoader(BaseDatasetLoader):
    """
    LegacyDirectoryDatasetLoader

    Supported inputs (same as legacy):
    1) Pure text: .json/.jsonl file
    - jsonl line: "text" or {"text": "..."}
    - json root: string/dict/list (same rules)
    2) Pure images (default_text): directory with images, no json/jsonl
    3) Images + custom text: directory with images + single json/jsonl
    4) Mixed: directory with images + json/jsonl containing mixed text-only and image+text entries

    Notes:
    - This loader intentionally keeps the legacy "best effort" behavior:
    - invalid lines/items are skipped with warnings (instead of failing fast)
    - images not referenced in json/jsonl are appended with default_text
    """
    SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        default_text: Optional[str] = None,
    ):
        super().__init__(default_text=default_text)
        self.supported_image_extensions = self.SUPPORTED_IMAGE_EXTENSIONS
        self.supported_description_file_extensions = {".json", ".jsonl"}

    def is_support(self, resolved_path: Path) -> bool:
        if resolved_path.is_dir():
            return True
        return resolved_path.is_file() and resolved_path.suffix.lower() in self.supported_description_file_extensions

    def load_data(self, resolved_path: Path) -> List[VlmCalibSample]:

        # Dispatch based on type
        if resolved_path.is_dir():
            # Type 2, 3, or 4: Directory (images only, images+json/jsonl, or mixed)
            directory = Path(get_valid_read_path(str(resolved_path), is_dir=True, check_user_stat=True))
            get_logger().info("Loading from directory: %s", directory)
            return self._load_from_directory(directory)

        if resolved_path.is_file() and resolved_path.suffix.lower() in self.supported_description_file_extensions:
            # Type 1: Pure text json/jsonl file
            file_suffix = resolved_path.suffix.lower()
            file_path = Path(get_valid_read_path(str(resolved_path), is_dir=False, check_user_stat=True))
            get_logger().info("Loading pure text dataset from %s file: %s", file_suffix, file_path)
            return self._load_text_from_file(file_path, file_suffix=file_suffix)

        # Unknown type or unsupported file format
        get_logger().error("Unsupported dataset type: %s", resolved_path)
        raise InvalidDatasetError(
            f"Dataset path exists but is not a valid type: {resolved_path}",
            action="Please provide either a directory (images/mixed) or a json/jsonl file (text)",
        )

    def _load_text_from_file(self, file_path: Union[Path, str], file_suffix: str = ".jsonl") -> List[VlmCalibSample]:
        """
        Load pure text dataset from json/jsonl file (Type 1).
        
        For .jsonl:
            Each line must be an independent JSON value:
            - Plain string: "Who are you?"
            - Dict with text: {"text": "Who are you?"}

        For .json:
            The entire file must be a single valid JSON value:
            - A list of strings or dicts (same formats as above)
            - Or a single string/dict
        
        Args:
            file_path: Path to json/jsonl file
        
        Returns:
            List of VlmCalibSample: [VlmCalibSample(text="...", image=None), ...]
        """
        file_path = get_valid_read_path(str(file_path))
        dataset: List[VlmCalibSample] = []

        def _push_item(item: object, line_num_hint: str) -> bool:
            if isinstance(item, str):
                # plain string is stripped, empty/whitespace-only is skipped.
                entry, _, _ = self._handle_plain_string(item, line_num_hint)
                if entry is None:
                    return False
                dataset.append(entry)
                return True
            if isinstance(item, dict) and "text" in item:
                text_val = self._get_text_with_warning(item, line_num_hint, None)
                if text_val is None:
                    return False
                dataset.append(VlmCalibSample(text=text_val, image=None))
                return True
            get_logger().warning(
                "%s: Invalid format (expected string or dict with 'text'), skipping",
                line_num_hint,
            )
            return False

        if file_suffix == ".jsonl":
            # Line-by-line JSONL parsing
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        ok = _push_item(item, f"Line {line_num}")
                        if not ok:
                            continue
                    except json.JSONDecodeError as e:
                        get_logger().warning("Line %d: Invalid JSON - %s, skipping", line_num, e)
                        continue
                    except Exception as e:
                        get_logger().warning("Line %d: Error - %s, skipping", line_num, e)
                        continue
        else:
            # Whole-file JSON parsing for .json
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
            except json.JSONDecodeError as e:
                raise InvalidDatasetError(
                    f"Failed to parse JSON file: {file_path}",
                    action=f"Ensure the file is a valid JSON value (list/string/dict): {e}",
                ) from e
            except Exception as e:
                raise InvalidDatasetError(
                    f"Failed to read JSON file: {file_path}",
                    action="Ensure the file is accessible and valid JSON",
                ) from e

            if isinstance(content, list):
                for idx, item in enumerate(content, 1):
                    ok = _push_item(item, f"Index {idx}")
                    if not ok:
                        continue
            else:
                ok = _push_item(content, "Root")
                if not ok:
                    get_logger().warning("Root: No valid text entry found in JSON root, skipping")

        if not dataset:
            raise InvalidDatasetError(
                f"No valid text entries found in file: {file_path}",
                action='Check json/jsonl format: use plain string or {"text": "..."}',
            )

        get_logger().info("Loaded %d pure text samples from %s", len(dataset), file_path)
        return dataset

    def _load_from_directory(self, directory: Union[Path, str]) -> List[VlmCalibSample]:
        """
        Load dataset from directory (Type 2, 3, or 4).
        
        Logic:
        - If no json/jsonl: Type 2 (pure images with default text)
        - If has json/jsonl: Type 3 or 4 (images with custom text, or mixed)
        
        Args:
            directory: Path to directory
        
        Returns:
            List of VlmCalibSample: [VlmCalibSample(text="...", image="..." or None), ...]
        """
        directory = Path(get_valid_read_path(str(directory), is_dir=True, check_user_stat=True))

        # Find images in directory
        image_files = [
            f
            for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in self.supported_image_extensions
        ]
        if not image_files:
            raise InvalidDatasetError(
                f"No images files found in directory: {directory}",
                action="Ensure directory contains images (.jpg/.jpeg/.png)",
            )

        # Find json/jsonl files in directory
        json_jsonl_files = [
            f
            for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in self.supported_description_file_extensions
        ]

        if len(json_jsonl_files) > 1:
            raise InvalidDatasetError(
                f"Found multiple json/jsonl files in {directory}",
                action="Please provide only one json/jsonl file",
            )

        if json_jsonl_files:
            # Type 3 or 4: Has json/jsonl file
            json_jsonl_file = json_jsonl_files[0]
            file_suffix = json_jsonl_file.suffix.lower()
            get_logger().info(
                "Found json/jsonl file: %s, loading mixed/image-text dataset",
                json_jsonl_file.name,
            )
            return self._load_mixed_from_jsonl(
                json_jsonl_file,
                base_directory=directory,
                available_images=image_files,
                file_suffix=file_suffix,
            )
        # Type 2: Pure images, no json/jsonl
        get_logger().info(
            "No json/jsonl file found, using default text for %d images",
            len(image_files),
        )
        return [VlmCalibSample(text=self.default_text, image=str(p)) for p in sorted(image_files)]

    def _load_mixed_from_jsonl(
        self,
        jsonl_path: Union[Path, str],
        base_directory: Path,
        available_images: List[Path],
        file_suffix: str = ".jsonl",
    ) -> List[VlmCalibSample]:
        """
        Load mixed dataset from JSONL file (Type 3 or 4).
        
        JSONL can contain:
        - Image+text: {"image": "img.jpg", "text": "What is this?"}
        - Text-only: {"text": "Who are you?"} or just "Who are you?"
        
        For Type 3 (pure images with custom text):
        - All entries must have 'image' field
        - Image count must match available images
        
        For Type 4 (mixed):
        - Can have both image+text and text-only entries
        - Image entries must reference existing images
        
        Args:
            jsonl_path: Path to JSONL file
            base_directory: Base directory for resolving relative image paths
            available_images: List of available image files in directory
        
        Returns:
            List of VlmCalibSample: [VlmCalibSample(text="...", image="..." or None), ...]
        """
        jsonl_path = get_valid_read_path(str(jsonl_path))

        # Build image filename lookup for validation
        image_filenames = {img.name: img for img in available_images}

        dataset: List[VlmCalibSample] = []
        processed_images: Set[str] = set()  # 记录已处理的图片路径

        image_entries_count = 0
        text_only_count = 0

        def _accumulate(entry_tuple: Tuple[Optional[VlmCalibSample], int, int]):
            """处理JSON条目，收集图像到文本的映射"""
            nonlocal image_entries_count, text_only_count
            entry, is_image, is_text = entry_tuple
            if entry is None:
                return
            dataset.append(entry)
            image_entries_count += is_image
            text_only_count += is_text

            # 如果是图片条目，记录已处理的图片
            if is_image and entry.image:
                processed_images.add(str(entry.image))

        if file_suffix == ".jsonl":
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    entry_tuple = self._parse_jsonl_line(
                        line=line,
                        line_num=line_num,
                        base_directory=base_directory,
                        image_filenames=image_filenames,
                    )
                    _accumulate(entry_tuple)
        else:
            # .json: whole-file JSON; support single item or list of items
            try:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
            except json.JSONDecodeError as e:
                raise InvalidDatasetError(
                    f"Failed to parse JSON file: {jsonl_path}",
                    action=f"Ensure the file is a valid JSON value (list/dict/string): {e}",
                ) from e
            except Exception as e:
                raise InvalidDatasetError(
                    f"Failed to read JSON file: {jsonl_path}",
                    action="Ensure the file is accessible and valid JSON",
                ) from e

            def _parse_item(item: object, hint: str) -> Tuple[Optional[VlmCalibSample], int, int]:
                """解析JSON项目，返回 (entry, is_image, is_text)"""
                # Plain string -> text-only
                if isinstance(item, str):
                    return self._handle_plain_string(item, hint)

                # Dict cases
                if isinstance(item, dict):
                    has_image = "image" in item
                    has_text = "text" in item

                    # Text-only dict
                    if (not has_image) and has_text:
                        # 纯文本场景：无效文本时传入None表示跳过
                        text_val = self._get_text_with_warning(item, hint, None)
                        if text_val is None:
                            return None, 0, 0
                        return VlmCalibSample(text=text_val, image=None), 0, 1

                    # Image + optional text
                    if has_image:
                        image_ref = item["image"]
                        resolved = self._resolve_image_path(image_ref, base_directory, image_filenames, hint)
                        if resolved is None:
                            return None, 0, 0
                        
                        # 图文混合场景：无效文本时使用default_text
                        text_val = self._get_text_with_warning(item, hint, self.default_text)
                        return VlmCalibSample(text=text_val, image=resolved), 1, 0

                    get_logger().warning("%s: Missing both 'image' and 'text' fields, skipping", hint)
                    return None, 0, 0

                get_logger().warning("%s: Invalid format (expected string or dict), skipping", hint)
                return None, 0, 0

            if isinstance(content, list):
                for idx, item in enumerate(content, 1):
                    _accumulate(_parse_item(item, f"Index {idx}"))
            else:
                _accumulate(_parse_item(content, "Root"))
        
        # 以目录中的图片为标准构建最终数据集
        # 1.JSON/JSONL中的图片（需存在于校准目录），校准图片+custom_text
        # 2.校准目录中，在JSON/JSONL中不存在，校准图片+default_text
        for img_path in sorted(available_images):
            abs_img_path = str(img_path.absolute())

            # 在JSON/JSONL已处理，跳过
            if abs_img_path in processed_images:
                get_logger().debug("Image %s already processed, skipping", img_path.name)
                continue
            
            # 图片不在JSON中，使用默认文本
            get_logger().warning("Image %s not found in JSON, using default text", abs_img_path)
            dataset.append(VlmCalibSample(text=self.default_text, image=abs_img_path))
            # 图像条数计数
            image_entries_count += 1

        if not dataset:
            raise InvalidDatasetError(
                f"No valid entries found in JSONL file: {jsonl_path}",
                action='Check each line: allow plain string, {"text": ...}, or {"image": ..., "text": ...}',
            )

        # Validate Type 3 vs Type 4
        if text_only_count == 0:
            # Type 3: Pure images with custom text
            get_logger().info(
                "Loading %d images with custom text referring %s.",
                image_entries_count,
                jsonl_path,
            )
            # Check if image count matches
            if image_entries_count != len(available_images):
                get_logger().warning(
                    "Image count mismatch: JSONL has %d entries, directory has %d images",
                    image_entries_count,
                    len(available_images),
                )
        else:
            # Type 4: Mixed dataset
            get_logger().info(
                "Loading %d image+text samples, %d text-only samples referring %s.",
                image_entries_count,
                text_only_count,
                jsonl_path,
            )

        return dataset

    def _parse_jsonl_line(
        self,
        line: str,
        line_num: int,
        base_directory: Path,
        image_filenames: Dict[str, Path],
    ) -> Tuple[Optional[VlmCalibSample], int, int]:
        """
        Parse a single jsonl line and return (entry, is_image_entry, is_text_entry).
        Returns (None, 0, 0) if the line is invalid or should be skipped.
        """
        line = line.strip()
        if not line:
            return None, 0, 0

        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            get_logger().warning("Line %d: Invalid JSON - %s, skipping", line_num, e)
            return None, 0, 0
        except Exception as e:
            get_logger().warning("Line %d: Error - %s, skipping", line_num, e)
            return None, 0, 0

        # Case 1: plain string -> text-only
        if isinstance(item, str):
            return self._handle_plain_string(item, f"Line {line_num}")
    
        # Case 2: dict
        if isinstance(item, dict):
            hint = f"Line {line_num}"
            if "image" in item:
                resolved = self._resolve_image_path(item["image"], base_directory, image_filenames, hint=hint)
                if resolved is None:
                    return None, 0, 0
                # 图文混合场景：无效文本时使用default_text
                text_val = self._get_text_with_warning(item, hint, self.default_text)
                return VlmCalibSample(text=text_val, image=resolved), 1, 0

            if "text" in item:
                # 纯文本场景：只有text字段，如果text无效则跳过
                text_val = self._get_text_with_warning(item, hint, None)
                if text_val is None:
                    return None, 0, 0
                return VlmCalibSample(text=text_val, image=None), 0, 1

            get_logger().warning("Line %d: Missing both 'image' and 'text' fields, skipping", line_num)
            return None, 0, 0

        get_logger().warning("Line %d: Invalid format (expected string or dict), skipping", line_num)
        return None, 0, 0

    def _resolve_image_path(
        self,
        image_ref: object,
        base_directory: Path,
        image_filenames: Dict[str, Path],
        hint: str,
    ) -> Optional[str]:
        """
        Resolve and validate image path for mixed json/jsonl entries.
        Returns validated path (str) or None if invalid.
        """
        if not image_ref or not isinstance(image_ref, str):
            get_logger().warning("%s: Image reference is empty or invalid type, skipping", hint)
            return None

        image_ref = image_ref.strip()
        if not image_ref:
            get_logger().warning("%s: Image reference is empty after stripping, skipping", hint)
            return None

        try:
            image_path = Path(image_ref)
        except Exception as e:
            get_logger().warning("%s: Invalid image reference %r: %s, skipping", hint, image_ref, e)
            return None

        # 检查文件名后缀
        if image_path.suffix.lower() not in self.supported_image_extensions:
            get_logger().warning(
                "%s: Image link %s has unsupported suffix (only %s are allowed); entry discarded",
                hint,
                image_path,
                self.supported_image_extensions,
            )
            return None

        if not image_path.is_absolute():
            if image_path.name in image_filenames:
                resolved_image = image_filenames[image_path.name]
            else:
                resolved_image = base_directory / image_ref
                if not resolved_image.exists():
                    get_logger().warning("%s: Image %r not found in directory, skipping", hint, image_ref)
                    return None
        else:
            resolved_image = image_path
            if not resolved_image.exists():
                get_logger().warning("%s: Image path %r does not exist, skipping", hint, image_ref)
                return None

        # 检查文件是否在校准目录
        if resolved_image.name not in image_filenames or resolved_image != image_filenames[resolved_image.name]:
            get_logger().warning("%s: Image %s not found in directory, skipping", hint, resolved_image)
            return None

        try:
            return get_valid_read_path(str(resolved_image))
        except Exception as e:
            get_logger().warning("%s: Failed to validate image path %s: %s, skipping", hint, resolved_image, e)
            return None

    def _get_text_with_warning(self, item: dict, hint: str, default_value: Optional[str]) -> Optional[str]:
        """
        获取文本值
        Args:
            item: 包含text字段的字典
            hint: 日志提示信息前缀
            default_value: 文本无效时使用的默认值，None表示跳过
        Returns:
            有效文本、默认值或None（default_value为None且文本无效时）
        """
        # 检查text字段是否存在
        if "text" not in item:
            if default_value is None:
                get_logger().warning("%s: Missing 'text' field, skipping", hint)
                return None
            get_logger().warning("%s: Missing 'text' field, using default text", hint)
            return default_value

        text_val = item["text"]

        # 检查是否为None
        if text_val is None:
            if default_value is None:
                get_logger().warning("%s: 'text' field is None, skipping", hint)
                return None
            get_logger().warning("%s: 'text' field is None, using default text", hint)
            return default_value

        # 转换为字符串并去除空白
        if not isinstance(text_val, str):
            try:
                text_val = str(text_val)
            except Exception as e:
                if default_value is None:
                    get_logger().warning("%s: Cannot convert 'text' to string: %s, skipping", hint, e)
                    return None
                get_logger().warning("%s: Cannot convert 'text' to string: %s, using default text", hint, e)
                return default_value

        # 检查去除空白后是否为空
        stripped = text_val.strip()
        if not stripped:
            if default_value is None:
                get_logger().warning("%s: 'text' field is empty/whitespace, skipping", hint)
                return None
            get_logger().warning("%s: 'text' field is empty/whitespace, using default text", hint)
            return default_value

        return stripped

    def _handle_plain_string(self, item: str, hint: str) -> Tuple[Optional[VlmCalibSample], int, int]:
        """处理纯字符串情况"""
        # 对于纯字符串，如果是空或只有空白字符，跳过
        stripped_str = item.strip()
        if not stripped_str:
            if item:  # 原始值非空但去除空白后为空
                get_logger().warning("%s: Text string contains only whitespace, skipping", hint)
            else:  # 原始值就是空字符串
                get_logger().warning("%s: Text string is empty, skipping", hint)
            return None, 0, 0
        return VlmCalibSample(text=stripped_str, image=None), 0, 1
