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
from typing import Any, Dict, List, Optional, Set, Union

from msmodelslim.utils.exception import InvalidDatasetError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.security import get_valid_read_path

from .base_dataset_loader import BaseDatasetLoader
from .vlm_dataset_loader import VlmCalibSample


def _non_empty_text(v: Any, hint: str) -> str:
    """
    Validate that a string is non-null and non-empty after stripping whitespace.
    """
    if not isinstance(v, str) or not v.strip():
        raise InvalidDatasetError(
            f"{hint}: field 'text' must be a non-empty string",
            action="Please provide a non-empty 'text' for each entry.",
        )
    return v.strip()


def _resolve_media_path(
    raw: Any,
    base_dir: Path,
    field_name: str,
    allowed_exts: Set[str],
    hint: str,
) -> Optional[str]:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidDatasetError(
            f"{hint}: field '{field_name}' must be a non-empty string when provided",
            action=f"Please fix '{field_name}' for this entry.",
        )

    p = Path(raw.strip())
    if not p.is_absolute():
        p = (base_dir / p).resolve()

    if p.suffix.lower() not in allowed_exts:
        raise InvalidDatasetError(
            f"{hint}: {field_name} has unsupported suffix {p.suffix!r}",
            action=f"Allowed suffixes for {field_name}: {sorted(allowed_exts)}",
        )
    if not p.exists() or not p.is_file():
        raise InvalidDatasetError(
            f"{hint}: {field_name} path does not exist or is not a file: {p}",
            action="Please check the path in index file and ensure it exists.",
        )

    return get_valid_read_path(str(p), is_dir=False, check_user_stat=True)


class JsonlDatasetLoader(BaseDatasetLoader):
    """
    JsonlDatasetLoader

    Supports:
    - dataset_name is an index.json or index.jsonl file (exact filename)

    Behavior:
    - Strict: each entry must be a JSON object containing at least "text".
    - Optional fields: image/audio/video. If present, paths must exist and file suffix must be supported.
    """
    SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3"}
    SUPPORTED_VIDEO_EXTENSIONS = {".mp4"}

    def __init__(
        self,
        default_text: Optional[str] = None,
    ):
        super().__init__(default_text=default_text)
        self.supported_image_extensions = self.SUPPORTED_IMAGE_EXTENSIONS
        self.supported_audio_extensions = self.SUPPORTED_AUDIO_EXTENSIONS
        self.supported_video_extensions = self.SUPPORTED_VIDEO_EXTENSIONS

    def is_support(self, resolved_path: Path) -> bool:
        if not resolved_path.is_file():
            return False
        return resolved_path.name in {"index.json", "index.jsonl"}

    def load_data(self, resolved_path: Path) -> List[VlmCalibSample]:
        resolved_path = Path(get_valid_read_path(str(resolved_path), is_dir=False, check_user_stat=True))
        base_dir = resolved_path.parent

        entries: List[Dict[str, Any]] = []
        if resolved_path.suffix.lower() == ".jsonl":
            # index.jsonl
            with open(resolved_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise InvalidDatasetError(
                            f"Line {line_num}: invalid JSON: {e}",
                            action="Please fix the JSONL formatting.",
                        ) from e
                    if not isinstance(item, dict):
                        raise InvalidDatasetError(
                            f"Line {line_num}: each JSONL line must be an object",
                            action="Each line must be like: {\"text\": \"...\", \"audio\": \"...\"}.",
                        )
                    entries.append(item)
        else:
            # index.json
            with open(resolved_path, "r", encoding="utf-8") as f:
                try:
                    content = json.load(f)
                except json.JSONDecodeError as e:
                    raise InvalidDatasetError(
                        f"Failed to parse JSON file: {resolved_path}",
                        action=f"Please fix JSON formatting: {e}",
                    ) from e
            if isinstance(content, dict):
                entries = [content]
            elif isinstance(content, list):
                if not all(isinstance(x, dict) for x in content):
                    raise InvalidDatasetError(
                        "index.json must be an object or a list of objects",
                        action="Please ensure each entry is a JSON object with at least 'text'.",
                    )
                entries = content
            else:
                raise InvalidDatasetError(
                    "index.json must be an object or a list of objects",
                    action="Please provide a JSON object or JSON list.",
                )

        if not entries:
            raise InvalidDatasetError(
                f"No valid entries found in {resolved_path}",
                action="Please add at least one entry with 'text' field.",
            )

        dataset: List[VlmCalibSample] = []
        for i, item in enumerate(entries, 1):
            hint = f"{resolved_path.name} entry {i}"
            text = _non_empty_text(item.get("text"), hint) if item.get("text") else self.default_text
            image = _resolve_media_path(
                item.get("image"), base_dir, "image", self.supported_image_extensions, hint
            )
            audio = _resolve_media_path(
                item.get("audio"), base_dir, "audio", self.supported_audio_extensions, hint
            )
            video = _resolve_media_path(
                item.get("video"), base_dir, "video", self.supported_video_extensions, hint
            )
            dataset.append(VlmCalibSample(text=text, image=image, audio=audio, video=video))

        get_logger().info("Loaded %d entries from %s", len(dataset), resolved_path)
        return dataset
