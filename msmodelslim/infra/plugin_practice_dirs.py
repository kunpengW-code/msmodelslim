# -*- coding: UTF-8 -*-
"""
Discover practice directories from plugins via entry point msmodelslim.naive_quantization.practice_dirs.

Each entry point is a callable (e.g. module:function) that returns a Path or a list of Paths
pointing to practice root directories (same layout as lab_practice: pedigree subdirs with YAMLs).
"""
import sys
from importlib.metadata import entry_points
from pathlib import Path
from typing import List

from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.security import get_valid_read_path

PRACTICE_DIRS_ENTRY_POINT = "msmodelslim.naive_quantization.practice_dirs"


def discover_plugin_practice_dirs() -> List[Path]:
    """
    Discover all plugin-provided practice root directories via entry point.
    Returns a list of valid, existing directory Paths (read-only).
    """
    if sys.version_info >= (3, 10):
        eps = entry_points().select(group=PRACTICE_DIRS_ENTRY_POINT)
    else:
        eps = entry_points().get(PRACTICE_DIRS_ENTRY_POINT, [])

    result: List[Path] = []
    for ep in eps:
        try:
            callable_obj = ep.load()
            value = callable_obj()
            if value is None:
                continue
            paths = [value] if isinstance(value, (Path, str)) else value
            for p in paths:
                path = Path(p) if isinstance(p, str) else p
                if not path.exists() or not path.is_dir():
                    continue
                get_valid_read_path(str(path), is_dir=True)
                result.append(path.resolve())
        except Exception as e:  # noqa: B902
            get_logger().warning(
                "Failed to load practice dir from plugin %s: %s", ep.name, e
            )
    if result:
        get_logger().info(
            "Discovered %d plugin practice dir(s): %s",
            len(result),
            [str(p) for p in result],
        )
    return result
