#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Convert 阶段的设备解析与 worker 并发上限工具。

``worker_backend=process`` 时全程固定 CPU；thread 后端可解析到 NPU。
"""

from __future__ import annotations

import torch

from msmodelslim.utils.logging import get_logger

logger = get_logger()


def npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def resolve_worker_device(worker_device: str | None) -> str:
    """
    将 ``parallel.worker_device`` 解析为 safetensors / torch 可用的设备字符串。

    支持：
      - ``auto``：有 NPU 时用 ``npu:0``，否则 ``cpu``
      - ``cpu`` / ``npu``：简写；``npu`` 等价于 ``npu:0``
      - ``npu:0`` / ``npu:1`` 等：显式指定单卡
    """
    spec = (worker_device or "auto").strip()
    lowered = spec.lower()

    if lowered == "auto":
        if npu_available():
            return "npu:0"
        logger.warning("No NPU available; convert weights on CPU instead")
        return "cpu"

    if lowered == "cpu":
        return "cpu"

    if lowered == "npu":
        if not npu_available():
            logger.warning("worker_device=npu but NPU unavailable; falling back to CPU")
            return "cpu"
        return "npu:0"

    if lowered.startswith("npu"):
        if not npu_available():
            logger.warning("worker_device=%r but NPU unavailable; falling back to CPU", spec)
            return "cpu"
        return spec

    raise ValueError(
        f"Unsupported worker_device {worker_device!r}; "
        "expected cpu, npu, auto, or an explicit npu:<index> device string"
    )


def effective_convert_workers(
    max_workers: int,
    resolved_worker_device: str,
    npu_max_workers: int,
) -> int:
    """
    NPU 上多 worker 并发会把多张 2D 权重 + 量化中间张量同时驻留显存，易 OOM。
    在 accelerator 模式下将组内并发上限压到 ``npu_max_workers``（默认 1）。
    """
    workers = max(1, max_workers)
    if resolved_worker_device == "cpu":
        return workers
    cap = max(1, npu_max_workers)
    return min(workers, cap)
