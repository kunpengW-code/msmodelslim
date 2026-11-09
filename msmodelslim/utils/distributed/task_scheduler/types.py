#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""DTS 任务与执行记录等类型定义（与调度 backend 无关）。"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from torch import nn


@dataclass
class _TaskSpec:
    task_id: str
    dependencies: List[str] = field(default_factory=list)
    fn: Optional[Callable[..., Any]] = None
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    parallel: bool = True
    semantic_hash: str = ""


@dataclass
class Task:
    spec: _TaskSpec
    sync_fn: Optional[Callable[["TaskExecutionRecord", "TaskSyncContext"], Any]] = None


@dataclass
class TaskExecutionRecord:
    """单任务执行摘要。"""

    task_id: str
    executor_rank: int
    result: Any = None
    dependencies: List[str] = field(default_factory=list)
    exception: Optional[str] = None
    exec_time_s: float = 0.0
    sync_time_s: float = 0.0
    sync_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSyncContext:
    model: nn.Module
    rank: int
    world_size: int
