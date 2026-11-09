#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""IR graph edge definition (shared by router and tasks)."""

from __future__ import annotations

from dataclasses import dataclass, field

from msmodelslim.core.convert.types import IRKind, LossLevel


@dataclass
class RouteConstraints:
    """Explicit routing requirements from convert_rules."""

    required_nodes: list[IRKind] = field(default_factory=list)
    forbidden_edges: list[tuple[IRKind, IRKind]] = field(default_factory=list)
    explicit_route: list[IRKind] | None = None


@dataclass(frozen=True)
class TransformEdge:
    """One edge in the convert IR graph."""

    src_ir: IRKind
    dst_ir: IRKind
    processor_name: str
    loss_level: LossLevel = LossLevel.LOSSY
    requirements: tuple[str, ...] = ()
    cost: int = 1
