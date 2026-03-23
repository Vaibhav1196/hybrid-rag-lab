from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class StageTiming:
    """Latency measurement for a single pipeline stage."""

    stage: str
    duration_ms: float


@dataclass(slots=True)
class PipelineTrace:
    """Basic observability record for a pipeline run."""

    stage_timings: list[StageTiming] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_stage(self, stage: str, duration_ms: float) -> None:
        self.stage_timings.append(StageTiming(stage=stage, duration_ms=duration_ms))

    @property
    def total_duration_ms(self) -> float:
        return sum(item.duration_ms for item in self.stage_timings)
