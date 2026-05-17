"""Pipeline run modes and execution-plan data structures.

PROD mode runs the live weekly pipeline against upcoming games. TEST mode
replays a known historical week so the end-to-end pipeline can be verified
deterministically -- e.g. during the offseason when no upcoming games exist.

These are plain dataclasses (no Dagster dependency) so both the Dagster assets
and the predict.py CLI share one definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PipelineMode(str, Enum):
    """Execution mode for a pipeline run. PROD is the default."""

    PROD = "prod"
    TEST = "test"


@dataclass
class PipelineRunConfig:
    """Inputs that parameterize a single pipeline run.

    Args:
        mode: PROD (live) or TEST (replay a historical week).
        week: NFL week. Required in TEST mode; derived from the schedule in PROD.
        season: NFL season year. Required in TEST mode.

    Raises:
        ValueError: if TEST mode is selected without both week and season.
    """

    mode: PipelineMode = PipelineMode.PROD
    week: int | None = None
    season: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = PipelineMode(self.mode)
        if self.mode is PipelineMode.TEST and (self.week is None or self.season is None):
            raise ValueError(
                "TEST mode requires both 'week' and 'season' "
                "(e.g. week=10, season=2025)."
            )

    @property
    def label(self) -> str:
        """Human-readable one-line description of this run."""
        if self.mode is PipelineMode.TEST:
            return f"TEST (season {self.season}, week {self.week})"
        return "PROD"


@dataclass
class CheckResult:
    """Outcome of a single deterministic preflight question."""

    name: str
    passed: bool
    detail: str


@dataclass
class StepDecision:
    """Whether a pipeline step should run for this configuration, and why."""

    name: str
    run: bool
    reason: str


@dataclass
class ExecutionPlan:
    """Result of preflight: validation outcomes plus the constructed step plan."""

    config: PipelineRunConfig
    checks: list[CheckResult] = field(default_factory=list)
    steps: list[StepDecision] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True only if every validation check passed."""
        return all(c.passed for c in self.checks)

    def steps_to_run(self) -> list[str]:
        """Names of the steps the plan says should execute."""
        return [s.name for s in self.steps if s.run]

    def steps_skipped(self) -> list[str]:
        """Names of the steps the plan says should be skipped."""
        return [s.name for s in self.steps if not s.run]

    def render(self) -> str:
        """Multi-line human-readable summary of checks and step decisions."""
        lines = [f"Execution plan -- {self.config.label}", "  Validation checks:"]
        for c in self.checks:
            lines.append(f"    [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
        lines.append("  Steps:")
        for s in self.steps:
            lines.append(f"    [{'RUN ' if s.run else 'SKIP'}] {s.name}: {s.reason}")
        return "\n".join(lines)
