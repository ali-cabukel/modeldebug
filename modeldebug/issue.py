"""Core data structures used across all checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence


class Severity(Enum):
    """Severity level of a detected issue, ordered low -> high."""

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __lt__(self, other: "Severity") -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.value < other.value

    def __str__(self) -> str:
        return self.name


@dataclass
class Issue:
    """A single detected problem with the model/data.

    Attributes:
        check_name: Identifier of the check that produced this issue
            (e.g. "leakage.duplicate_rows").
        title: Short human-readable summary.
        description: Longer explanation of what was found and why it matters.
        severity: How serious the issue is.
        affected_rows: Optional list/array of row indices implicated in the issue.
        affected_columns: Optional list of column names implicated in the issue.
        metric: Optional numeric value backing the finding (e.g. overlap %).
        suggested_fix: Actionable advice for resolving the issue.
        extra: Free-form dict for check-specific details (e.g. per-row scores).
    """

    check_name: str
    title: str
    description: str
    severity: Severity
    affected_rows: Optional[Sequence[int]] = None
    affected_columns: Optional[Sequence[str]] = None
    metric: Optional[float] = None
    suggested_fix: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<Issue [{self.severity}] {self.check_name}: {self.title}>"

    def to_dict(self) -> dict:
        return {
            "check_name": self.check_name,
            "title": self.title,
            "description": self.description,
            "severity": str(self.severity),
            "affected_rows": list(self.affected_rows) if self.affected_rows is not None else None,
            "affected_columns": list(self.affected_columns) if self.affected_columns is not None else None,
            "metric": self.metric,
            "suggested_fix": self.suggested_fix,
            "extra": self.extra,
        }


@dataclass
class CheckContext:
    """Bundle of everything a check function might need.

    Not every field is required for every check; checks should defensively
    handle missing/None fields (e.g. a check needing X_test should raise a
    clear error if X_test wasn't provided, rather than crashing obscurely).
    """

    model: Any
    X_train: Any
    y_train: Any
    X_test: Optional[Any] = None
    y_test: Optional[Any] = None
    feature_names: Optional[Sequence[str]] = None
    task: str = "classification"  # or "regression"
