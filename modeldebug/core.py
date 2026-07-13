"""Debugger: the main entry point for running diagnostic checks."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

from .checks import leakage
from .issue import CheckContext, Issue
from .report import Report

CheckFn = Callable[[CheckContext], List[Issue]]

# Checks run by default, in order. Add new categories here as they land
# (labels, imbalance, drift, overfitting, ...).
DEFAULT_CHECKS: List[CheckFn] = [
    *leakage.ALL_CHECKS,
]


class Debugger:
    """Runs a battery of diagnostic checks against a model and its data.

    Example:
        >>> db = Debugger(model, X_train, y_train, X_test, y_test)
        >>> report = db.run()
        >>> report.show()

    Args:
        model: A fitted model exposing ``.predict()`` (and ``.predict_proba()``
            for classification, if available). Framework-agnostic — anything
            with a sklearn-style interface works.
        X_train, y_train: Training features/labels.
        X_test, y_test: Optional held-out features/labels. Several checks
            (leakage, drift) require these and are skipped without them.
        feature_names: Optional column names, used when X is a plain array.
        task: "classification" or "regression".
        checks: Optional custom list of check functions to run instead of
            the default battery.
    """

    def __init__(
        self,
        model: Any,
        X_train: Any,
        y_train: Any,
        X_test: Optional[Any] = None,
        y_test: Optional[Any] = None,
        feature_names: Optional[Sequence[str]] = None,
        task: str = "classification",
        checks: Optional[List[CheckFn]] = None,
    ):
        self.ctx = CheckContext(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            task=task,
        )
        self.checks = checks if checks is not None else list(DEFAULT_CHECKS)

    def run(self, verbose: bool = False) -> Report:
        """Run all configured checks and return an aggregated Report."""
        all_issues: List[Issue] = []
        for check_fn in self.checks:
            try:
                issues = check_fn(self.ctx)
                all_issues.extend(issues)
                if verbose:
                    print(f"[modeldebug] {check_fn.__name__}: {len(issues)} issue(s)")
            except Exception as exc:  # noqa: BLE001 - surface as an Issue instead of crashing the whole run
                if verbose:
                    print(f"[modeldebug] {check_fn.__name__} failed: {exc}")
        return Report(all_issues)
