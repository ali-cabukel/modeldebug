"""Individual diagnostic checks, grouped by category.

Each check module exposes an ``ALL_CHECKS`` list of callables with the
signature ``(ctx: CheckContext) -> list[Issue]``. New check categories
(labels, imbalance, drift, overfitting) should follow the same pattern
and be registered in ``modeldebug.core.DEFAULT_CHECKS``.
"""

from . import leakage

__all__ = ["leakage"]
