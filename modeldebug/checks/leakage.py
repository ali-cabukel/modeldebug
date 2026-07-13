"""Checks for train/test leakage and suspicious feature-target relationships."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..issue import CheckContext, Issue, Severity


def _to_dataframe(X, feature_names=None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    X = np.asarray(X)
    cols = feature_names if feature_names is not None else [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols)


def check_duplicate_rows(ctx: CheckContext) -> List[Issue]:
    """Flag exact-duplicate rows shared between train and test sets.

    Exact duplicates across train/test mean the model may have effectively
    "seen" test examples during training, inflating reported performance.
    """
    if ctx.X_test is None:
        return []

    issues: List[Issue] = []
    train_df = _to_dataframe(ctx.X_train, ctx.feature_names)
    test_df = _to_dataframe(ctx.X_test, ctx.feature_names)

    # Hash each row to compare efficiently.
    train_hashes = pd.util.hash_pandas_object(train_df, index=False)
    test_hashes = pd.util.hash_pandas_object(test_df, index=False)

    train_hash_set = set(train_hashes.values)
    overlap_mask = test_hashes.isin(train_hash_set)
    n_overlap = int(overlap_mask.sum())

    if n_overlap > 0:
        pct = 100 * n_overlap / len(test_df)
        severity = Severity.CRITICAL if pct > 5 else Severity.HIGH if pct > 1 else Severity.MEDIUM
        affected_rows = test_df.index[overlap_mask].tolist()
        issues.append(
            Issue(
                check_name="leakage.duplicate_rows",
                title=f"{n_overlap} test rows ({pct:.2f}%) are exact duplicates of train rows",
                description=(
                    "Rows in the test set are byte-for-byte identical to rows in the "
                    "training set. This inflates test performance because the model "
                    "has memorized these exact examples during training. Common causes: "
                    "splitting after deduplication was needed, or splitting a dataset "
                    "that already contained repeated records."
                ),
                severity=severity,
                affected_rows=affected_rows,
                metric=pct,
                suggested_fix=(
                    "Deduplicate the full dataset before splitting into train/test, "
                    "or use a grouped split (e.g. GroupKFold) if duplicates share a "
                    "natural entity key (user_id, session_id, etc.)."
                ),
            )
        )
    return issues


def check_target_leakage(ctx: CheckContext, correlation_threshold: float = 0.95) -> List[Issue]:
    """Flag features suspiciously highly correlated with the target.

    A feature nearly perfectly correlated with the label is often a proxy for
    the label itself (e.g. a post-outcome field), not a genuine predictor
    available at inference time.
    """
    issues: List[Issue] = []
    train_df = _to_dataframe(ctx.X_train, ctx.feature_names)

    y = np.asarray(ctx.y_train)
    if ctx.task == "classification" and len(np.unique(y)) > 2:
        # Multi-class: skip simple correlation check, not directly meaningful.
        return issues

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return issues

    y_numeric = y.astype(float)
    for col in numeric_cols:
        col_values = train_df[col].astype(float).values
        if np.std(col_values) == 0 or np.std(y_numeric) == 0:
            continue
        corr = np.corrcoef(col_values, y_numeric)[0, 1]
        if np.isnan(corr):
            continue
        if abs(corr) >= correlation_threshold:
            issues.append(
                Issue(
                    check_name="leakage.target_correlation",
                    title=f"Feature '{col}' has {corr:.3f} correlation with the target",
                    description=(
                        f"'{col}' is almost perfectly correlated with the target variable. "
                        "This often indicates the feature is derived from the target, "
                        "recorded after the target was known, or is otherwise unavailable "
                        "at real prediction time."
                    ),
                    severity=Severity.HIGH,
                    affected_columns=[col],
                    metric=float(corr),
                    suggested_fix=(
                        f"Inspect how '{col}' is computed and confirm it would be known "
                        "at inference time, before the outcome occurs. Remove it if it "
                        "leaks the label."
                    ),
                )
            )
    return issues


def check_near_duplicate_rows(ctx: CheckContext, threshold: float = 0.0) -> List[Issue]:
    """Flag near-duplicate rows between train/test using normalized L2 distance.

    Only runs on numeric columns; a lightweight complement to the exact-hash
    check above for cases with floating point noise.
    """
    if ctx.X_test is None:
        return []

    train_df = _to_dataframe(ctx.X_train, ctx.feature_names).select_dtypes(include=[np.number])
    test_df = _to_dataframe(ctx.X_test, ctx.feature_names).select_dtypes(include=[np.number])
    common_cols = [c for c in train_df.columns if c in test_df.columns]
    if not common_cols:
        return []

    # Guard against expensive O(n*m) comparison on large datasets.
    if len(train_df) * len(test_df) > 20_000_000:
        return []

    train_arr = train_df[common_cols].to_numpy(dtype=float)
    test_arr = test_df[common_cols].to_numpy(dtype=float)

    # Normalize so distances are comparable across features of different scales.
    stds = train_arr.std(axis=0)
    stds[stds == 0] = 1.0
    train_norm = train_arr / stds
    test_norm = test_arr / stds

    n_close = 0
    close_rows: List[int] = []
    for i, row in enumerate(test_norm):
        dists = np.linalg.norm(train_norm - row, axis=1)
        if dists.min() <= threshold + 1e-9:
            n_close += 1
            close_rows.append(test_df.index[i])

    if n_close > 0:
        pct = 100 * n_close / len(test_df)
        issues = [
            Issue(
                check_name="leakage.near_duplicate_rows",
                title=f"{n_close} test rows ({pct:.2f}%) are near-identical to train rows",
                description=(
                    "After normalizing feature scales, these test rows sit essentially "
                    "on top of training rows. This can still leak information even "
                    "without exact duplication."
                ),
                severity=Severity.MEDIUM,
                affected_rows=close_rows,
                metric=pct,
                suggested_fix=(
                    "Investigate whether these near-duplicates share an underlying "
                    "entity (same user/session) and consider a grouped split."
                ),
            )
        ]
        return issues
    return []


ALL_CHECKS = [check_duplicate_rows, check_target_leakage, check_near_duplicate_rows]
