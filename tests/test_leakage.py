import numpy as np
import pandas as pd
import pytest

from modeldebug import Debugger, Severity
from modeldebug.issue import CheckContext
from modeldebug.checks.leakage import (
    check_duplicate_rows,
    check_target_leakage,
    check_near_duplicate_rows,
)


class DummyModel:
    """Minimal sklearn-style stub; checks in this suite don't need real predictions."""

    def predict(self, X):
        return np.zeros(len(X))


def _ctx(X_train, y_train, X_test=None, y_test=None, task="classification"):
    return CheckContext(
        model=DummyModel(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task=task,
    )


def test_duplicate_rows_detected():
    rng = np.random.default_rng(0)
    train = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    # Test set contains 5 exact copies of train rows.
    test = pd.concat([train.iloc[:5], pd.DataFrame(rng.normal(size=(20, 3)), columns=["a", "b", "c"])], ignore_index=True)

    ctx = _ctx(train, np.zeros(len(train)), test, np.zeros(len(test)))
    issues = check_duplicate_rows(ctx)

    assert len(issues) == 1
    assert issues[0].check_name == "leakage.duplicate_rows"
    assert issues[0].severity in (Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)
    assert len(issues[0].affected_rows) == 5


def test_duplicate_rows_none_when_no_overlap():
    rng = np.random.default_rng(1)
    train = pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])
    test = pd.DataFrame(rng.normal(loc=100, size=(20, 3)), columns=["a", "b", "c"])

    ctx = _ctx(train, np.zeros(len(train)), test, np.zeros(len(test)))
    issues = check_duplicate_rows(ctx)
    assert issues == []


def test_duplicate_rows_skipped_without_test_set():
    train = pd.DataFrame({"a": [1, 2, 3]})
    ctx = _ctx(train, np.zeros(3))
    assert check_duplicate_rows(ctx) == []


def test_target_leakage_detects_proxy_feature():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=200).astype(float)
    train = pd.DataFrame(
        {
            "leaky": y + rng.normal(scale=0.001, size=200),  # near-perfect proxy
            "noise": rng.normal(size=200),
        }
    )
    ctx = _ctx(train, y)
    issues = check_target_leakage(ctx)

    assert any(i.check_name == "leakage.target_correlation" and "leaky" in i.affected_columns for i in issues)
    assert not any("noise" in (i.affected_columns or []) for i in issues)


def test_target_leakage_skips_multiclass():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 5, size=100).astype(float)
    train = pd.DataFrame({"x": y})  # would be perfectly correlated if not skipped
    ctx = _ctx(train, y, task="classification")
    assert check_target_leakage(ctx) == []


def test_near_duplicate_rows_detects_close_points():
    rng = np.random.default_rng(4)
    train = pd.DataFrame(rng.normal(size=(50, 2)), columns=["a", "b"])
    # Test rows are train rows plus tiny noise -> distance ~0 after scaling.
    near = train.iloc[:3] + 1e-6
    test = pd.concat([near, pd.DataFrame(rng.normal(loc=50, size=(10, 2)), columns=["a", "b"])], ignore_index=True)

    ctx = _ctx(train, np.zeros(len(train)), test, np.zeros(len(test)))
    issues = check_near_duplicate_rows(ctx, threshold=0.01)

    assert len(issues) == 1
    assert issues[0].metric == pytest.approx(100 * 3 / len(test))


def test_debugger_run_end_to_end():
    rng = np.random.default_rng(5)
    train = pd.DataFrame(rng.normal(size=(30, 2)), columns=["a", "b"])
    test = pd.concat([train.iloc[:2], pd.DataFrame(rng.normal(loc=20, size=(10, 2)), columns=["a", "b"])], ignore_index=True)

    db = Debugger(
        model=DummyModel(),
        X_train=train,
        y_train=np.zeros(len(train)),
        X_test=test,
        y_test=np.zeros(len(test)),
    )
    report = db.run()

    assert len(report) >= 1
    assert report.issues[0].severity.value >= report.issues[-1].severity.value
    assert len(report.summary()) > 0
