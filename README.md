# modeldebug

Diagnostic checks that explain **why your ML model is failing** — instead of manually
digging through notebooks, run a battery of checks and get a prioritized report.

```bash
pip install modeldebug
```

## Quick start

```python
from modeldebug import Debugger

db = Debugger(model, X_train, y_train, X_test, y_test)
report = db.run()
report.show()       # renders a table in Jupyter, or prints to terminal
```

```python
for issue in report.issues:
    print(issue.severity, issue.title, issue.suggested_fix)
```

## What it checks (v0.1)

- **Leakage**
  - Exact duplicate rows shared between train and test
  - Near-duplicate rows (after scaling) between train and test
  - Features suspiciously highly correlated with the target

More check categories (label errors, class imbalance, distribution drift,
overfitting signals) are planned — see [Roadmap](#roadmap).

## Why

Most eval tooling tells you *what* your metric is. `modeldebug` tries to tell
you *why* it's wrong — surfacing the data problems that inflate or deflate
reported performance before you waste time tuning hyperparameters.

## Design

- **Framework-agnostic**: works with anything exposing a sklearn-style
  `.predict()` — sklearn, xgboost, lightgbm, or a thin wrapper around a
  PyTorch model.
- **Tabular-first**: v0.1 targets tabular classification/regression.
- **Extensible**: each check is a plain function `(CheckContext) -> list[Issue]`.
  Add your own and pass `checks=[...]` to `Debugger`.

## Roadmap

- [ ] Label-quality checks (mislabeled example detection)
- [ ] Class imbalance + misleading-metric warnings
- [ ] Train/test distribution shift (KS test, PSI)
- [ ] Overfitting signals (train/val gap, learning curves)
- [ ] Segment-level error clustering

## Contributing

Issues and PRs welcome. Adding a new check is the easiest way to contribute:
drop a function following the `(ctx) -> list[Issue]` signature into
`modeldebug/checks/`, register it in `modeldebug/core.py`, and add a test.

## License

MIT
