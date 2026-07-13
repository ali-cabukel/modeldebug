"""End-to-end example: train a leaky model on purpose, then catch it.

Run with: python examples/demo.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from modeldebug import Debugger

rng = np.random.default_rng(42)
n = 500

# Build a dataset where one feature leaks the target directly.
y = rng.integers(0, 2, size=n)
X = pd.DataFrame(
    {
        "real_feature_1": rng.normal(size=n),
        "real_feature_2": rng.normal(size=n),
        "leaky_feature": y + rng.normal(scale=0.01, size=n),  # oops
    }
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Intentionally duplicate a few test rows back into train, another common bug.
X_train = pd.concat([X_train, X_test.iloc[:5]], ignore_index=True)
y_train = np.concatenate([y_train, y_test[:5]])

model = LogisticRegression().fit(X_train, y_train)

print(f"Test accuracy: {model.score(X_test, y_test):.3f}  (suspiciously high? let's check why)\n")

db = Debugger(model, X_train, y_train, X_test, y_test)
report = db.run(verbose=True)

print("\n" + report.summary())
