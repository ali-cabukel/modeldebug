"""modeldebug: diagnostic checks for finding why your ML model is failing.

Quick start:
    >>> from modeldebug import Debugger
    >>> db = Debugger(model, X_train, y_train, X_test, y_test)
    >>> report = db.run()
    >>> report.show()
"""

from .core import Debugger
from .issue import Issue, Severity
from .report import Report

__version__ = "0.1.0"
__all__ = ["Debugger", "Issue", "Severity", "Report"]
