"""Aggregation and rendering of check results."""

from __future__ import annotations

from typing import List

from .issue import Issue, Severity

_SEVERITY_COLOR = {
    Severity.INFO: "#6c757d",
    Severity.LOW: "#0d6efd",
    Severity.MEDIUM: "#fd7e14",
    Severity.HIGH: "#dc3545",
    Severity.CRITICAL: "#7d0a0a",
}


class Report:
    """Holds all issues found during a debugging run and renders them."""

    def __init__(self, issues: List[Issue]):
        # Highest severity first.
        self.issues = sorted(issues, key=lambda i: i.severity, reverse=True)

    def __len__(self) -> int:
        return len(self.issues)

    def __repr__(self) -> str:
        return f"<Report: {len(self.issues)} issue(s) found>"

    def filter(self, min_severity: Severity = Severity.INFO) -> List[Issue]:
        """Return issues at or above a minimum severity."""
        return [i for i in self.issues if i.severity.value >= min_severity.value]

    def summary(self) -> str:
        """Plain-text one-issue-per-line summary, sorted by severity."""
        if not self.issues:
            return "No issues found."
        lines = []
        for issue in self.issues:
            lines.append(f"[{issue.severity}] {issue.check_name}: {issue.title}")
        return "\n".join(lines)

    def to_dict(self) -> List[dict]:
        return [i.to_dict() for i in self.issues]

    def show(self) -> None:
        """Render the report. Uses rich HTML in notebooks, plain text otherwise."""
        try:
            from IPython.display import HTML, display

            display(HTML(self._to_html()))
        except ImportError:
            print(self.summary())

    def _to_html(self) -> str:
        if not self.issues:
            return "<p><b>modeldebug:</b> No issues found. ✅</p>"

        rows = []
        for issue in self.issues:
            color = _SEVERITY_COLOR.get(issue.severity, "#000")
            metric_str = f"{issue.metric:.4g}" if issue.metric is not None else "-"
            fix_str = issue.suggested_fix or "-"
            rows.append(
                f"""
                <tr>
                  <td style="padding:6px;border-bottom:1px solid #eee;">
                    <span style="background:{color};color:white;padding:2px 8px;
                    border-radius:10px;font-size:0.8em;">{issue.severity}</span>
                  </td>
                  <td style="padding:6px;border-bottom:1px solid #eee;"><b>{issue.title}</b><br>
                    <span style="color:#555;font-size:0.9em;">{issue.description}</span><br>
                    <span style="color:#0a7d0a;font-size:0.85em;"><i>Suggested fix: {fix_str}</i></span>
                  </td>
                  <td style="padding:6px;border-bottom:1px solid #eee;">{metric_str}</td>
                </tr>
                """
            )

        return f"""
        <div style="font-family:sans-serif;">
          <h3>modeldebug report — {len(self.issues)} issue(s) found</h3>
          <table style="border-collapse:collapse;width:100%;">
            <tr style="text-align:left;background:#f7f7f7;">
              <th style="padding:6px;">Severity</th>
              <th style="padding:6px;">Issue</th>
              <th style="padding:6px;">Metric</th>
            </tr>
            {''.join(rows)}
          </table>
        </div>
        """
