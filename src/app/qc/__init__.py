"""Automated quality control.

QC answers one question with numbers rather than opinions: *did this render
preserve the performer and the performance, and change only the clothes?*

Two tolerance regimes are used deliberately:

* **lossless intermediates** (the composited PNG frames) are checked with
  ``max_diff == 0`` outside the editable mask. There is no excuse for a single
  changed byte there.
* the **compressed final video** is checked with configurable tolerances,
  because H.264 + yuv420p chroma subsampling perturbs every pixel slightly.
"""

from app.qc.checks import CheckResult, CheckSeverity, QCReport
from app.qc.contact_sheet import build_contact_sheets
from app.qc.report import render_text_report, run_qc

__all__ = [
    "CheckResult",
    "CheckSeverity",
    "QCReport",
    "build_contact_sheets",
    "render_text_report",
    "run_qc",
]
