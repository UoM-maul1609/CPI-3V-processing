"""Shared test setup: tidy up Qt windows at the end of the session.

Windows left alive by GUI tests are otherwise destroyed by Python's shutdown
after Qt's own teardown, which can crash the interpreter on exit (no test
fails, but the run exits with SIGSEGV/SIGABRT).
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session", autouse=True)
def _close_qt_windows_at_end():
    yield
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    app = QApplication.instance()
    if app is None:
        return
    app.closeAllWindows()
    for widget in app.topLevelWidgets():
        widget.deleteLater()
    app.processEvents()
    import gc
    gc.collect()
    app.processEvents()
