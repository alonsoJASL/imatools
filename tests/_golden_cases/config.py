"""Capture cases for ``imatools.common.config`` (T1l — Concern A).

Functions:
  - ``configure_logging(log_name, log_level, log_format)``
  - ``add_file_handler(logger, file_path, log_level)``

Both are side-effecting (they mutate logging state), so we reduce to stable
structural attributes rather than capturing the logger/handler objects directly.

For ``configure_logging`` we capture:
  {name, level, handler_types (sorted list), handler_formatters (sorted list)}

For ``add_file_handler`` we call it, inspect the added FileHandler, and capture:
  {level (int), formatter_fmt, basename}

NOTE: ``add_file_handler`` in master takes ``log_level=logging.ERROR``; dev's copy
LACKS this parameter (signature diverged). The test for it xfails until T2d migrates
the master signature. Both capture cases here run against MASTER so they both
succeed at capture time.
"""

from __future__ import annotations

import logging
import os
import tempfile

from _capture_golden import CaptureCase

from imatools.common.config import add_file_handler, configure_logging

# ---------------------------------------------------------------------------
# Reducer helpers — extract only serializable, stable attributes
# ---------------------------------------------------------------------------


def _reduce_configure_logging(logger: logging.Logger) -> dict:
    """Reduce a Logger to stable, serializable attributes."""
    handler_types = sorted(type(h).__name__ for h in logger.handlers)
    handler_fmts = sorted(h.formatter._fmt for h in logger.handlers if h.formatter is not None)
    return {
        "name": logger.name,
        "level": logger.level,
        "handler_types": handler_types,
        "handler_fmts": handler_fmts,
    }


def _make_add_file_handler_case(log_level: int, case_name: str) -> CaptureCase:
    """Build a CaptureCase for add_file_handler by calling it inside a reducer."""
    import tempfile as _tempfile

    def _run_and_reduce(_log_level=log_level, _case_name=case_name):
        # Use a fresh logger name to avoid interference between capture runs.
        logger = logging.getLogger(f"_capture_afh_{_case_name}")
        logger.handlers.clear()
        with _tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            add_file_handler(logger, log_path, _log_level)
            # Find the FileHandler that was added.
            fh = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
            return {
                "level": fh.level,
                "formatter_fmt": fh.formatter._fmt if fh.formatter else None,
                "basename": os.path.basename(fh.baseFilename),
            }

    # Wrap _run_and_reduce as a zero-arg function for CaptureCase.
    return CaptureCase(
        name=f"config/{case_name}",
        func=_run_and_reduce,
        args=(),
        fmt="json",
    )


# ---------------------------------------------------------------------------
# configure_logging cases
# ---------------------------------------------------------------------------


# Case 1: default log_level (INFO) and default log_format.
def _capture_configure_logging_defaults():
    logger = logging.getLogger("_capture_config_defaults")
    logger.handlers.clear()
    result = configure_logging("_capture_config_defaults")
    reduced = _reduce_configure_logging(result)
    # Clean up so the logger does not accumulate handlers across invocations.
    logger.handlers.clear()
    return reduced


# Case 2: custom log_level (DEBUG) and custom format.
def _capture_configure_logging_custom():
    logger = logging.getLogger("_capture_config_custom")
    logger.handlers.clear()
    result = configure_logging(
        "_capture_config_custom",
        log_level=logging.DEBUG,
        log_format="%(levelname)s %(message)s",
    )
    reduced = _reduce_configure_logging(result)
    logger.handlers.clear()
    return reduced


# ---------------------------------------------------------------------------
# add_file_handler cases
# ---------------------------------------------------------------------------


# Case 3: default log_level (ERROR).
def _capture_add_file_handler_default():
    logger = logging.getLogger("_capture_afh_default")
    logger.handlers.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "test.log")
        add_file_handler(logger, log_path)
        fh = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
        result = {
            "level": fh.level,
            "formatter_fmt": fh.formatter._fmt if fh.formatter else None,
            "basename": os.path.basename(fh.baseFilename),
        }
    logger.handlers.clear()
    return result


# Case 4: explicit log_level=WARNING.
def _capture_add_file_handler_warning():
    logger = logging.getLogger("_capture_afh_warning")
    logger.handlers.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "test.log")
        add_file_handler(logger, log_path, logging.WARNING)
        fh = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
        result = {
            "level": fh.level,
            "formatter_fmt": fh.formatter._fmt if fh.formatter else None,
            "basename": os.path.basename(fh.baseFilename),
        }
    logger.handlers.clear()
    return result


CASES = [
    CaptureCase(
        name="config/configure_logging_defaults",
        func=_capture_configure_logging_defaults,
        args=(),
        fmt="json",
    ),
    CaptureCase(
        name="config/configure_logging_custom",
        func=_capture_configure_logging_custom,
        args=(),
        fmt="json",
    ),
    CaptureCase(
        name="config/add_file_handler_default",
        func=_capture_add_file_handler_default,
        args=(),
        fmt="json",
    ),
    CaptureCase(
        name="config/add_file_handler_warning",
        func=_capture_add_file_handler_warning,
        args=(),
        fmt="json",
    ),
]
