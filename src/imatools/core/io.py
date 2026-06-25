# src/imatools/core/io.py
# ---------------------------------------------------------------------------
# check_file_exists has been relocated to imatools.io.paths (T2c3) to keep
# the core/ layer I/O-free.  This shim preserves the legacy import path:
#   from imatools.core.io import check_file_exists
# ---------------------------------------------------------------------------

from imatools.io.paths import check_file_exists  # noqa: F401
