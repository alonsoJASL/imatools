# ---------------------------------------------------------------------------
# Re-export shim — functions moved to core/parfile and io/parfile_io (T2e;
# get_empty_pot/save_to_json/load_from_json migrated M2a-1).
# Legacy callers that import from this module continue to work.
# ---------------------------------------------------------------------------
from imatools.core.parfile import update_pot  # noqa: E402,F401
from imatools.io.parfile_io import (  # noqa: E402,F401
    get_empty_pot,
    load_from_json,
    load_from_par,
    save_pot,
    save_to_json,
)
