"""Legacy ``common.ioutils`` surface.

Only ``cout`` remains as a real function (still referenced by the deferred
top-level scripts ``qulati_downsample_pair.py`` / ``test_submodules.py``);
everything else was migrated to ``core.metrics`` / ``io.carp_io`` / ``io.paths``
(re-exported via the shim blocks below) or deleted as dead code (M2b).
"""


def cout(msg, typeMsg="INFO", print2console=True, logger=None):  # noqa: N803
    """
    LOGGING FUNCTION
    """
    if print2console == True:  # noqa: E712
        if logger is not None:

            logger.info(f"_ {msg}")
        else:
            print(f"[{typeMsg}] {msg}")


# ---------------------------------------------------------------------------
# Re-export shim — metric functions now live in imatools.core.metrics (T2c1).
# Legacy callers that import from imatools.common.ioutils continue to work.
# ---------------------------------------------------------------------------
from imatools.core.metrics import (  # noqa: E402,F401,I001
    l2_norm,
    near,
    performanceMetrics,
    get_boxplot_values,
    compare_large_arrays,
    classify_array,
    count_values_in_ranges,
)

# ---------------------------------------------------------------------------
# Re-export shim — CARP I/O functions now live in imatools.io.carp_io (T2c2).
# Legacy callers that import from imatools.common.ioutils continue to work.
# ---------------------------------------------------------------------------
from imatools.io.carp_io import (  # noqa: E402,F401
    read_pts,
    read_elem,
    read_lon,
    readParsePts,
    readParseElem,
    loadCarpMesh,
    saveToCarpTxt,
)

# ---------------------------------------------------------------------------
# Re-export shim — path helper functions now live in imatools.io.paths (T2c3).
# Legacy callers that import from imatools.common.ioutils continue to work.
# ---------------------------------------------------------------------------
from imatools.io.paths import (  # noqa: E402,F401
    ext,
    get_subfolders,
    find_file,
    slot_in_path_hrchy,
    fullfile,
    mkdirplus,
    searchFileByType,
    num2padstr,
)
