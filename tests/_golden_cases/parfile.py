"""Capture cases for the m3dutils parfile surface (T1l — Concern B).

Functions (sourced from master ``imatools/common/m3dutils.py``):

Pure function (target ``imatools.core.parfile``):
  - ``update_pot(pot, new_pot)`` — dict merge, 8 lines, no I/O.

I/O functions (target ``imatools.io.parfile_io``):
  - ``load_from_par(filename)`` — reads a Meshtools3d ``.par`` file.
  - ``save_pot(pot, filename)`` — writes a pot dict as a ``.par`` file.

NOTE: ``get_empty_pot`` is NOT in scope (it is a private helper / factory;
see PLAN §3). ``save_to_json`` / ``load_from_json`` are deferred CLI helpers —
out of scope. We characterize only the three in-scope functions above.

The ``.par`` format:
  # comment lines (skipped)
  [section]          — sets current_section
  key = value        — stored as pot[section][key] = value (as strings)
  (blank lines skipped)

We write a minimal valid ``.par`` inline (no fixture file needed) and capture the
parsed dict. For ``save_pot`` we temp-write then read back the file text (list of
lines) as the golden value.
"""

from __future__ import annotations

import os
import tempfile

from _capture_golden import CaptureCase

from imatools.common.m3dutils import load_from_par, save_pot, update_pot

# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

# A minimal pot to use as base for update_pot / save_pot tests.
# Keys must be present in the default pot returned by get_empty_pot().
_BASE_POT = {
    "segmentation": {
        "seg_dir": "/data/segs",
        "seg_name": "heart",
        "mesh_from_segmentation": True,
        "boundary_relabeling": False,
    },
    "meshing": {
        "facet_angle": 30,
        "facet_size": 0.8,
        "facet_distance": 0.1,
        "cell_rad_edge_ratio": 2,
        "cell_size": 0.8,
        "rescaleFactor": 1000,
    },
    "laplacesolver": {
        "abs_toll": 1e-6,
        "rel_toll": 1e-6,
        "itr_max": 700,
        "dimKrilovSp": 500,
        "verbose": True,
    },
    "others": {
        "eval_thickness": False,
    },
    "output": {
        "outdir": "/data/output",
        "name": "result",
        "out_medit": False,
        "out_vtk": True,
        "out_carp": True,
        "out_vtk_binary": False,
        "out_carp_binary": False,
        "out_potential": False,
    },
}

# A partial "new_pot" with only the keys to overwrite (non-None values are merged in).
_UPDATE_PATCH = {
    "meshing": {
        "facet_angle": 25,
        "facet_size": 0.5,
        "facet_distance": None,  # None -> kept from base
        "cell_rad_edge_ratio": 2,
        "cell_size": 0.6,
        "rescaleFactor": 1000,
    },
    "output": {
        "outdir": "/data/output_v2",
        "name": None,  # None -> kept from base
        "out_medit": False,
        "out_vtk": True,
        "out_carp": False,
        "out_vtk_binary": True,
        "out_carp_binary": False,
        "out_potential": True,
    },
}

# Minimal .par text for load_from_par.
_MINIMAL_PAR_TEXT = """\
# Meshtools3d parameter file — minimal synthetic fixture
[segmentation]
seg_dir = /data/segs
seg_name = heart
mesh_from_segmentation = True
boundary_relabeling = False

[meshing]
facet_angle = 30
facet_size = 0.8
facet_distance = 0.1
cell_rad_edge_ratio = 2
cell_size = 0.8
rescaleFactor = 1000

[laplacesolver]
abs_toll = 1e-06
rel_toll = 1e-06
itr_max = 700
dimKrilovSp = 500
verbose = True

[others]
eval_thickness = False

[output]
outdir = /data/output
name = result
out_medit = False
out_vtk = True
out_carp = True
out_vtk_binary = False
out_carp_binary = False
out_potential = False
"""


# ---------------------------------------------------------------------------
# Case functions
# ---------------------------------------------------------------------------


def _capture_update_pot_basic():
    """update_pot: merge _UPDATE_PATCH into _BASE_POT; returns merged dict.

    NOTE: master's update_pot does a shallow copy of pot, so calling it mutates
    _BASE_POT's sub-dicts in place.  We use deep copies here to make each case
    self-contained and reproducible regardless of execution order.
    """
    import copy

    return update_pot(copy.deepcopy(_BASE_POT), copy.deepcopy(_UPDATE_PATCH))


def _capture_update_pot_identity():
    """update_pot: merge empty sections (no keys) — pot unchanged."""
    import copy

    # Build a new_pot with only empty section dicts (no keys) to check identity.
    empty_patch = {section: {} for section in _BASE_POT}
    return update_pot(copy.deepcopy(_BASE_POT), empty_patch)


def _capture_load_from_par():
    """load_from_par: write minimal .par to temp file, parse, return dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        par_path = os.path.join(tmpdir, "test.par")
        with open(par_path, "w", encoding="utf-8") as fh:
            fh.write(_MINIMAL_PAR_TEXT)
        # load_from_par appends '.par' if not present; pass without extension.
        return load_from_par(os.path.join(tmpdir, "test"))


def _capture_save_pot():
    """save_pot: write an explicitly merged pot to temp file, read back lines.

    We explicitly construct the merged dict (equivalent to update_pot(_BASE_POT,
    _UPDATE_PATCH) but using a deep copy to avoid mutation-order dependencies) so
    this case is self-contained and reproducible regardless of CASES execution order.
    """
    import copy

    merged = copy.deepcopy(_BASE_POT)
    for section in _UPDATE_PATCH:
        for key, value in _UPDATE_PATCH[section].items():
            if value is not None:
                merged[section][key] = value

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "out.par")
        save_pot(merged, out_path)
        with open(out_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    return lines


CASES = [
    # ------------------------------------------------------------------
    # update_pot — pure function
    # ------------------------------------------------------------------
    CaptureCase(
        name="parfile/update_pot_basic",
        func=_capture_update_pot_basic,
        args=(),
        fmt="json",
    ),
    CaptureCase(
        name="parfile/update_pot_identity",
        func=_capture_update_pot_identity,
        args=(),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # load_from_par — I/O
    # ------------------------------------------------------------------
    CaptureCase(
        name="parfile/load_from_par_minimal",
        func=_capture_load_from_par,
        args=(),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # save_pot — I/O (round-trip: write then read back)
    # ------------------------------------------------------------------
    CaptureCase(
        name="parfile/save_pot_lines",
        func=_capture_save_pot,
        args=(),
        fmt="json",
    ),
]
