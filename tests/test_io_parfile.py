"""Characterization tests for the m3dutils parfile surface (T1l — Concern B).

Target locations (both ABSENT until migration T2e):
  - ``imatools.core.parfile``   → ``update_pot``
  - ``imatools.io.parfile_io``  → ``load_from_par``, ``save_pot``

All tests import from these target locations and are marked
``xfail(strict=False, reason="awaiting migration T2e")`` until T2e creates those
modules and removes the xfail markers.

NOTE: ``imatools.core.parfile`` is a PROVISIONAL module name — chosen to sit
alongside ``imatools.io.parfile_io``.  T2e may rename it if the orchestrator
decides differently; the ``reason`` string on each xfail documents the dependency.

Golden values were captured from master via::

    ~/opt/anaconda3/bin/conda run -n imatools env PYTHONPATH=$M:$M/imatools \\
        python tests/_capture_golden.py --module parfile --out tests/golden

where ``M = ~/dev/python/imatools.worktrees/master``.

Shallow-copy note
-----------------
Master's ``update_pot`` does a **shallow** ``dict.copy()`` of the base pot, so
calling ``update_pot(base, patch)`` mutates ``base``'s sub-dicts in place.
Test inputs are constructed with ``copy.deepcopy`` to make each test independent.

Comparison strategy
-------------------
  - ``update_pot``     → golden is a JSON dict; compared key-by-key with ``==``.
  - ``load_from_par``  → golden is a JSON dict (all values are strings after
                          parsing); compared with ``==``.
  - ``save_pot``       → golden is a JSON list of lines; compared with ``==``.
"""

from __future__ import annotations

import copy
import os
import tempfile

# ---------------------------------------------------------------------------
# Shared fixture data (mirrors _golden_cases/parfile.py definitions)
# ---------------------------------------------------------------------------

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

# Minimal .par file text — same as in _golden_cases/parfile.py.
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
# update_pot — target: imatools.core.parfile (PROVISIONAL)
# ---------------------------------------------------------------------------


def test_update_pot_basic(golden):
    """update_pot merges non-None values from patch into base."""
    from imatools.core.parfile import update_pot

    result = update_pot(copy.deepcopy(_BASE_POT), copy.deepcopy(_UPDATE_PATCH))
    expected = golden("parfile/update_pot_basic")

    assert set(result.keys()) == set(expected.keys())
    for section in expected:
        assert set(result[section].keys()) == set(
            expected[section].keys()
        ), f"section {section!r}: key mismatch"
        for key, exp_val in expected[section].items():
            res_val = result[section][key]
            assert (
                res_val == exp_val
            ), f"section={section!r} key={key!r}: got {res_val!r}, expected {exp_val!r}"


def test_update_pot_identity(golden):
    """update_pot with empty-section patch leaves base unchanged."""
    from imatools.core.parfile import update_pot

    empty_patch = {section: {} for section in _BASE_POT}
    result = update_pot(copy.deepcopy(_BASE_POT), empty_patch)
    expected = golden("parfile/update_pot_identity")

    assert set(result.keys()) == set(expected.keys())
    for section in expected:
        for key, exp_val in expected[section].items():
            res_val = result[section][key]
            assert (
                res_val == exp_val
            ), f"section={section!r} key={key!r}: got {res_val!r}, expected {exp_val!r}"


def test_update_pot_does_not_mutate_input():
    """M3-C3: update_pot deep-copies its base, so the caller's pot is untouched
    (master's shallow copy mutated the input's sub-dicts in place)."""
    from imatools.core.parfile import update_pot

    base = copy.deepcopy(_BASE_POT)
    base_before = copy.deepcopy(base)
    _ = update_pot(base, copy.deepcopy(_UPDATE_PATCH))
    assert base == base_before


# ---------------------------------------------------------------------------
# load_from_par — target: imatools.io.parfile_io
# ---------------------------------------------------------------------------


def test_load_from_par_minimal(golden):
    """load_from_par parses a minimal .par file; values are strings."""
    from imatools.io.parfile_io import load_from_par

    with tempfile.TemporaryDirectory() as tmpdir:
        par_path = os.path.join(tmpdir, "test.par")
        with open(par_path, "w", encoding="utf-8") as fh:
            fh.write(_MINIMAL_PAR_TEXT)
        # load_from_par appends '.par' if absent — pass stem without extension.
        result = load_from_par(os.path.join(tmpdir, "test"))

    expected = golden("parfile/load_from_par_minimal")

    assert set(result.keys()) == set(expected.keys())
    for section in expected:
        assert set(result[section].keys()) == set(
            expected[section].keys()
        ), f"section {section!r}: key mismatch"
        for key, exp_val in expected[section].items():
            res_val = result[section][key]
            assert (
                res_val == exp_val
            ), f"section={section!r} key={key!r}: got {res_val!r}, expected {exp_val!r}"


# ---------------------------------------------------------------------------
# save_pot — target: imatools.io.parfile_io
# ---------------------------------------------------------------------------


def test_save_pot_lines(golden):
    """save_pot writes a .par file; round-trip via readlines matches golden."""
    from imatools.io.parfile_io import save_pot

    # Build the merged dict the same way _golden_cases/parfile.py does.
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

    expected = golden("parfile/save_pot_lines")
    assert lines == expected
