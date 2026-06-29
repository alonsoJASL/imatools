"""Characterization tests for ``imatools.io.scar_io`` (M1.6a).

Functions are imported from their TARGET home ``imatools.io.scar_io``.
The golden values were captured from master's ``ScarQuantificationTools`` methods
using the same synthetic ``prodStats.txt`` content as the capture cases.
"""

from __future__ import annotations

import json

import pytest

from imatools.io.scar_io import (
    create_scar_options_file,
    get_bloodpool_stats_from_file,
    read_stats_from_file,
)

# ---------------------------------------------------------------------------
# Synthetic prodStats.txt content (must be identical to capture cases)
# ---------------------------------------------------------------------------

_MEAN_BP = 152.3
_STD_BP = 31.7

_PROD_STATS_IIR = (
    "IIR_method\n"
    f"{_MEAN_BP}\n"
    f"{_STD_BP}\n"
    "V=0.970, SCORE=5.2\n"
    "V=1.200, SCORE=3.1\n"
    "V=1.320, SCORE=1.4\n"
)

_PROD_STATS_MSD = (
    "MSD_method\n"
    f"{_MEAN_BP}\n"
    f"{_STD_BP}\n"
    "V=0.970, SCORE=5.2\n"
    "V=1.200, SCORE=3.1\n"
    "V=1.320, SCORE=1.4\n"
)


@pytest.fixture
def prod_stats_iir_path(tmp_path):
    p = tmp_path / "prodStats_iir.txt"
    p.write_text(_PROD_STATS_IIR, encoding="utf-8")
    return p


@pytest.fixture
def prod_stats_msd_path(tmp_path):
    p = tmp_path / "prodStats_msd.txt"
    p.write_text(_PROD_STATS_MSD, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# read_stats_from_file
# ---------------------------------------------------------------------------


def test_read_stats_iir(golden, prod_stats_iir_path):
    expected = golden("scar/read_stats_iir")
    method, mean_bp, std_bp, thresholds, scores = read_stats_from_file(str(prod_stats_iir_path))
    assert method == expected["method"]
    assert abs(mean_bp - expected["mean_bp"]) < 1e-9
    assert abs(std_bp - expected["std_bp"]) < 1e-9
    assert thresholds == pytest.approx(expected["thresholds"], abs=1e-9)
    assert scores == pytest.approx(expected["scores"], abs=1e-9)


def test_read_stats_msd(golden, prod_stats_msd_path):
    expected = golden("scar/read_stats_msd")
    method, mean_bp, std_bp, thresholds, scores = read_stats_from_file(str(prod_stats_msd_path))
    assert method == expected["method"]
    assert abs(mean_bp - expected["mean_bp"]) < 1e-9
    assert abs(std_bp - expected["std_bp"]) < 1e-9
    assert thresholds == pytest.approx(expected["thresholds"], abs=1e-9)
    assert scores == pytest.approx(expected["scores"], abs=1e-9)


# ---------------------------------------------------------------------------
# get_bloodpool_stats_from_file
# ---------------------------------------------------------------------------


def test_bloodpool_stats_iir(golden, prod_stats_iir_path):
    expected = golden("scar/bloodpool_stats_iir")
    mean_bp, std_bp = get_bloodpool_stats_from_file(str(prod_stats_iir_path))
    assert abs(mean_bp - expected["mean_bp"]) < 1e-9
    assert abs(std_bp - expected["std_bp"]) < 1e-9


# ---------------------------------------------------------------------------
# create_scar_options_file
# ---------------------------------------------------------------------------


def test_create_scar_options_default(golden, tmp_path):
    expected = golden("scar/create_scar_options_default")
    create_scar_options_file(str(tmp_path))
    with open(tmp_path / "options.json", "r") as f:
        result = json.load(f)
    assert result == expected


def test_create_scar_options_legacy(golden, tmp_path):
    expected = golden("scar/create_scar_options_legacy")
    create_scar_options_file(str(tmp_path), legacy=True)
    with open(tmp_path / "options.json", "r") as f:
        result = json.load(f)
    assert result == expected


def test_create_scar_options_radius(golden, tmp_path):
    expected = golden("scar/create_scar_options_radius")
    create_scar_options_file(str(tmp_path), radius=True)
    with open(tmp_path / "options.json", "r") as f:
        result = json.load(f)
    assert result == expected


def test_create_scar_options_msd(golden, tmp_path):
    expected = golden("scar/create_scar_options_msd")
    create_scar_options_file(str(tmp_path), method=2, threshold_values=[0.5, 1.0])
    with open(tmp_path / "options.json", "r") as f:
        result = json.load(f)
    assert result == expected


def test_create_scar_options_invalid_method(tmp_path):
    with pytest.raises(ValueError):
        create_scar_options_file(str(tmp_path), method=3)


def test_create_scar_options_custom_filename(tmp_path):
    """Verify the opt_file argument controls the output filename."""
    create_scar_options_file(str(tmp_path), opt_file="custom.json")
    assert (tmp_path / "custom.json").exists()
