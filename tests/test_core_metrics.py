"""Characterization tests for ``imatools.core.metrics`` (T1f).

All tests import from the TARGET location ``imatools.core.metrics``.  That module
does not exist yet — it will be created by migration task T2c1.  Until then every
test is marked ``xfail(strict=False)`` so it is collected but does not block CI.

Golden values were captured from master via::

    ~/opt/anaconda3/bin/conda run -n imatools env \\
        PYTHONPATH=$M:$M/imatools \\
        python tests/_capture_golden.py --module metrics --out tests/golden

where ``M = ~/dev/python/imatools.worktrees/master``.

Comparison helpers
------------------
* **npy** goldens  → numpy arrays; compared with ``np.testing.assert_allclose``.
* **json** goldens → Python dicts/scalars; when values are ``list``, they are
  converted to arrays before numeric comparison so floating-point serialization
  round-trips do not cause false failures.
"""

from __future__ import annotations

import _fixtures as fx
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_dict_allclose(result: dict, golden: dict, rtol: float = 1e-7) -> None:
    """Assert that every key in *golden* matches the corresponding value in *result*.

    Lists in *golden* (JSON round-tripped arrays) are converted to numpy arrays
    before comparison.  Scalar floats are compared with ``pytest.approx``.
    """
    assert set(golden.keys()) == set(
        result.keys()
    ), f"key mismatch: golden={set(golden.keys())} result={set(result.keys())}"
    for key in golden:
        g_val = golden[key]
        r_val = result[key]
        if isinstance(g_val, list):
            np.testing.assert_allclose(
                np.asarray(r_val), np.asarray(g_val), rtol=rtol, err_msg=f"key={key!r}"
            )
        elif isinstance(g_val, float):
            assert r_val == pytest.approx(g_val, rel=rtol), f"key={key!r}"
        else:
            assert r_val == g_val, f"key={key!r}"


# ---------------------------------------------------------------------------
# Unit-normalised vector field (same derivation as in _golden_cases/metrics.py)
# ---------------------------------------------------------------------------

_vf_raw = fx.vector_field()
_norms = np.linalg.norm(_vf_raw, axis=1, keepdims=True)
_norms = np.where(_norms == 0, 1.0, _norms)
_UNIT_VECTOR_FIELD = _vf_raw / _norms
_N = len(_UNIT_VECTOR_FIELD)
_REGION = np.arange(_N, dtype=int) % 3


# ---------------------------------------------------------------------------
# performanceMetrics
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2c1", strict=False)
def test_performance_metrics_balanced(golden):
    from imatools.core.metrics import performanceMetrics

    result = performanceMetrics(50, 40, 5, 5)
    expected = golden("metrics/performance_balanced")
    _assert_dict_allclose(result, expected)


@pytest.mark.xfail(reason="awaiting migration T2c1", strict=False)
def test_performance_metrics_degenerate_zeros(golden):
    from imatools.core.metrics import performanceMetrics

    result = performanceMetrics(0, 0, 0, 0)
    expected = golden("metrics/performance_degenerate_zeros")
    # All-zero inputs produce NaN for every metric.
    assert set(result.keys()) == set(expected.keys())
    for key in result:
        assert result[key] != result[key], f"expected NaN for key {key!r}"


# ---------------------------------------------------------------------------
# get_boxplot_values
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2c1", strict=False)
def test_get_boxplot_values_linspace(golden):
    from imatools.core.metrics import get_boxplot_values

    result = get_boxplot_values(fx.scalar_field())
    expected = golden("metrics/boxplot_linspace")
    _assert_dict_allclose(result, expected)


# ---------------------------------------------------------------------------
# l2_norm
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2c1", strict=False)
def test_l2_norm_vector_field(golden):
    from imatools.core.metrics import l2_norm

    result = l2_norm(fx.vector_field())
    expected = golden("metrics/l2_norm_vector_field")
    np.testing.assert_allclose(result, expected, rtol=1e-7)


# ---------------------------------------------------------------------------
# classify_array
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2c1", strict=False)
def test_classify_array_bins(golden):
    from imatools.core.metrics import classify_array

    result = classify_array(fx.scalar_field(), fx.classification_thresholds())
    expected = golden("metrics/classify_array_bins")
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# count_values_in_ranges
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2c1", strict=False)
def test_count_values_in_ranges(golden):
    from imatools.core.metrics import count_values_in_ranges

    result = count_values_in_ranges(fx.scalar_field(), fx.classification_thresholds())
    expected = golden("metrics/count_values_in_ranges")
    # JSON keys are strings; result keys are ints.
    assert {str(k): v for k, v in result.items()} == {str(k): v for k, v in expected.items()}


# ---------------------------------------------------------------------------
# near
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2c1", strict=False)
def test_near_equal(golden):
    from imatools.core.metrics import near

    result = near(1.0, 1.0 + 1e-10)
    expected = golden("metrics/near_equal")
    assert bool(result) == bool(expected)


@pytest.mark.xfail(reason="awaiting migration T2c1", strict=False)
def test_near_far(golden):
    from imatools.core.metrics import near

    result = near(1.0, 1.0 + 1e-6)
    expected = golden("metrics/near_far")
    assert bool(result) == bool(expected)


# ---------------------------------------------------------------------------
# compare_large_arrays
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2c1", strict=False)
def test_compare_large_arrays_basic(golden):
    from imatools.core.metrics import compare_large_arrays

    result = compare_large_arrays(fx.scalar_field(), fx.scalar_field()[::-1])
    expected = golden("metrics/compare_large_arrays_basic")
    _assert_dict_allclose(result, expected)


@pytest.mark.xfail(reason="awaiting migration T2c1", strict=False)
def test_compare_large_arrays_identical(golden):
    from imatools.core.metrics import compare_large_arrays

    result = compare_large_arrays(fx.scalar_field(), fx.scalar_field())
    expected = golden("metrics/compare_large_arrays_identical")
    _assert_dict_allclose(result, expected)


# ---------------------------------------------------------------------------
# compare_scalar_field
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2c1", strict=False)
def test_compare_scalar_field_linspace(golden):
    from imatools.core.metrics import compare_scalar_field

    result = compare_scalar_field(fx.scalar_field(), fx.scalar_field()[::-1])
    expected = golden("metrics/compare_scalar_field_linspace")
    _assert_dict_allclose(result, expected)


@pytest.mark.xfail(reason="awaiting migration T2c1", strict=False)
def test_compare_scalar_field_identical(golden):
    from imatools.core.metrics import compare_scalar_field

    result = compare_scalar_field(fx.scalar_field(), fx.scalar_field())
    expected = golden("metrics/compare_scalar_field_identical")
    _assert_dict_allclose(result, expected)


# ---------------------------------------------------------------------------
# compare_vector_field
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2c1", strict=False)
def test_compare_vector_field_unit(golden):
    from imatools.core.metrics import compare_vector_field

    result = compare_vector_field(_UNIT_VECTOR_FIELD, _UNIT_VECTOR_FIELD[::-1], _REGION)
    expected = golden("metrics/compare_vector_field_unit")
    _assert_dict_allclose(result, expected)


@pytest.mark.xfail(reason="awaiting migration T2c1", strict=False)
def test_compare_vector_field_identical(golden):
    from imatools.core.metrics import compare_vector_field

    result = compare_vector_field(_UNIT_VECTOR_FIELD, _UNIT_VECTOR_FIELD, _REGION)
    expected = golden("metrics/compare_vector_field_identical")
    _assert_dict_allclose(result, expected)
