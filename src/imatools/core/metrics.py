# src/imatools/core/metrics.py
"""Metric and comparison functions migrated from ``imatools.common.ioutils``
(T2c1; that shim module was deleted in M2) plus the ``getSurfacesJaccard`` /
``compare_fibres`` / Hausdorff family relocated from ``common.vtktools`` and
``compareCarpMesh`` from ``common.ioutils`` in M2a-2.

``compare_vector_field`` / ``compare_scalar_field`` were copied here (T2c1) from
the old ``compare_from_mapping.py`` batch driver (since deleted in M1.7); this is
their canonical home.

All function bodies use only ``numpy`` / ``scipy.spatial.distance`` / ``vtk`` /
``pyvista`` — no cross-layer accessor needed.

M2a-2 additions: ``getHausdorffDistance``/``getHausdorffDistanceFilter``/
``getSurfacesJaccard``/``compare_fibres`` (from ``common/vtktools.py``) and
``compareCarpMesh`` (from ``common/ioutils.py``) — zero-caller-but-KEEP
functions relocated per Jose's M2 review (see MIGRATION_M2.md). These had no
callers anywhere and so no golden coverage; ``getSurfacesJaccard`` is rebuilt
on pyvista's boolean ops (the old VTK wrapper it depended on is DELETE-
category and being removed elsewhere in M2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyvista as pv
import vtk

from imatools.core.geometry import get_cog_per_element
from imatools.core.mesh import (
    convertCellDataToNpArray,
    getSurfaceArea,
    thresholdExactValue,
    ugrid2polydata,
)

# ---------------------------------------------------------------------------
# Moved from imatools.common.ioutils (verbatim)
# ---------------------------------------------------------------------------


def l2_norm(a):
    return np.linalg.norm(a, axis=1)


def near(value1, value2, tol=1e-8):
    return np.abs(value1 - value2) <= tol


def performanceMetrics(tp, tn, fp, fn):  # noqa: N802
    den_jaccard = tp + fn + fp
    den_precision = tp + fp
    den_recall = tp + fn
    den_accuracy = tp + tn + fp + fn
    den_dice = 2 * tp + fp + fn

    jaccard = tp / (tp + fn + fp) if den_jaccard > 0 else np.nan
    precision = tp / (tp + fp) if den_precision > 0 else np.nan
    recall = tp / (tp + fn) if den_recall > 0 else np.nan
    accuracy = (tp + tn) / (tp + tn + fp + fn) if den_accuracy > 0 else np.nan
    dice = (2 * tp) / (2 * tp + fp + fn) if den_dice > 0 else np.nan

    out_dic = {
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "dice": dice,
    }

    return out_dic


def get_boxplot_values(data, whisker=1.5):
    low_quartile = np.nanpercentile(data, 25, method="nearest")
    high_quartile = np.nanpercentile(data, 75, method="nearest")
    iqr = high_quartile - low_quartile
    low_whis = low_quartile - whisker * iqr
    high_whis = high_quartile + whisker * iqr
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    midic = {
        "min": min_val,
        "low_whisker": np.max([low_whis, min_val]),
        "low_quartile": low_quartile,
        "median": np.nanmedian(data),
        "high_quartile": high_quartile,
        "high_whisker": np.min([high_whis, max_val]),
        "max": max_val,
    }
    return midic


def compare_large_arrays(s0, s1, name0="s0", name1="s1"):
    from scipy.spatial.distance import cosine  # noqa: PLC0415

    l2 = (s0 - s1) ** 2
    abs_diff = np.abs(s0 - s1)
    cosine_similarity = 1 - cosine(s0, s1)
    midic = {
        "diff_square": l2,
        "diff_abs": abs_diff,
        "cosine_similarity": cosine_similarity,
        name0: s0,
        name1: s1,
    }

    return midic


def classify_array(arr, thresholds: list):
    """
    Classifies an array based on given thresholds.

    Parameters:
    arr (numpy.ndarray): Array of values to classify.
    thresholds (list): List of tuples where each tuple contains the name of the classification and the corresponding threshold values.

    Returns:
    numpy.ndarray: Array of classifications.
    """
    classifications = np.zeros_like(arr, dtype=int)
    for i, value in enumerate(arr):
        for j, threshold in enumerate(thresholds):
            if threshold[0] <= value < threshold[1]:
                classifications[i] = j
                break
    return classifications


def count_values_in_ranges(arr, thresholds: list) -> dict:
    """
    Counts the number of values in each range.

    Parameters:
    arr (numpy.ndarray): Array of values to classify.
    thresholds (list): List of tuples where each tuple contains the name of the classification and the corresponding threshold values.

    Returns:
    dict: Dictionary containing the number of values in each range.
    """
    classifications = classify_array(arr, thresholds)
    counts = {}
    for ix in range(len(thresholds)):
        counts[ix] = np.count_nonzero(classifications == ix)

    return counts


def compareCarpMesh(pts1, el1, pts2, el2):  # noqa: N802
    """
    Compare Carp Mesh
    Returns: mean(l2_norm(pts)), median(l2_norm(elem)), comparison_code
            |'COMPARISON_POSSIBLE' : 0
    CODES = |'DIFF_NPTS'           : 1,
            |'DIFF_NELEMS'         : 2,
    """
    comp_codes = {"DIFF_NPTS": 1, "DIFF_NELEMS": 2, "COMPARISON_POSSIBLE": 0}

    if len(pts1) != len(pts2):
        return -1, -1, comp_codes["DIFF_NPTS"]

    if len(el1) != len(el2):
        return -1, -1, comp_codes["DIFF_NELEMS"]

    l2_norm_pts = l2_norm(pts1 - pts2)
    l2_norm_el = l2_norm(np.array(el1) - np.array(el2))

    return np.mean(l2_norm_pts), np.mean(l2_norm_el), comp_codes["COMPARISON_POSSIBLE"]


# ---------------------------------------------------------------------------
# Moved from imatools.common.vtktools (verbatim unless noted; M2a-2)
# ---------------------------------------------------------------------------


def getHausdorffDistance(input_mesh0, input_mesh1, label=0):
    """
    Get Hausdorf Distance between 2 surface meshes
    """
    hd = getHausdorffDistanceFilter(input_mesh0, input_mesh1, label)

    return hd.GetOutput()


def getHausdorffDistanceFilter(input_mesh0, input_mesh1, label=0):
    """
    Get vtkHausdorffDistancePointSetFilter output between 2 surface meshes
    """
    mesh0 = vtk.vtkPolyData()
    mesh1 = vtk.vtkPolyData()
    if label == 0:

        mesh0.DeepCopy(input_mesh0)
        mesh1.DeepCopy(input_mesh1)
    else:

        mesh0 = ugrid2polydata(thresholdExactValue(input_mesh0, label))
        mesh1 = ugrid2polydata(thresholdExactValue(input_mesh1, label))

    hd = vtk.vtkHausdorffDistancePointSetFilter()
    hd.SetInputData(0, mesh0)
    hd.SetInputData(1, mesh1)
    hd.SetTargetDistanceMethodToPointToCell()
    hd.Update()

    return hd


def getSurfacesJaccard(pd1, pd2):
    """Jaccard = surface_area(intersection) / surface_area(union), via pyvista's
    boolean ops (rebuilt M2a-2; the old getBooleanOperation* VTK wrapper is
    superseded by pyvista.PolyData.boolean_union/_intersection)."""
    pv1 = pv.wrap(pd1)
    pv2 = pv.wrap(pd2)
    union = pv1.boolean_union(pv2)
    intersection = pv1.boolean_intersection(pv2)
    return getSurfaceArea(intersection) / getSurfaceArea(union)


def compare_fibres(msh_a, msh_b, f_a, f_b):
    tot_left = msh_a.GetNumberOfPoints()
    tot_right = msh_b.GetNumberOfPoints()

    msh0 = vtk.vtkPolyData()
    msh1 = vtk.vtkPolyData()
    if tot_left >= tot_right:
        msh0.DeepCopy(msh_a)
        msh1.DeepCopy(msh_b)
        f0 = f_a
        f1 = f_b
    else:
        msh0.DeepCopy(msh_b)
        msh1.DeepCopy(msh_a)
        f0 = f_b
        f1 = f_a

    # pts1, el1 = extractPointsAndElemsFromVtk(msh1)
    cog1 = get_cog_per_element(msh1)

    cell_loc = vtk.vtkCellLocator()
    cell_loc.SetDataSet(msh0)
    cell_loc.BuildLocator()

    num_elements1 = len(cog1)
    f0v1_dot = np.zeros(num_elements1)
    f0v1_dist = np.zeros(num_elements1)
    centres = np.zeros(num_elements1)

    norm_vec_f0 = np.divide(f0.T, np.linalg.norm(f0, axis=1)).T
    norm_vec_f1 = np.divide(f1.T, np.linalg.norm(f1, axis=1)).T

    region1 = convertCellDataToNpArray(msh1, "elemTag")

    for jx in range(num_elements1):
        cellId = vtk.reference(0)
        c = [0.0, 0.0, 0.0]
        subId = vtk.reference(0)
        d = vtk.reference(0.0)

        cell_loc.FindClosestPoint(cog1[jx], c, cellId, subId, d)
        centres[jx] = c
        a = norm_vec_f0[cellId.get()]
        b = norm_vec_f1[jx]
        f0v1_dot[jx] = np.dot(a, b)

    f0v1_dist = np.linalg.norm(cog1 - c, axis=1)
    f0v1_angles = np.arccos(f0v1_dot)
    f0v1_abs_dot = np.abs(f0v1_dot)
    f0v1_angle_abs_dot = np.arccos(f0v1_abs_dot)

    d = {
        "region": region1,
        "dot_product": f0v1_dot,
        "angle": f0v1_angles,
        "distance_to_point": f0v1_dist,
        "abs_dot_product": f0v1_abs_dot,
        "angle_from_absdot": f0v1_angle_abs_dot,
    }

    return pd.DataFrame(data=d)


# ---------------------------------------------------------------------------
# Copied from imatools.compare_from_mapping (verbatim — NOT a re-export shim;
# compare_from_mapping.py is a deferred CLI script and must NOT be imported).
# ---------------------------------------------------------------------------


def compare_vector_field(v0, v1, r):
    dotp = np.sum(np.multiply(v0, v1), axis=1)
    abs_dotp = np.abs(dotp)
    midic = {
        "region": r,
        "dot_product": dotp,
        "angle": np.arccos(dotp),
        "abs_dot_product": abs_dotp,
        "angle_from_absdot": np.arccos(abs_dotp),
    }

    return midic


def compare_scalar_field(s0, s1):
    l2 = (s0 - s1) ** 2
    abs_diff = np.abs(s0 - s1)
    midic = {
        "diff_square": l2,
        "diff_abs": abs_diff,
        "s0": s0,
        "s1": s1,
    }

    return midic
