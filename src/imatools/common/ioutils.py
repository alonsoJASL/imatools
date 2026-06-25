import os
import platform as pltf
import sys
import logging

import numpy as np


def dot_prod_vec(a, b):
    return np.sum(a * b, axis=1)


def cout(msg, typeMsg="INFO", print2console=True, logger=None):
    """
    LOGGING FUNCTION
    """
    if print2console == True:
        if logger is not None:

            logger.info(f"_ {msg}")
        else:
            print(f"[{typeMsg}] {msg}")


def getTotal(fname):
    """
    Get total number of elements at the top of a file
    """
    try:
        with open(fname, encoding="utf-8") as f:
            numNodes = int(f.readline().strip())

    except Exception as e:
        print("[getTotal] Error - file not found")
        sys.exit(-1)

    return numNodes


def getFileContentWithTotal(fname):
    """
    Get first line separated from the rest (as long string)
    """
    try:
        with open(fname, encoding="utf-8") as f:
            numNodes = int(f.readline().strip())
            restOfFile = f.readlines()

    except Exception as e:
        print("[getTotal] Error - file not found")
        sys.exit(-1)

    return numNodes, restOfFile


def check_file(file):
    if not os.path.isfile(file):
        raise Exception(f"With the options selected, you need to have {file}")


def readFileToList(fname, delim=","):
    """
    Read File to list. Input is normally a table, like a csv
    """
    try:
        with open(fname, encoding="utf-8") as f:
            fileContents = f.readlines()
            fileContentsInList = [(line.strip()).split(sep=delim) for line in fileContents]
            return fileContentsInList

    except Exception as e:
        print("[readFileToList] Error - file not found")
        sys.exit(-1)


def chooseplatform():
    return pltf.platform().split("-")[0]


def compareCarpMesh(pts1, el1, pts2, el2):
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


def print_progress_bar(
    iteration, total, prefix="", suffix="", decimals=1, length=100, fill="=", printEnd="\r"
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def save_json(fname, data):
    import json

    with open(fname, "w") as f:
        json.dump(data, f)


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
