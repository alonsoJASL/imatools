"""Path helper utilities.

Migrated from ``imatools.common.ioutils`` (functions: ext, get_subfolders,
find_file, slot_in_path_hrchy, fullfile, mkdirplus, searchFileByType,
num2padstr; that shim module was deleted in M2) and from ``imatools.core.io``
(check_file_exists) as part of T2c3.  ``imatools.core.io`` still re-exports
``check_file_exists`` from here for legacy callers.

Note: ``slot_in_path_hrchy`` is moved verbatim but has no characterization
test (T1j did not cover it); it is included here as a clear path helper.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def ext(fname, extension):
    """
    Returns filename with extension
    """
    xt = f".{extension}" if extension[0] != "." else extension
    fname = fname if fname[-len(xt) :] == xt else fname + xt
    return fname


def get_subfolders(directory: str) -> list:
    """
    Returns list of subfolders in a directory
    """
    return [f.path for f in os.scandir(directory) if f.is_dir()]


def find_file(directory: str, fname: str, extension="") -> str:
    """
    Returns path of file in a directory
    """
    list_of_files = []
    for name in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, name)):
            if fname in name:
                list_of_files.append(os.path.join(directory, name))

    if len(list_of_files) == 0:
        return ""

    if len(list_of_files) == 1:
        return list_of_files[0]

    if len(list_of_files) > 1:
        if extension == "":
            return list_of_files[0]
        else:
            for f in list_of_files:
                if extension in f:
                    return f

    return ""


def slot_in_path_hrchy(filepath: str, fname="", num_levels_above=1) -> str:
    """
    Returns path of file in a directory
    """
    filepath = os.path.normpath(filepath)
    num_levels_above = np.abs(num_levels_above)

    res = "/".join(filepath.split("/")[0:-num_levels_above])
    if fname != "":
        res = os.path.join(res, fname)

    return res


def fullfile(*paths):
    """
    Returns path separated by '/'
    """
    s = "/"
    return s.join(paths)


def mkdirplus(*paths):
    """
    Joins paths with fullfile, then creates path
    returns path
    """
    res = fullfile(*paths)
    os.makedirs(res, exist_ok=True)
    return res


def searchFileByType(directory, prefix="", extension=""):  # noqa: N802
    """
    Search file by filetype
    """
    l = glob.glob(fullfile(directory, prefix + "*." + extension))  # noqa: E741
    return l


def num2padstr(number, padding=3):  # noqa: N802
    padstr = str(number)
    if len(padstr) < padding:
        for ix in range(padding - len(padstr)):
            padstr = "0" + padstr

    return padstr


def check_file_exists(path_to_file: Path) -> None:
    """
    Checks if the file exists at the given path.

    :param path_to_file: Path to the file
    :type path_to_file: Path
    :return: True if the file exists, False otherwise
    :rtype: bool
    """
    if not path_to_file.exists():
        logger.error(f"File not found: {path_to_file}")
        raise FileNotFoundError(f"File not found: {path_to_file}")
    logger.info(f"File found: {path_to_file}")
