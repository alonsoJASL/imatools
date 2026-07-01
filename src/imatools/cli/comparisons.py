"""
CLI entry point for collaborator study-pipeline batch drivers.

Subcommands:
  map-fibres   Batch-map closest pts/elems between each comparison's two case
               meshes -> <comparison_dir>/MAPPING/<mname>_<map_type>.csv
  compare      Batch-read MAPPING csvs, load .dat/.lon field files, index by
               the mapping and compute scalar/vector comparisons ->
               <case>/COMPARISONS/<name>[_fibre].csv

Both subcommands read the SAME manifest CSV (columns: comparison_dir,
case_left, case_right) rather than assuming a hardcoded
``011_comparisons/C0..C49`` folder tree.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from imatools.core import mesh as core_mesh
from imatools.core import metrics as core_metrics
from imatools.io import carp_io, mesh_io

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MAP_FIBRES_NAMES = {"scar": "scar", "l": "fibre_l", "1": "fibre_1", "in": "input"}

_COMPARE_FILES_AND_MAPPING = {
    "lat": ("LAT_RSPV_X.dat", "fibre_X_pts.csv"),
    "gradlat": ("lat_X.gradmag.dat", "fibre_X_pts.csv"),
    "ps": ("PSNodeSmooth.dat", "input_pts.csv"),
    "f_endo": ("fibre_X_endo.lon", "fibre_X_endo_elem.csv"),
    "f_epi": ("fibre_X_epi.lon", "fibre_X_epi_elem.csv"),
}


def load_manifest(manifest_path: str) -> pd.DataFrame:
    """Load the comparisons manifest CSV.

    Expected columns: ``comparison_dir``, ``case_left``, ``case_right``
    (one row per comparison). Generalises the old fixed ``011_comparisons/
    C0..C49`` tree / ``comparisons_path.csv`` conventions.
    """
    df = pd.read_csv(manifest_path)
    required_cols = {"comparison_dir", "case_left", "case_right"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest {manifest_path} is missing columns: {sorted(missing)}")
    return df


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def handle_map_fibres(args: argparse.Namespace) -> int:
    which_name = args.name
    map_type = args.map_type
    layer = args.layer
    verbose = args.verbose

    if which_name in ("in", "scar"):
        layer = None

    mname = (
        _MAP_FIBRES_NAMES[which_name]
        if layer is None
        else f"{_MAP_FIBRES_NAMES[which_name]}_{layer}"
    )
    mname_ext = mname + ".vtk"

    logger.info("Calculating %s-mapping", map_type)
    if verbose:
        logger.info("FILE: %s", which_name)
        logger.info("MSH_NAME: %s", mname)
        logger.info("MAPPING_TYPE: %s", map_type)

    manifest = load_manifest(str(args.manifest))

    n_comparisons = len(manifest)
    for count, row in enumerate(manifest.itertuples(index=False), start=1):
        comparison_dir = str(row.comparison_dir)
        id_left = str(row.case_left)
        id_right = str(row.case_right)

        path_left = os.path.join(comparison_dir, id_left, mname_ext)
        path_right = os.path.join(comparison_dir, id_right, mname_ext)

        logger.info("[%d/%d] Mapping %s <-> %s", count, n_comparisons, id_left, id_right)
        msh_left = mesh_io.read_vtk(path_left)
        msh_right = mesh_io.read_vtk(path_right)

        midic = core_mesh.create_mapping(
            msh_left, msh_right, left_id=id_left, right_id=id_right, map_type=map_type
        )
        if midic is None:
            logger.error("Mapping failed for %s — check map type and mesh files.", comparison_dir)
            continue

        df = pd.DataFrame(midic)
        odir = os.path.join(comparison_dir, "MAPPING")
        os.makedirs(odir, exist_ok=True)
        oname = f"{mname}_{map_type}.csv"
        out_path = os.path.join(odir, oname)
        df.to_csv(out_path, index=False)
        logger.info("Mapping CSV written to %s", out_path)

    logger.info("Finished map-fibres")
    return 0


def handle_compare(args: argparse.Namespace) -> int:
    which_name = args.name
    max_distance = args.max_distance
    fibre = args.fibre
    debug = args.debug
    verbose = args.verbose

    if debug:
        logger.info("DEBUG: running only first comparison")

    dat_file, map_name = _COMPARE_FILES_AND_MAPPING[which_name]
    is_vector_field = "f_" in which_name

    if is_vector_field and fibre is None:
        sys.exit("[ERROR] specify which fibre file with -f {1, l}")

    if fibre is not None:
        dat_file = dat_file.replace("X", fibre)
        map_name = map_name.replace("X", fibre)

    if verbose:
        logger.info("COMPARISON: %s", dat_file)
        logger.info("MAPPING: %s", map_name)

    max_distance_set = max_distance is not None
    if max_distance_set and verbose:
        logger.info("DISTANCE: Maximum distance = %s mm", max_distance)

    manifest = load_manifest(str(args.manifest))

    rows_to_skip = 1 if is_vector_field else 0
    n_comparisons = 1 if debug else len(manifest)

    for count, row in enumerate(manifest.itertuples(index=False), start=1):
        if count > n_comparisons:
            break

        comparison_dir = str(row.comparison_dir)

        mapping_files_dir = os.path.join(comparison_dir, "MAPPING")
        map_path = os.path.join(mapping_files_dir, map_name)

        logger.info("[%d/%d] Comparing in %s", count, n_comparisons, comparison_dir)
        df = pd.read_csv(map_path)

        # Case ids come from the MAPPING csv columns, which create_mapping orders
        # by mesh size (small_id first, large_id second). idx0/idx1 index those
        # two meshes respectively, so the .dat field arrays MUST be loaded from the
        # SAME ids — NOT the manifest case_left/case_right, whose order is
        # independent of size (loading from the wrong one misaligns arr with idx).
        map_case0 = df.columns[0]
        map_case1 = df.columns[1]
        idx0 = df[map_case0]
        idx1 = df[map_case1]

        if max_distance_set:
            idx0 = idx0[df.distance_manual <= max_distance]
            idx1 = idx1[df.distance_manual <= max_distance]

        case0_path = os.path.join(comparison_dir, str(map_case0))
        case1_path = os.path.join(comparison_dir, str(map_case1))

        arr0 = np.loadtxt(os.path.join(case0_path, dat_file), skiprows=rows_to_skip)
        arr1 = np.loadtxt(os.path.join(case1_path, dat_file), skiprows=rows_to_skip)

        if which_name == "gradlat":
            arr_idx0 = 1 / arr0[idx0]
            arr_idx1 = 1 / arr1[idx1]
        else:
            arr_idx0 = arr0[idx0]
            arr_idx1 = arr1[idx1]

        if is_vector_field:
            _, _, r = carp_io.loadCarpMesh(dat_file[:-4], case0_path)
            r = r[idx0]
            my_dic = core_metrics.compare_vector_field(arr_idx0, arr_idx1, r)
        else:
            my_dic = core_metrics.compare_scalar_field(arr_idx0, arr_idx1)
            my_dic[which_name + "_0"] = my_dic.pop("s0")
            my_dic[which_name + "_1"] = my_dic.pop("s1")

        my_dic["distance"] = df.distance_manual[idx0]

        odf = pd.DataFrame(my_dic)
        odir = os.path.join(comparison_dir, "COMPARISONS")
        os.makedirs(odir, exist_ok=True)
        oname = which_name
        oname += "" if fibre is None else "_" + fibre
        oname += ".csv"
        out_path = os.path.join(odir, oname)

        odf.to_csv(out_path, index=False)
        if debug:
            print(odf)
        logger.info("Comparison CSV written to %s", out_path)

    logger.info("Finished compare")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="imatools-comparisons", description="Collaborator study-pipeline batch drivers"
    )
    sub = p.add_subparsers(dest="command")

    # --- map-fibres ---
    pmf = sub.add_parser(
        "map-fibres",
        help="Batch-map closest pts/elems between each comparison's two case meshes",
    )
    pmf.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Manifest CSV (columns: comparison_dir, case_left, case_right)",
    )
    pmf.add_argument("-n", "--name", choices=["in", "l", "1", "scar"], required=True, type=str)
    pmf.add_argument("-map", "--map-type", choices=["elem", "pts"], required=True, type=str)
    pmf.add_argument(
        "-layer",
        "--layer",
        choices=["endo", "epi"],
        required=False,
        help="Define layer for fibre files",
    )
    pmf.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    pmf.set_defaults(func=handle_map_fibres)

    # --- compare ---
    pc = sub.add_parser(
        "compare", help="Batch-compare scalar/vector fields using a previously computed mapping"
    )
    pc.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Manifest CSV (columns: comparison_dir, case_left, case_right)",
    )
    pc.add_argument(
        "-n",
        "--name",
        choices=["lat", "gradlat", "ps", "f_endo", "f_epi"],
        required=True,
        type=str,
    )
    pc.add_argument(
        "-mm",
        "--max-distance",
        required=False,
        type=float,
        help="Maximum distance (mm) to consider with comparison",
    )
    pc.add_argument("-f", "--fibre", choices=["1", "l"], required=True, type=str)
    pc.add_argument("-debug", "--debug", action="store_true", help="Debug code")
    pc.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    pc.set_defaults(func=handle_compare)

    return p


def main(args=None):
    parser = _build_parser()
    parsed = parser.parse_args(args)

    if not getattr(parsed, "func", None):
        parser.print_help()
        return 1

    return parsed.func(parsed)


if __name__ == "__main__":
    sys.exit(main())
