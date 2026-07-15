"""
CLI entry point for mesh processing tools.

Subcommands:
  export            Convert a VTK file to another mesh format (ply/stl/obj/vtp)
  flip-xy           Flip X and Y coordinates of a polydata mesh
  from-dotmesh      Convert a .mesh (dotmesh) file to CARP .pts/.elem files
  points-to-image   Paint image voxels near a point-cloud with a label value
  project-scalars   Project cell/point scalar data from one mesh onto another
  map               Map closest pts/elems between two meshes → CSV
  map-stats         Summarise a mapping CSV with boxplot statistics
  fibrosis-overlap  Compute fibrosis overlap + performance metrics between two meshes
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from imatools.core import mesh as core_mesh
from imatools.core import metrics as core_metrics
from imatools.io import image_io, mesh_io
from imatools.parsers import dotmesh

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _write_mesh_output(mesh, output_path: Path) -> None:
    """Save *mesh* (vtkPolyData) to *output_path* via io.mesh_io.write_vtk."""
    directory = str(output_path.parent)
    # write_vtk appends .vtk; strip it if already present so we don't double-up
    outname = output_path.stem
    mesh_io.write_vtk(mesh, directory, outname)
    logger.info("Mesh written to %s", output_path)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def handle_export(args: argparse.Namespace) -> int:
    logger.info("Loading %s", args.input)
    mesh = mesh_io.read_vtk(str(args.input), input_type="polydata")
    output_file = f"{args.output}.{args.format}"
    logger.info("Exporting as %s → %s", args.format, output_file)
    mesh_io.export_as(mesh, output_file, export_as=args.format)
    return 0


def handle_flip_xy(args: argparse.Namespace) -> int:
    logger.info("Loading %s", args.input)
    mesh = mesh_io.read_vtk(str(args.input))
    core_mesh.flip_xy(mesh)
    output = args.output if args.output else Path(str(args.input)).stem + "_flipped.vtk"
    output_path = Path(output)
    _write_mesh_output(mesh, output_path)
    return 0


def handle_from_dotmesh(args: argparse.Namespace) -> int:
    input_path = Path(str(args.input))
    folder = str(input_path.parent)
    output_base = args.output if args.output else input_path.stem
    logger.info("Parsing dotmesh file %s", input_path)
    _gen_attr, pts_attr, elem_attr = dotmesh.parse_dotmesh_file(str(input_path), "iso-8859-1")
    pts = pts_attr["points"]
    elem = elem_attr["elements"]
    pts_file = os.path.join(folder, f"{output_base}.pts")
    elem_file = os.path.join(folder, f"{output_base}.elem")
    dotmesh.save_array(pts, pts_file, is_elem=False)
    dotmesh.save_array(elem, elem_file, is_elem=True)
    logger.info("Written: %s, %s", pts_file, elem_file)
    return 0


def handle_points_to_image(args: argparse.Namespace) -> int:
    points_path = Path(str(args.points))
    if args.output:
        output_path = str(args.output)
    else:
        stem = points_path.stem
        if not stem.endswith(".nii"):
            stem += ".nii"
        output_path = os.path.join(os.path.dirname(str(args.input)), stem)
    logger.info("Reference image: %s", args.input)
    logger.info("Points file:     %s", args.points)
    logger.info("Output:          %s", output_path)
    result = image_io.pointfile_to_image(
        str(args.input),
        str(args.points),
        label=args.label,
        girth=args.girth,
        points_are_indices=args.indices,
    )
    image_io.save_image(result, output_path, overwrite=True)
    return 0


def handle_project_scalars(args: argparse.Namespace) -> int:
    logger.info("Loading source mesh: %s", args.source)
    msh_source = mesh_io.read_vtk(str(args.source))
    logger.info("Loading target mesh: %s", args.target)
    msh_target = mesh_io.read_vtk(str(args.target))
    logger.info("Projecting %s data", args.data_type)
    if args.data_type == "cell":
        out_mesh = core_mesh.project_cell_data(msh_source, msh_target)
    else:
        out_mesh = core_mesh.project_point_data(msh_source, msh_target)
    output_path = Path(str(args.output))
    _write_mesh_output(out_mesh, output_path)
    return 0


def handle_map(args: argparse.Namespace) -> int:
    input1 = str(args.input1)
    input2 = str(args.input2)
    id_left = args.name1 if args.name1 else os.path.splitext(os.path.basename(input1))[0]
    id_right = args.name2 if args.name2 else os.path.splitext(os.path.basename(input2))[0]

    logger.info("Loading meshes: %s, %s", input1, input2)
    msh_left = mesh_io.read_vtk(input1)
    msh_right = mesh_io.read_vtk(input2)

    logger.info("Creating %s mapping: %s → %s", args.map_type, id_left, id_right)
    midic = core_mesh.create_mapping(msh_left, msh_right, id_left, id_right, args.map_type)
    if midic is None:
        logger.error("Mapping failed — check map type and mesh files.")
        return 1

    base_dir = os.path.dirname(os.path.abspath(input1))
    odir = os.path.join(base_dir, "MAPPING")
    os.makedirs(odir, exist_ok=True)
    oname = f"{id_left}_{id_right}_{args.map_type}.csv"
    out_path = os.path.join(odir, oname)
    pd.DataFrame(midic).to_csv(out_path, index=False)
    logger.info("Mapping CSV written to %s", out_path)
    return 0


def handle_map_stats(args: argparse.Namespace) -> int:
    mapping_path = str(args.mapping)
    if os.path.isdir(mapping_path):
        list_of_files = [
            os.path.join(mapping_path, f) for f in os.listdir(mapping_path) if f.endswith(".csv")
        ]
    else:
        list_of_files = [mapping_path]

    for f in list_of_files:
        if not f.endswith(".csv"):
            continue
        df = pd.read_csv(f)
        stats = core_metrics.get_boxplot_values(df["distance_auto"])
        print(stats)

    return 0


def get_threshold_from_file(filename):
    """Read a 5-line threshold file and return (val, threshold).

    Copied verbatim from ``compare_fibrosis_overlap.py``.
    """
    with open(filename, "r") as f:
        lines = f.readlines()
        val = float(lines[0].strip())
        _method = int(lines[1].strip())  # noqa: F841
        _mean_bp = float(lines[2].strip())  # noqa: F841
        _std_bp = float(lines[3].strip())  # noqa: F841
        thres = float(lines[4].strip())
    return (val, thres)


def parse_threshold(threshold):
    """Parse a threshold argument (number or 5-line file path).

    Copied verbatim from ``compare_fibrosis_overlap.py``.
    """
    try:
        threshold = float(threshold)
        val = threshold
    except ValueError:
        if os.path.isfile(threshold):
            val, threshold = get_threshold_from_file(threshold)
        else:
            raise ValueError("Threshold is not a file or a number")
    return (val, threshold)


def handle_fibrosis_overlap(args: argparse.Namespace) -> int:
    dir_path = str(args.dir)
    msh_input0 = str(args.msh_input0)
    msh_input1 = str(args.msh_input1)
    data_type = args.data_type
    verbose = args.verbose

    val0, t0 = parse_threshold(str(args.threshold0))
    val1, t1 = parse_threshold(str(args.threshold1))

    thio = "TH{}_".format(str(val1).replace(".", "d")) if args.threshold_in_output else ""
    msh_output = thio + "_" + data_type + "_" + str(args.msh_output)

    if verbose:
        logger.info("Parsed arguments")

    if verbose:
        logger.info("Loading meshes")
    msh_input0 = msh_input0 + ".vtk" if ".vtk" not in msh_input0 else msh_input0
    msh_input1 = msh_input1 + ".vtk" if ".vtk" not in msh_input1 else msh_input1
    msh0 = mesh_io.read_vtk(os.path.join(dir_path, msh_input0))
    msh1 = mesh_io.read_vtk(os.path.join(dir_path, msh_input1))

    if verbose:
        logger.info("Calculating fibrosis overlap")
    omsh, counts = core_mesh.fibrosis_overlap(msh0, msh1, t0, t1, type=data_type)

    if verbose:
        logger.info("Saving output mesh")
    mesh_io.write_vtk(omsh, dir_path, msh_output)

    if verbose:
        logger.info("Calculating fibrosis scores")
    fib0 = core_mesh.fibrosis_score(msh0, t0, type=data_type)
    fib1 = core_mesh.fibrosis_score(msh1, t1, type=data_type)

    if verbose:
        logger.info("Calculating performance metrics for msh1 (%s)", msh_input1)
    perf = core_metrics.performance_metrics(
        tp=counts["overlap"], tn=counts["none"], fp=counts["msh1"], fn=counts["msh0"]
    )

    basestr = f"{args.id},{t0},{t1},{fib0},{fib1}"
    if args.output_type == "df":
        print(f"{basestr},{perf['jaccard']}, jaccard")
        print(f"{basestr},{perf['precision']}, precision")
        print(f"{basestr},{perf['recall']}, recall")
        print(f"{basestr},{perf['accuracy']}, accuracy")
        print(f"{basestr},{perf['dice']}, dice")
    else:
        print(
            f"{basestr},{counts['overlap']},{counts['none']},{counts['msh1']},{counts['msh0']},"
            f"{perf['jaccard']},{perf['precision']},{perf['recall']},{perf['accuracy']}"
        )

    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="imatools-mesh", description="Mesh processing tools")
    sub = p.add_subparsers(dest="command")

    # --- export ---
    pe = sub.add_parser("export", help="Export a VTK polydata file to another mesh format")
    pe.add_argument("-in", "--input", type=Path, required=True, help="Input VTK file")
    pe.add_argument(
        "-out", "--output", type=str, required=True, help="Output base name (no extension)"
    )
    pe.add_argument(
        "--format", choices=["ply", "stl", "obj", "vtp"], required=True, help="Target format"
    )
    pe.set_defaults(func=handle_export)

    # --- flip-xy ---
    pf = sub.add_parser("flip-xy", help="Flip X/Y coordinates of a VTK polydata mesh")
    pf.add_argument("-in", "--input", type=Path, required=True, help="Input VTK file")
    pf.add_argument(
        "-out",
        "--output",
        type=str,
        default=None,
        help="Output VTK path (default: <stem>_flipped.vtk in same directory)",
    )
    pf.set_defaults(func=handle_flip_xy)

    # --- from-dotmesh ---
    pd = sub.add_parser("from-dotmesh", help="Convert a .mesh (dotmesh) file to CARP .pts/.elem")
    pd.add_argument("-in", "--input", type=Path, required=True, help="Input .mesh file")
    pd.add_argument(
        "-out",
        "--output",
        type=str,
        default=None,
        help="Output base name (no extension, default = stem of input)",
    )
    pd.set_defaults(func=handle_from_dotmesh)

    # --- points-to-image ---
    pp = sub.add_parser(
        "points-to-image", help="Paint image voxels near a point-cloud with a label"
    )
    pp.add_argument("-in", "--input", type=Path, required=True, help="Reference image path")
    pp.add_argument("-pts", "--points", type=Path, required=True, help="Points file path")
    pp.add_argument("-l", "--label", type=int, default=1, help="Label value (default 1)")
    pp.add_argument(
        "-g", "--girth", type=int, default=2, help="Cube half-width around each point (default 2)"
    )
    pp.add_argument(
        "-out",
        "--output",
        type=str,
        default=None,
        help="Output image path (default: stem of points file + .nii, placed beside input image)",
    )
    pp.add_argument(
        "--indices",
        action="store_true",
        help="Treat points as image indices rather than world coordinates",
    )
    pp.set_defaults(func=handle_points_to_image)

    # --- project-scalars ---
    ps = sub.add_parser(
        "project-scalars",
        help="Project cell/point scalar data from a source mesh onto a target mesh",
    )
    ps.add_argument("-src", "--source", type=Path, required=True, help="Source VTK file")
    ps.add_argument("-tgt", "--target", type=Path, required=True, help="Target VTK file")
    ps.add_argument(
        "-out",
        "--output",
        type=str,
        default="output.vtk",
        help="Output VTK path (default: output.vtk)",
    )
    ps.add_argument(
        "-dt",
        "--data-type",
        choices=["cell", "point"],
        required=True,
        help="Type of scalar data to project",
    )
    ps.set_defaults(func=handle_project_scalars)

    # --- map ---
    pm = sub.add_parser("map", help="Map closest pts/elems between two meshes and write a CSV")
    pm.add_argument("-in1", "--input1", type=Path, required=True, help="Input mesh file 1")
    pm.add_argument("-in2", "--input2", type=Path, required=True, help="Input mesh file 2")
    pm.add_argument(
        "-n1",
        "--name1",
        type=str,
        default="",
        help="Identifier for mesh 1 (default: stem of file 1)",
    )
    pm.add_argument(
        "-n2",
        "--name2",
        type=str,
        default="",
        help="Identifier for mesh 2 (default: stem of file 2)",
    )
    pm.add_argument(
        "-map",
        "--map-type",
        choices=["elem", "pts"],
        required=True,
        help="Mapping mode: element-to-element or point-to-point",
    )
    pm.set_defaults(func=handle_map)

    # --- map-stats ---
    pms = sub.add_parser(
        "map-stats", help="Summarise a mapping CSV (or directory of CSVs) with boxplot statistics"
    )
    pms.add_argument(
        "-m",
        "--mapping",
        type=Path,
        required=True,
        help="Path to a mapping CSV file or a directory of CSV files",
    )
    pms.set_defaults(func=handle_map_stats)

    # --- fibrosis-overlap ---
    pfo = sub.add_parser(
        "fibrosis-overlap",
        help="Compute fibrosis overlap and performance metrics between two meshes",
    )
    pfo.add_argument(
        "-d", "--dir", type=Path, required=True, help="Directory containing mesh files"
    )
    pfo.add_argument(
        "-imsh0",
        "--msh-input0",
        type=str,
        required=True,
        help="Source mesh name (no extension needed)",
    )
    pfo.add_argument(
        "-imsh1",
        "--msh-input1",
        type=str,
        required=True,
        help="Target mesh name (no extension needed)",
    )
    pfo.add_argument(
        "-omsh",
        "--msh-output",
        type=str,
        default="overlap",
        help="Output mesh name (default: overlap)",
    )
    pfo.add_argument(
        "-t0", "--threshold0", type=str, required=True, help="Threshold for mesh 0 (number or file)"
    )
    pfo.add_argument(
        "-t1", "--threshold1", type=str, required=True, help="Threshold for mesh 1 (number or file)"
    )
    pfo.add_argument(
        "-dt",
        "--data-type",
        choices=["cell", "point"],
        default="cell",
        help="Data type to use for fibrosis computation (default: cell)",
    )
    pfo.add_argument(
        "-thio",
        "--threshold-in-output",
        action="store_true",
        help="Prefix output mesh name with threshold value",
    )
    pfo.add_argument(
        "-id", "--id", type=str, default="ID", help="Identifier string printed in output rows"
    )
    pfo.add_argument(
        "-type",
        "--output-type",
        choices=["df", "compact"],
        default="df",
        help="Output format: one metric per line (df) or compact single line (compact)",
    )
    pfo.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    pfo.set_defaults(func=handle_fibrosis_overlap)

    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
