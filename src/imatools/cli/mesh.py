"""
CLI entry point for mesh processing tools.

Subcommands:
  export            Convert a VTK file to another mesh format (ply/stl/obj/vtp)
  flip-xy           Flip X and Y coordinates of a polydata mesh
  from-dotmesh      Convert a .mesh (dotmesh) file to CARP .pts/.elem files
  points-to-image   Paint image voxels near a point-cloud with a label value
  project-scalars   Project cell/point scalar data from one mesh onto another
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from imatools.core import mesh as core_mesh
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
