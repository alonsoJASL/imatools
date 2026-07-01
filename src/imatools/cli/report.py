"""
CLI entry point for mesh-quality-assessment reporting and general VTK
rendering.

Subcommands:
  report          Build a multi-page PDF report of a simulation-ready mesh
                  (whole mesh, fibres, pericardium, epicardium, endocardia,
                  veins, early activation sites) from a `--sims-folder`.
  render-single   Render a folder of VTKs into a single grid PNG.
  render-multi    Render a folder of VTKs into individual PNGs.
"""

import argparse
import logging
import os
import sys

import numpy as np

from imatools import render
from imatools.contracts.report import MeshReportInputs, RenderParams
from imatools.core import mesh as core_mesh
from imatools.io import carp_io

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_vtk_files(base_dir):
    vtk_files = []
    for root, _dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".vtk"):
                vtk_files.append(os.path.join(root, file))
    return vtk_files


def _load_report_inputs(
    sims_folder,
    mesh,
    lon,
    print_pericardium,
    print_epicardium,
    print_endocardia,
    print_veins,
    print_eas,
) -> MeshReportInputs:
    """Populate a MeshReportInputs from a sims_folder, skipping absent files.

    Per-anatomy/per-chamber granularity: each requested field is checked and
    read independently; a missing file logs a warning and leaves that field
    None rather than crashing.
    """
    inputs = MeshReportInputs(mesh=mesh, lon=lon)

    if print_pericardium:
        path = os.path.join(sims_folder, "pericardium_scale.dat")
        if os.path.exists(path):
            inputs.pericardium_scale = np.genfromtxt(path, dtype=float)
        else:
            logger.warning("Pericardium requested but %s not found — skipping", path)

    if print_epicardium:
        path = os.path.join(sims_folder, "epicardium_for_sim.surf")
        if os.path.exists(path):
            inputs.epicardium_surf = carp_io.read_elem(path, el_type="Tr", tags=False)
        else:
            logger.warning("Epicardium requested but %s not found — skipping", path)

    if print_endocardia:
        for field, fname, label in (
            ("lv_endo_surf", "LV_endo.surf", "LV endocardium"),
            ("rv_endo_surf", "RV_endo.surf", "RV endocardium"),
            ("la_endo_surf", "LA_endo.surf", "LA endocardium"),
            ("ra_endo_surf", "RA_endo.surf", "RA endocardium"),
        ):
            path = os.path.join(sims_folder, fname)
            if os.path.exists(path):
                setattr(inputs, field, carp_io.read_elem(path, el_type="Tr", tags=False))
            else:
                logger.warning("%s requested but %s not found — skipping", label, path)

    if print_veins:
        for field, fname, label in (
            ("rpvs_surf", "RPVs.surf", "Right pulmonary veins"),
            ("svc_surf", "SVC.surf", "Superior vena cava"),
        ):
            path = os.path.join(sims_folder, fname)
            if os.path.exists(path):
                setattr(inputs, field, carp_io.read_elem(path, el_type="Tr", tags=False))
            else:
                logger.warning("%s requested but %s not found — skipping", label, path)

    if print_eas:
        for field, fname, label in (
            ("san_vtx", "SAN.vtx", "Sino-atrial node"),
            ("fascicles_lv_vtx", "fascicles_lv.vtx", "LV fascicles"),
            ("fascicles_rv_vtx", "fascicles_rv.vtx", "RV fascicles"),
        ):
            path = os.path.join(sims_folder, fname)
            if os.path.exists(path):
                setattr(inputs, field, np.genfromtxt(path, skip_header=2, dtype=int))
            else:
                logger.warning("%s requested but %s not found — skipping", label, path)

    return inputs


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def handle_report(args: argparse.Namespace) -> int:
    from matplotlib.backends.backend_pdf import PdfPages

    sims_folder = args.sims_folder
    report_name = args.report_name

    print_whole_mesh = args.print_whole_mesh
    print_fibres = args.print_fibres
    print_pericardium = args.print_pericardium
    print_epicardium = args.print_epicardium
    print_endocardia = args.print_endocardia
    print_veins = args.print_veins
    print_eas = args.print_eas
    print_all = args.print_all

    if print_all:
        print_whole_mesh = True
        print_fibres = True
        print_pericardium = True
        print_epicardium = True
        print_endocardia = True
        print_veins = True
        print_eas = True

    # BUG FIX (M1.8): the original repeated `print_epicardium` twice and
    # never checked `print_endocardia`/`print_veins`/`print_EAS`.
    print_any = any(
        [
            print_whole_mesh,
            print_fibres,
            print_pericardium,
            print_epicardium,
            print_endocardia,
            print_veins,
            print_eas,
        ]
    )

    if not print_any:
        logger.error("You need to choose what to print.")
        return 1

    report_name_full = os.path.abspath(os.path.normpath(report_name))
    # BUG FIX (M1.8): use os.path.dirname instead of manual '/'.join(...split('/'))
    os.makedirs(os.path.dirname(os.path.abspath(report_name_full)), exist_ok=True)

    meshname = os.path.join(sims_folder, "myocardium_AV_FEC_BB_lvrv")

    pts = carp_io.read_pts(meshname + ".pts")
    elem = carp_io.read_elem(meshname + ".elem", el_type="Tt", tags=True)

    mesh = render.pts_elem_to_pyvista(pts=pts, elem=elem, add_tags=True)

    lon = None
    if print_fibres:
        lon_path = meshname + ".lon"
        if os.path.exists(lon_path):
            lon_initial = carp_io.read_lon(lon_path)
            mesh, lon = core_mesh.rotate_mesh(plt_msh=mesh, fibres=lon_initial[:, :3])
        else:
            logger.warning("Fibres requested but %s not found — skipping", lon_path)
            mesh = core_mesh.rotate_mesh(mesh)
    else:
        mesh = core_mesh.rotate_mesh(mesh)

    inputs = _load_report_inputs(
        sims_folder,
        mesh,
        lon,
        print_pericardium,
        print_epicardium,
        print_endocardia,
        print_veins,
        print_eas,
    )

    params = RenderParams(
        fig_w=args.fig_w,
        fig_h=args.fig_h,
        colormap=args.colormap,
        zoom=args.zoom,
        dpi=args.dpi,
        title_fontsize=args.title_fontsize,
        title_position=args.title_position,
    )

    with PdfPages(report_name) as pdf:
        render.render_mesh_views(inputs, params, pdf)
        render.render_fibres_views(inputs, params, pdf)
        render.render_pericardium_views(inputs, params, pdf)
        render.render_epicardium_views(inputs, params, pdf)
        render.render_endocardia_views(inputs, params, pdf)
        render.render_veins_views(inputs, params, pdf)
        render.render_eas_views(inputs, params, pdf)

    logger.info("Report written to %s", report_name)
    return 0


def handle_render_single(args: argparse.Namespace) -> int:
    logger.info("Rendering vtk files to a single png")
    base_dir = args.base_dir
    output = os.path.join(base_dir, args.output)

    grid_size = tuple(args.grid_size)
    window_size = tuple(args.window_size)

    vtk_files = _get_vtk_files(base_dir)

    input_data_type = "polydata" if args.polydata else "ugrid"

    render.render_vtk_to_single_png(
        vtk_files, output, grid_size, window_size, input_type=input_data_type
    )
    logger.info("Finished render-single")
    return 0


def handle_render_multi(args: argparse.Namespace) -> int:
    logger.info("Rendering vtk files to multiple pngs")
    base_dir = args.base_dir
    output = os.path.join(base_dir, args.output)

    window_size = tuple(args.window_size)

    vtk_files = _get_vtk_files(base_dir)

    render.render_vtk_to_png(vtk_files, output, window_size)
    logger.info("Finished render-multi")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="imatools-report", description="Mesh report generation and VTK rendering"
    )
    sub = p.add_subparsers(dest="command")

    # --- report ---
    pr = sub.add_parser("report", help="Create a report with images for mesh quality assessment")
    pr.add_argument(
        "--sims-folder",
        type=str,
        required=True,
        help="Path to the folder containing simulation-ready files.",
    )
    pr.add_argument("--report-name", type=str, default="output.pdf", help="Output file name.")

    print_options_group = pr.add_argument_group("Print options")
    print_options_group.add_argument(
        "--fig-w", type=int, default=2480, help="Page width in pixels."
    )
    print_options_group.add_argument(
        "--fig-h", type=int, default=3508, help="Page height in pixels."
    )
    print_options_group.add_argument(
        "--colormap", type=str, default="RdBu", help="Matplotlib colormap for tags."
    )
    print_options_group.add_argument("--zoom", type=float, default=1, help="Zoom magnitude.")
    print_options_group.add_argument(
        "--dpi", type=float, default=100, help="Dots per inch (resolution)."
    )
    print_options_group.add_argument(
        "--title-fontsize",
        type=float,
        default=44,
        help="Fontsize of the title of each page.",
    )
    print_options_group.add_argument(
        "--title-position",
        type=float,
        default=0.9,
        help="Title position value. 1 is at the top of the page, the lower the "
        "value, the lower the position of the title",
    )

    outputs_group = pr.add_argument_group(
        "Available Outputs", "Choose what to include in the report."
    )
    outputs_group.add_argument(
        "--print-whole-mesh",
        action="store_true",
        help="Include anterior and posterior mesh views (opaque and translucent).",
    )
    outputs_group.add_argument(
        "--print-fibres",
        action="store_true",
        help="Include mesh with fibres, separated by chamber, from different views.",
    )
    outputs_group.add_argument(
        "--print-pericardium",
        action="store_true",
        help="Include mesh with pericardium penalty map from different views.",
    )
    outputs_group.add_argument(
        "--print-epicardium",
        action="store_true",
        help="Include epicardium surface views.",
    )
    outputs_group.add_argument(
        "--print-endocardia",
        action="store_true",
        help="Include endocardia surface views.",
    )
    outputs_group.add_argument(
        "--print-veins",
        action="store_true",
        help="Include right pulmonary veins and superior vena cava surface views.",
    )
    outputs_group.add_argument(
        "--print-eas",
        action="store_true",
        help="Include early activation sites (sino-atrial node and ventricular "
        "fascicles) views.",
    )
    outputs_group.add_argument(
        "--print-all",
        action="store_true",
        help="Include all possible images in the report.",
    )
    pr.set_defaults(func=handle_report)

    # --- render-single ---
    prs = sub.add_parser(
        "render-single", help="Render a folder of VTK files into a single grid PNG"
    )
    prs.add_argument("--base-dir", required=True, help="Input folder of VTK files")
    prs.add_argument("--output", default="output.png", help="Output png file")
    prs.add_argument(
        "--grid-size",
        nargs=2,
        type=int,
        default=[1, 1],
        help="Grid size of the output image",
    )
    prs.add_argument(
        "--window-size",
        nargs=2,
        type=int,
        default=[1000, 1000],
        help="Window size of the output image",
    )
    prs.add_argument("--polydata", action="store_true", help="Use polydata instead of structured")
    prs.set_defaults(func=handle_render_single)

    # --- render-multi ---
    prm = sub.add_parser("render-multi", help="Render a folder of VTK files into individual PNGs")
    prm.add_argument("--base-dir", required=True, help="Input folder of VTK files")
    prm.add_argument("--output", default="output.png", help="Output png file")
    prm.add_argument(
        "--window-size",
        nargs=2,
        type=int,
        default=[1000, 1000],
        help="Window size of the output image",
    )
    prm.add_argument("--polydata", action="store_true", help="Use polydata instead of structured")
    prm.set_defaults(func=handle_render_multi)

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
