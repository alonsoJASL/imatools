"""CLI entry point for volume calculation tools."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from imatools.core import label as core_label
from imatools.core import mesh as core_mesh
from imatools.io import image_io, mesh_io

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _volume_message(label_idx, vols, units):
    multiplier = 1e-3 if units == "mL" else 1
    unit_str = "mL" if units == "mL" else "mm³"
    return f"Label {label_idx}: {round(vols[label_idx] * multiplier, 3)} {unit_str}"


def handle_mesh_props(args):
    mesh = mesh_io.read_vtk(str(args.input))
    area = core_mesh.getSurfaceArea(mesh)
    volume = core_mesh.get_mesh_volume(mesh)
    print(f"Area: {area:.4f} mm², Volume: {volume:.4f} mm³")
    return 0


def handle_label_volumes(args):
    im = image_io.load_image(args.input, return_contract=False)

    if args.label is not None:
        im = core_label.extract_single_label(im, args.label, binarise=True)

    vols = core_label.get_labels_volumes(im)

    if args.output == "":
        for i in sorted(vols.keys()):
            print(_volume_message(i, vols, args.units))
    else:
        output_path = os.path.join(os.path.dirname(str(args.input)), args.output)
        if args.output.endswith(".json"):
            with open(output_path, "w") as f:
                json.dump(vols, f)
        else:
            with open(output_path, "w") as f:
                for i in sorted(vols.keys()):
                    f.write(_volume_message(i, vols, args.units))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="imatools-volume", description="Volume calculation tools")
    sub = p.add_subparsers(dest="command")

    # mesh-props subcommand
    pm = sub.add_parser("mesh-props", help="Print surface area and volume of a VTK mesh")
    pm.add_argument("-in", "--input", type=Path, required=True, help="Path to VTK mesh file")
    pm.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    pm.set_defaults(func=handle_mesh_props)

    # label-volumes subcommand
    pl = sub.add_parser(
        "label-volumes", help="Print or save volumes of labels in a segmentation image"
    )
    pl.add_argument("-in", "--input", type=Path, required=True, help="Path to segmentation image")
    pl.add_argument(
        "-l", "--label", type=int, default=None, help="Single label to calculate volume for"
    )
    pl.add_argument(
        "-out", "--output", type=str, default="", help="Output file name (.json or .txt)"
    )
    pl.add_argument(
        "--units", choices=["mm", "mL"], default="mL", help="Units for output volumes (default: mL)"
    )
    pl.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    pl.set_defaults(func=handle_label_volumes)

    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
