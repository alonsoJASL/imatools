import argparse
import logging
import os
import sys

import numpy as np
import SimpleITK as sitk
from bs4 import BeautifulSoup

from imatools.core.image import SegmentationGenerator
from imatools.io import image_io

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# parse_xml — CemrgApp .mps PointSet reader (single use, kept inline)
# ---------------------------------------------------------------------------


def parse_xml(xml_path):
    """Parse PointSet file from CemrgApp (mps)."""
    with open(xml_path, "r") as f:
        soup = BeautifulSoup(f, "xml")

    points = soup.find_all("point")
    coords = np.zeros((len(points), 3), dtype=np.float32)

    for i, p in enumerate(points):
        coords[i, 0] = float(p.find("x").text)
        coords[i, 1] = float(p.find("y").text)
        coords[i, 2] = float(p.find("z").text)

    return coords


# ---------------------------------------------------------------------------
# coords-to-index handler
# ---------------------------------------------------------------------------


def handle_coords_to_index(args):
    """Convert physical coordinates to image voxel indices."""
    image_path = args.input
    coords_path = args.coords
    output_name = args.output
    verbose = args.verbose

    if verbose:
        logger.info(f"image_path : {image_path}")
        logger.info(f"coords_path : {coords_path}")
        logger.info(f"output_name : {output_name}")

    if ".mps" in coords_path:
        coordinates = parse_xml(coords_path)
    elif ".txt" in coords_path:
        coordinates = np.loadtxt(coords_path, delimiter=",")
    elif "," in coords_path:
        coordinates = np.array(args.coords.split(","), dtype=np.float32)
    else:
        logger.error("Error: coords_path is not .mps or .txt file.")
        return 1

    # Get folder from image_path
    path = os.path.dirname(image_path)
    output_path = os.path.join(path, output_name) if output_name != "" else ""
    write_chr = "w" if os.path.exists(output_path) else "a"

    image = sitk.ReadImage(image_path)

    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())

    if verbose:
        logger.info(f"origin : {origin}")
        logger.info(f"spacing : {spacing}")

    for coord in coordinates:
        index = image.TransformPhysicalPointToIndex(coord.tolist())
        index_str = f"{str(index)[1:-2]}"  # remove brackets

        if output_path == "":
            print(index_str)
        else:
            with open(output_path, write_chr) as f:
                f.write(f"{index_str}\n")
                write_chr = "a"  # append after first line

    return 0


# ---------------------------------------------------------------------------
# gen-circle handler
# ---------------------------------------------------------------------------


def handle_gen_circle(args):
    """Generate a synthetic circle segmentation image."""
    outname = args.output_name if args.output_name != "" else "circle.nii"

    generator = SegmentationGenerator(size=args.size, origin=args.origin, spacing=args.spacing)
    circle = generator.generate_circle(args.radius, args.center)

    out_dir = os.path.dirname(os.path.abspath(outname))
    os.makedirs(out_dir, exist_ok=True)

    image_io.save_image(circle, outname, overwrite=True)
    logger.info("Saved circle image to %s", outname)
    return 0


# ---------------------------------------------------------------------------
# gen-cube handler
# ---------------------------------------------------------------------------


def handle_gen_cube(args):
    """Generate a synthetic cube segmentation image."""
    outname = args.output_name if args.output_name != "" else "cube.nii"

    generator = SegmentationGenerator(size=args.size, origin=args.origin, spacing=args.spacing)
    # generate_cube(size, origin) — 'size' is the Gaussian sigma (side), 'origin' is the center
    cube = generator.generate_cube(args.side, args.center)

    out_dir = os.path.dirname(os.path.abspath(outname))
    os.makedirs(out_dir, exist_ok=True)

    image_io.save_image(cube, outname, overwrite=True)
    logger.info("Saved cube image to %s", outname)
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="imatools-image", description="Image-domain utilities")
    sub = p.add_subparsers(dest="command")

    # --- coords-to-index ---
    pc = sub.add_parser(
        "coords-to-index",
        help="Convert physical coordinates to image voxel indices",
    )
    pc.add_argument("-im", "--input", type=str, required=True, help="Input image path")
    pc.add_argument(
        "-xyz",
        "--coords",
        type=str,
        required=True,
        help="Coordinates to convert to index. File (.mps, .txt) or comma-separated values.",
    )
    pc.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Output filename (default: print to console)",
    )
    pc.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    pc.set_defaults(func=handle_coords_to_index)

    # --- gen-circle ---
    pcirc = sub.add_parser(
        "gen-circle",
        help="Generate a synthetic circle segmentation image",
    )
    pcirc.add_argument(
        "-out",
        "--output-name",
        type=str,
        default="",
        help="Output image name (default: circle.nii)",
    )
    pcirc.add_argument(
        "--size",
        nargs=3,
        type=int,
        default=[300, 300, 100],
        help="Size of the image (default: 300 300 100)",
    )
    pcirc.add_argument(
        "--origin",
        nargs=3,
        type=float,
        default=[0, 0, 0],
        help="Origin of the image (default: 0 0 0)",
    )
    pcirc.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        default=[1, 1, 1],
        help="Spacing of the image (default: 1 1 1)",
    )
    pcirc.add_argument(
        "-r",
        "--radius",
        type=int,
        default=80,
        help="Radius of the circle (default: 80)",
    )
    pcirc.add_argument(
        "-c",
        "--center",
        nargs=3,
        type=int,
        default=[150, 150, 50],
        help="Center of the circle (default: 150 150 50)",
    )
    pcirc.set_defaults(func=handle_gen_circle)

    # --- gen-cube ---
    pcube = sub.add_parser(
        "gen-cube",
        help="Generate a synthetic cube segmentation image",
    )
    pcube.add_argument(
        "-out",
        "--output-name",
        type=str,
        default="",
        help="Output image name (default: cube.nii)",
    )
    pcube.add_argument(
        "--size",
        nargs=3,
        type=int,
        default=[300, 300, 100],
        help="Size of the image (default: 300 300 100)",
    )
    pcube.add_argument(
        "--origin",
        nargs=3,
        type=float,
        default=[0, 0, 0],
        help="Origin of the image (default: 0 0 0)",
    )
    pcube.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        default=[1, 1, 1],
        help="Spacing of the image (default: 1 1 1)",
    )
    pcube.add_argument(
        "-s",
        "--side",
        type=int,
        default=80,
        help="Side length of the cube (default: 80)",
    )
    pcube.add_argument(
        "-c",
        "--center",
        nargs=3,
        type=int,
        default=[150, 150, 50],
        help="Center of the cube (default: 150 150 50)",
    )
    pcube.set_defaults(func=handle_gen_cube)

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
