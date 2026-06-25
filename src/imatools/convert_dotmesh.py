import argparse
import os

import imatools.common.vtktools as vtku
from imatools.parsers.dotmesh import save_array  # noqa: F401


def main(args):
    folder = os.path.dirname(args.input)
    name = os.path.basename(args.input)
    if args.output == "":
        args.output = name.replace(".mesh", "")

    gen_attr, pts_attr, elem_attr = vtku.parse_dotmesh_file(args.input, "iso-8859-1")
    pts = pts_attr["points"]
    elem = elem_attr["elements"]

    save_array(pts, f"{os.path.join(folder, args.output)}.pts")
    save_array(elem, f"{os.path.join(folder, args.output)}.elem", True)

    print(f"VTK file saved as {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dotmesh to carp")
    parser.add_argument("-in", "--input", help="Input dotmesh file")
    parser.add_argument("-out", "--output", required=False, default="", help="Output name")

    args = parser.parse_args()
    main(args)
