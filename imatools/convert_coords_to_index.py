# Description: Convert coordinates to index in the image.
# Dependencies: SimpleITK, beautifulsoup4, lxml, numpy
# Usage: python convert_coords_to_index.py -im <image_path> -xyz <coords_path> -o <output_name> -v

import os
import argparse 

import SimpleITK as sitk
from bs4 import BeautifulSoup
import numpy as np

def parse_xml(xml_path) :
    """
    Parse PointSet file from CemrgApp (mps)
    """ 
    with open(xml_path, "r") as f : 
        soup = BeautifulSoup(f, "xml")

    points = soup.find_all("point")
    coords = np.zeros((len(points), 3), dtype=np.float32)

    for i, p in enumerate(points) : 
        coords[i, 0] = float(p.find("x").text)
        coords[i, 1] = float(p.find("y").text)
        coords[i, 2] = float(p.find("z").text)

    return coords

def main(args) : 
    """ 
    Convert coordinates to index in the image.
    """

    image_path = args.input
    coords_path = args.coords
    output_name = args.output
    verbose = args.verbose

    if verbose :
        print(f"image_path : {image_path}")
        print(f"coords_path : {coords_path}")
        print(f"output_name : {output_name}")

    if ".mps" in coords_path :
        coordinates = parse_xml(coords_path) 
    elif ".txt" in coords_path :
        coordinates = np.loadtxt(coords_path, delimiter=",")
    elif ".pts" in coords_path: 
        from imatools.common.vtktools import read_pts
        coordinates = read_pts(coords_path)
    elif "," in coords_path :
        coordinates = np.array(args.coords.split(","), dtype=np.float32)
    else :
        # error 
        print("Error: coords_path is not .mps or .txt file.")
        return

    # get folder from image_path
    path = os.path.dirname(image_path)
    output_path = os.path.join(path, output_name) if output_name != "" else ""
    write_chr = 'w' if os.path.exists(output_path) else 'a'

    image = sitk.ReadImage(image_path)

    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    im_size = np.array(image.GetSize())

    if verbose : 
        print("origin : {origin}")
        print("spacing : {spacing}")

    for coord in coordinates :
        index = image.TransformPhysicalPointToIndex(coord.tolist())
        index_str = f"{str(index)[1:-2]}" # remove brackets 

        if output_path == "" : 
            # print comma separated index
            print(index_str) 
        else : 
            with open(output_path, write_chr) as f : 
                f.write(f"{index_str}\n") 
                write_chr = 'a' # append after first line

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description="Convert coordinates to index in the image.")
    parser.add_argument("-im", "--input", type=str, help="Input image path")
    parser.add_argument("-xyz", "--coords", type=str, help="Coordinates to convert to index. File (txt, mps, pts) or separated by comma.") 
    parser.add_argument("-o", "--output", type=str, help="Output filename (default = print to console)", default="")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    args = parser.parse_args()

    main(args)
