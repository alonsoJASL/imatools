from common import ioutils as vtku
from common import vtktools as iou
import argparse

"""
Loads a mesh
"""

inputParser = argparse.ArgumentParser(description="Test on ioutils")
inputParser.add_argument("base_dir", metavar="base_dir", type=str, help="Directory with data")
inputParser.add_argument("mshname1", metavar="msh_name1", type=str, help="Mesh name")

args = inputParser.parse_args()

baseDir=args.base_dir
mshname1=args.mshname1

iou.cout("Parsed Arguments")

pd=vtku.readVtk(iou.fullfile(baseDir, mshname1))
print(pd)
