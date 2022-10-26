import sys
import os
IMATOOLS_DIR = os.getcwd()+'/../imatools'
sys.path.insert(1, IMATOOLS_DIR)

import imatools.ioutils as iou
import imatools.vtktools as vtktools
import argparse

inputParser = argparse.ArgumentParser(description="Compare fibrosis on meshes")
inputParser.add_argument("-d", "--dir", metavar="dir", type=str, help="Directory with data")
inputParser.add_argument("-imsh0", "--msh-input0", metavar="mshname", type=str, help="Source mesh name")
inputParser.add_argument("-imsh1", "--msh-input1", metavar="mshname", type=str, help="Target mesh name")
inputParser.add_argument("-omsh", "--msh-output", metavar="mshname", type=str, default='output', help="Output mesh name")

inputParser.add_argument("-v", "--verbose", action='store_true', help="Verbose output")

args = inputParser.parse_args()

dir =args.dir
msh0 = args.msh_input0
msh1 = args.msh_input1
msh_output = args.msh_output

data_type = args.data_type