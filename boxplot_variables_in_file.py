import argparse
import os, sys
import math
import vtk
import numpy
from ioutils import *
from vtktools import *

import plotutils as myplot
#
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

inputParser = argparse.ArgumentParser(description="Plot file contents")
inputParser.add_argument("base_dir", metavar="base_dir", type=str, help="Directory with data")
inputParser.add_argument("comparisons_file", metavar="comparisons_file", type=str, help="CSV file with comparisons")
inputParser.add_argument("-l", "--labels", metavar="labels", nargs='?', default="1,11,13,15,17,19", type=str, help="Labels (comma-separated, 0=all)")
inputParser.add_argument("-var", "--variables", metavar="variables_to_plot", nargs='?', default="Hausdorff,Mean_Distance,jaccard", type=str, help="variables to plot (comma separated)")
inputParser.add_argument("-v", "--verbose", action='store_true', help="Verbose output")

args = inputParser.parse_args()

baseDir=args.base_dir
comparisons_file=args.comparisons_file
labels_str=args.labels
variables=args.variables
verbose=args.verbose
cout("Parsed arguments")

cout(labels_str, "LABELS", verbose)
if "." not in labels_str:
    labels = [int(n.strip()) for n in labels_str.split(',')]
else:
    labels = [float(n.strip()) for n in labels_str.split(',')]

vars=[n.strip() for n in variables.split(',')]

df = pd.read_csv(fullfile(baseDir, comparisons_file))
LABEL_STR = ['FULL', 'LA', 'LSPV', 'LIPV', 'RSPV', 'RIPV', 'LAA']
LABEL_LIST = [0, 1, 11, 13, 15, 17, 19]

for va in vars:
    try:
        mydic = dict()
        for l in labels:
            mydic[LABEL_STR[LABEL_LIST.index(l)]] = df[va][df.label==l]

            odir=baseDir
            fig, ax = myplot.plot_dict(mydic, 'boxplot', odir, va)
    except Exception as e:
        cout("Variable <{}> not found in file".format(va), 'ERROR')
