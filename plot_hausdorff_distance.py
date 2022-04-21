import argparse
import os, sys
import math
import vtk
import numpy
from imatools.ioutils import *
from imatools.vtktools import *

import imatools.plotutils as myplot
#
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

inputParser = argparse.ArgumentParser(description="Load Hausdorff distance mesh, calculate some stats")
inputParser.add_argument("base_dir", metavar="base_dir", type=str, help="Directory with data")
inputParser.add_argument("comparisons_file", metavar="comparisons_file", type=str, help="CSV file with comparisons")
inputParser.add_argument("-l", "--labels", metavar="labels", nargs='?', default="1,11,13,15,17,19", type=str, help="Labels (comma-separated, 0=all)")
inputParser.add_argument("-v", "--verbose", action='store_true', help="Verbose output")

args = inputParser.parse_args()

baseDir=args.base_dir
comparisons_file=args.comparisons_file
labels_str=args.labels
verbose=args.verbose
cout("Parsed arguments")

cout(labels_str, "LABELS", verbose)
labels = [int(n.strip()) for n in labels_str.split(',')]

df = pd.read_csv(fullfile(baseDir, comparisons_file))
LABEL_STR = ['FULL', 'LA', 'LSPV', 'LIPV', 'RSPV', 'RIPV', 'LAA']
LABEL_LIST = [0, 1, 11, 13, 15, 17, 19]

mydic_haus = dict()
mydic_distance = dict()
for l in labels:
    mydic_haus[LABEL_STR[LABEL_LIST.index(l)]] = df.Hausdorff[df.label==l]
    mydic_distance[LABEL_STR[LABEL_LIST.index(l)]] = df.mean_dist[df.label==l]

odir=baseDir
fig1, ax1 = myplot.plot_dict(mydic_haus, 'boxplot', odir, 'Hausdorf')
fig2, ax2 = myplot.plot_dict(mydic_distance, 'boxplot', odir, 'Mean_Distance')
