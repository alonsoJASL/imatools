from common import ioutils as iou
from common import vtktools as vtku
import argparse

inputParser = argparse.ArgumentParser(description="Compare fibrosis on meshes")
inputParser.add_argument("-d", "--dir", metavar="dir", type=str, help="Directory with data")
inputParser.add_argument("-imsh0", "--msh-input0", metavar="mshname", type=str, help="Source mesh name")
inputParser.add_argument("-imsh1", "--msh-input1", metavar="mshname", type=str, help="Target mesh name")
inputParser.add_argument("-omsh", "--msh-output", metavar="mshname", type=str, default='overlap', help="Output mesh name")
inputParser.add_argument("-t0", "--threshold0", type=float, help="Mesh 0 threshold")
inputParser.add_argument("-t1", "--threshold1", type=float, help="Mesh 0 threshold")
inputParser.add_argument("-thio", "--threshold-in-output", action='store_true')
inputParser.add_argument("-v", "--verbose", action='store_true', help="Verbose output")

args = inputParser.parse_args()

dir =args.dir
msh_input0 = args.msh_input0
msh_input1 = args.msh_input1
t0 = args.threshold0
t1 = args.threshold1
threshold_in_output = args.threshold_in_output 
verbose = args.verbose

thio = "TH{}_".format(str(t1).replace('.','d')) if (threshold_in_output) else ""
msh_output = thio + args.msh_output

iou.cout("Parsed arguments", print2console=verbose)

iou.cout("Loading meshes", print2console=verbose)
msh_input0 += ".vtk" if ('.vtk' not in msh_input0) else ""
msh0 = vtku.readVtk(iou.fullfile(dir, msh_input0))

msh_input1 += ".vtk" if ('.vtk' not in msh_input1) else ""
msh1 = vtku.readVtk(iou.fullfile(dir, msh_input1))

iou.cout("Calculating fibrosis overlap", print2console=verbose)
omsh, counts = vtku.fibrosisOverlapCell(msh0, msh1, t0, t1)

iou.cout("Saving output mesh", print2console=verbose)
vtku.writeVtk(omsh, dir, msh_output)

iou.cout("Calculating fibrosis scores for meshes", print2console=verbose)
fib0 = vtku.fibrorisScore(msh0, t0) 
fib1 = vtku.fibrorisScore(msh1, t1)

iou.cout("Calculating performance metrics for msh1 ({})".format(msh_input1), print2console=verbose)
Tp = counts['overlap']
Tn = counts['none']
Fp = counts['msh1']
Fn = counts['msh0']

perf = iou.performanceMetrics(tp=Tp, tn=Tn, fp=Fp, fn=Fn)

outstr = "{},{},{},{},{},{},{},{},{},{},{},{}".format(
    t0, t1, fib0, fib1, Tp, Tn, Fp, Fn, perf['jaccard'], perf['precision'], perf['recall'], perf['accuracy'])

print("{},{},{},{},{}, jaccard".format(t0, t1, fib0, fib1, perf['jaccard']))
print("{},{},{},{},{}, precision".format(t0, t1, fib0, fib1, perf['precision']))
print("{},{},{},{},{}, recall".format(t0, t1, fib0, fib1, perf['recall']))
print("{},{},{},{},{}, accuracy".format(t0, t1, fib0, fib1, perf['accuracy']))
print("{},{},{},{},{}, dice".format(t0, t1, fib0, fib1, perf['dice']))

#perf['precision'], perf['recall'], perf['accuracy']