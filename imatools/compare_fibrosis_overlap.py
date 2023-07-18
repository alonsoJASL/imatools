from common import ioutils as iou
from common import vtktools as vtku
import os
import argparse

def get_threshold_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        val = float(lines[0].strip())
        method = int(lines[1].strip())
        mean_bp = float(lines[2].strip())
        std_bp = float(lines[3].strip())
        thres = float(lines[4].strip())
    return (val, thres) 

def parse_threshold(threshold):
    try: 
        threshold = float(threshold)
        val = threshold
    except ValueError:
        if os.path.isfile(threshold): 
            (val, threshold) = get_threshold_from_file(threshold)
        else: 
            raise ValueError("Threshold is not a file or a number")

    return (val, threshold)

def main(args):
    dir =args.dir
    msh_input0 = args.msh_input0
    msh_input1 = args.msh_input1
    t0 = args.threshold0
    t1 = args.threshold1
    data_type = args.data_type
    threshold_in_output = args.threshold_in_output 
    verbose = args.verbose

    (val0, t0) = parse_threshold(t0)
    (val1, t1) = parse_threshold(t1)

    thio = "TH{}_".format(str(val1).replace('.','d')) if (threshold_in_output) else ""
    msh_output = thio + '_' + data_type + '_' + args.msh_output

    iou.cout("Parsed arguments", print2console=verbose)

    iou.cout("Loading meshes", print2console=verbose)
    msh_input0 += ".vtk" if ('.vtk' not in msh_input0) else ""
    msh0 = vtku.readVtk(iou.fullfile(dir, msh_input0))

    msh_input1 += ".vtk" if ('.vtk' not in msh_input1) else ""
    msh1 = vtku.readVtk(iou.fullfile(dir, msh_input1))

    iou.cout("Calculating fibrosis overlap", print2console=verbose)
    # omsh, counts = vtku.fibrosisOverlapCell(msh0, msh1, t0, t1)
    omsh, counts = vtku.fibrosis_overlap(msh0, msh1, t0, t1, type=data_type)

    iou.cout("Saving output mesh", print2console=verbose)
    vtku.writeVtk(omsh, dir, msh_output)

    iou.cout("Calculating fibrosis scores for meshes", print2console=verbose)
    fib0 = vtku.fibrosis_score(msh0, t0, type=data_type) 
    fib1 = vtku.fibrosis_score(msh1, t1, type=data_type)

    iou.cout("Calculating performance metrics for msh1 ({})".format(msh_input1), print2console=verbose)
    Tp = counts['overlap']
    Tn = counts['none']
    Fp = counts['msh1']
    Fn = counts['msh0']

    perf = iou.performanceMetrics(tp=Tp, tn=Tn, fp=Fp, fn=Fn)

    basestr = f'{args.id},{t0},{t1},{fib0},{fib1}'
    if args.output_type == 'df':
        print(f"{basestr},{perf['jaccard']}, jaccard")
        print(f"{basestr},{perf['precision']}, precision")
        print(f"{basestr},{perf['recall']}, recall")
        print(f"{basestr},{perf['accuracy']}, accuracy")
        print(f"{basestr},{perf['dice']}, dice")
    else : 
        print(f"{basestr},{Tp},{Tn},{Fp},{Fn},{perf['jaccard']},{perf['precision']},{perf['recall']},{perf['accuracy']}")

if __name__ == "__main__":
    inputParser = argparse.ArgumentParser(description="Compare fibrosis on meshes")
    inputParser.add_argument("-d", "--dir", metavar="dir", type=str, help="Directory with data")
    inputParser.add_argument("-imsh0", "--msh-input0", metavar="mshname", type=str, help="Source mesh name")
    inputParser.add_argument("-imsh1", "--msh-input1", metavar="mshname", type=str, help="Target mesh name")
    inputParser.add_argument("-omsh", "--msh-output", metavar="mshname", type=str, default='overlap', help="Output mesh name")
    inputParser.add_argument("-t0", "--threshold0", type=str, help="Mesh 0 threshold")
    inputParser.add_argument("-t1", "--threshold1", type=str, help="Mesh 0 threshold")
    inputParser.add_argument("-dt", "--data-type", type=str, choices=['cell', 'point'], default='cell')
    inputParser.add_argument("-thio", "--threshold-in-output", action='store_true')
    inputParser.add_argument("-id", "--id", type=str, default="ID")
    inputParser.add_argument("-type", "--output-type", type=str, choices=['df','compact'], default='df')
    inputParser.add_argument("-v", "--verbose", action='store_true', help="Verbose output")

    args = inputParser.parse_args()

    main(args)

