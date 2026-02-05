from common import ioutils as iou
from common import vtktools as vtku
import argparse

def main(args) : 
    dir_source =args.dir_source
    msh_source = args.msh_source

    dir_target =args.dir_target
    msh_target = args.msh_target

    dir_output =args.dir_output
    msh_output = args.msh_output

    data_type = args.data_type

    verbose = args.verbose
    iou.cout("Parsed arguments", print2console=verbose)

    msh_source += '.vtk' if ('.vtk' not in msh_source) else ""
    msh_target += '.vtk' if ('.vtk' not in msh_target) else ""

    msh_src = vtku.readVtk(iou.fullfile(dir_source, msh_source))
    msh_trg = vtku.readVtk(iou.fullfile(dir_target, msh_target))

    iou.cout("Projecting {} data".format(data_type), print2console=verbose)

    if (data_type == 'cell') : 
        msh_out = vtku.projectCellData(msh_source=msh_src, msh_target=msh_trg)
    elif (data_type == 'points') : 
        msh_out = vtku.projectPointData(msh_source=msh_src, msh_target=msh_trg)

    if ('.vtk' in msh_output) : 
        msh_output = msh_output[:-4]

    iou.cout("Writing file {}".format(iou.fullfile(dir_output, msh_output+'.vtk')), print2console=verbose)
    vtku.writeVtk(msh_out, dir_output, msh_output)

    iou.cout("Goodbye", print2console=verbose)

if __name__ == '__main__' :
    inputParser = argparse.ArgumentParser(description="Project scalar data of TARGET mesh onto SOURCE", epilog="NOTICE: Output mesh shape is source but with target scalars")
    inputParser.add_argument("-sdir", "--dir-source", metavar="dir", type=str, help="Directory with data (source)")
    inputParser.add_argument("-tdir", "--dir-target", metavar="dir", type=str, help="Directory with data (target)")
    inputParser.add_argument("-smsh", "--msh-source", metavar="mshname", type=str, help="Source mesh name")
    inputParser.add_argument("-tmsh", "--msh-target", metavar="mshname", type=str, help="Target mesh name")

    inputParser.add_argument("-odir", "--dir-output", metavar="dir", type=str, help="Output directory")
    inputParser.add_argument("-omsh", "--msh-output", metavar="mshname", type=str, default='output', help="Output mesh name")

    inputParser.add_argument("-dt", "--data-type", choices=['cell', 'point'], type=str)

    inputParser.add_argument("-v", "--verbose", action='store_true', help="Verbose output")

    args = inputParser.parse_args()

    main(args)
     

