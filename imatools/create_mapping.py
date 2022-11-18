import argparse
import os
import sys

import numpy as np
import pandas as pd
from common import ioutils as iou
from common import vtktools as vtku

def remove_from_dirlist(dir_list, name_to_rm) : 
    if name_to_rm in dir_list:
        dir_list.remove(name_to_rm)
    
    return dir_list

inputParser = argparse.ArgumentParser(description="Map closest point/element on meshes")
inputParser.add_argument("-d", "--dir", metavar="dir", type=str, help="Directory with data")
inputParser.add_argument("-n", "--name", choices=['in', 'l', '1', 'scar'], type=str)
inputParser.add_argument("-map", "--map-type", choices=['elem', 'pts'], type=str)
inputParser.add_argument("-layer", "--layer", choices=['endo', 'epi'], required=False, help="Define layer for fibre files")
inputParser.add_argument("-v", "--verbose", action='store_true', help="Verbose output")

args = inputParser.parse_args()


base_dir = args.dir
which_name = args.name
map_type = args.map_type
layer = args.layer
verbose = args.verbose

if which_name == 'in' or which_name == 'scar' : 
    layer = None

iou.cout('Calculating {}-mapping'.format(map_type), 'START')

num_comparisons = 50
N = ['M' + str(n) for n in np.linspace(1, 100, num=100, dtype=int)]
names = {'scar': 'scar', 'l': 'fibre_l', '1': 'fibre_1', 'in': 'input'}
mname = names[which_name] if (layer is None) else names[which_name] + '_' + layer

mname_ext = mname + '.vtk'

comparison_dir = [iou.fullfile(base_dir, '011_comparisons', 'C'+str(c)) for c in np.arange(num_comparisons)]

iou.cout(base_dir, 'BASE_DIRECTORY', print2console=verbose)
iou.cout(which_name, 'FILE', print2console=verbose)
iou.cout(mname, 'MSH_NAME', print2console=verbose)
iou.cout(map_type, 'MAPPING_TYPE', print2console=verbose)

iou.cout('Working...')
count=0
if (verbose) : 
    iou.print_progress_bar(0, num_comparisons, prefix='Progress', suffix='', length=50)
for this_comparison in comparison_dir : 
    
    sub_dirs = remove_from_dirlist(os.listdir(this_comparison), 'MAPPING')
    sub_dirs = remove_from_dirlist(sub_dirs, 'COMPARISONS')
        
    id_left = sub_dirs[0]
    id_right = sub_dirs[1]
    path_left = iou.fullfile(this_comparison, id_left, mname_ext)
    path_right = iou.fullfile(this_comparison, id_right, mname_ext)

    midic = vtku.create_mapping(msh_left_name=path_left, msh_right_name=path_right,
                            left_id=id_left, right_id=id_right, map_type=map_type)

    df = pd.DataFrame(midic)
    odir = iou.mkdirplus(this_comparison, 'MAPPING')
    oname = mname + '_' +  map_type + '.csv'

    df.to_csv(iou.fullfile(odir, oname), index=False)
    if (verbose) : 
        count += 1
        iou.print_progress_bar(count, num_comparisons, prefix='Progress', suffix='', length=50)


iou.cout('Goodbye', 'FINISHED')
