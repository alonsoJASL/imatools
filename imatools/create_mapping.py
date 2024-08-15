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

def execute_map(args) : 
    path_left = args.input1
    path_right = args.input2

    if args.name1 == '' : 
        id_left = os.path.basename(path_left).split('.')[0]
    else :
        id_left = args.name1
    
    if args.name2 == '' : 
        id_right = os.path.basename(path_right).split('.')[0]
    else :
        id_right = args.name2

    midic = vtku.create_mapping(msh_left_name=path_left, msh_right_name=path_right,
                            left_id=id_left, right_id=id_right, map_type=args.map_type)
    
    base_dir = os.path.dirname(path_left)
    df = pd.DataFrame(midic)
    odir = iou.mkdirplus(base_dir, 'MAPPING')
    oname = f'{id_left}_{id_right}_{args.map_type}.csv'

    df.to_csv(iou.fullfile(odir, oname), index=False)

def execute_compare(args) :
    if args.mapping == '' : 
        iou.cout('Mapping file is required', 'ERROR')
        sys.exit(1)

    if os.path.isdir(args.mapping) : 
        list_of_files = os.listdir(args.mapping)
    else : 
        list_of_files = [args.mapping]

    for f in list_of_files : 
        if f.split('.')[-1] != 'csv' : 
            continue

        mapping = pd.read_csv(iou.fullfile(args.mapping, f))
        distdic = iou.get_boxplot_values(mapping['distance_auto'])
        print(distdic)    

def main(args) : 
    if args.mode == 'map' : 
        execute_map(args)
    elif args.mode == 'compare' :
        execute_compare(args)
    else : 
        iou.cout('Invalid mode', 'ERROR')
        sys.exit(1)


if __name__ == '__main__' :
    inputParser = argparse.ArgumentParser(description="Map closest point/element on meshes")
    inputParser.add_argument('mode', choices=['map', 'compare'])

    map_group = inputParser.add_argument_group('map')
    map_group.add_argument('-in1', '--input1', type=str, help='Input file 1')
    map_group.add_argument('-in2', '--input2', type=str, help='Input file 2')
    map_group.add_argument('-n1', '--name1', type=str, help='Name on mapping file 1', default='')
    map_group.add_argument('-n2', '--name2', type=str, help='Name on mapping file 2', default='')
    map_group.add_argument("-map", "--map-type", choices=['elem', 'pts'], type=str)

    compare_group = inputParser.add_argument_group('compare')
    compare_group.add_argument('-m', '--mapping', type=str, help='Mapping file', default='')

    args = inputParser.parse_args()

    main(args)

    iou.cout('Goodbye', 'FINISHED')


