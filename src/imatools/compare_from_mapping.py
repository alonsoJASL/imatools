import argparse
import os
import sys

import numpy as np
import pandas as pd
from common import ioutils as iou
from common import vtktools as vtku

import warnings
warnings.filterwarnings("ignore")

def compare_vector_field(v0, v1, r) :
    dotp = np.sum(np.multiply(v0, v1), axis=1)
    abs_dotp = np.abs(dotp)
    midic = {
        'region' : r, 
        'dot_product' : dotp, 
        'angle' : np.arccos(dotp),
        'abs_dot_product' : abs_dotp, 
        'angle_from_absdot' : np.arccos(abs_dotp)
    }

    return midic

def compare_scalar_field(s0, s1) : 
    l2 = (s0-s1)**2 
    abs_diff = np.abs(s0-s1)
    midic = {
        'diff_square' : l2, 
        'diff_abs' : abs_diff,
        's0' : s0 ,
        's1' : s1
    }

    return midic


inputParser = argparse.ArgumentParser(description="Compare scalar/vector field on meshes")
inputParser.add_argument("-d", "--dir", metavar="dir", type=str, help="Directory with data")
inputParser.add_argument("-n", "--name", choices=['lat', 'gradlat', 'ps', 'f_endo', 'f_epi'], required=True, type=str)
inputParser.add_argument("-mm", "--max-distance", required=False, type=float, help='Maximum distance (mm) to consider with comparison')
inputParser.add_argument("-f", "--fibre", choices=['1', 'l'], required=True, type=str)
inputParser.add_argument("-debug", "--debug", action='store_true', help="Debug code")
inputParser.add_argument("-v", "--verbose", action='store_true', help="Verbose output")

args = inputParser.parse_args()


base_dir = args.dir
which_name = args.name
max_distance = args.max_distance
fibre = args.fibre
debug = args.debug
verbose = args.verbose

iou.cout('RUNNING ONLY FIRST COMPARISON', 'DEBUG', print2console=debug)

files_and_mapping = {
    'lat': ('LAT_RSPV_X.dat', 'fibre_X_pts.csv'),
    'gradlat': ('lat_X.gradmag.dat', 'fibre_X_pts.csv'),
    'ps': ('PSNodeSmooth.dat', 'input_pts.csv'),
    'f_endo': ('fibre_X_endo.lon', 'fibre_X_endo_elem.csv'),
    'f_epi': ('fibre_X_epi.lon', 'fibre_X_epi_elem.csv')
}

is_vector_field = ('f_' in which_name)

if is_vector_field and fibre is None :
    sys.exit("[ERROR] specify which fibre file with -f {1, l}")

dat_file = files_and_mapping[which_name][0]
map_name = files_and_mapping[which_name][1]

if fibre is not None : 
    dat_file = dat_file.replace('X', fibre)
    map_name = map_name.replace('X', fibre)

comparison_dir = iou.fullfile(base_dir, '011_comparisons')
df_comparisons = pd.read_csv(iou.fullfile(comparison_dir, 'comparisons_path.csv'))

iou.cout(dat_file, 'COMPARISON', print2console=verbose)
iou.cout(map_name, 'MAPPING', print2console=verbose)

max_distance_set = (max_distance is not None)
if max_distance_set : 
    iou.cout('Maximum distance = {} mm'.format(max_distance), 'DISTANCE', print2console=verbose)

CX = os.listdir(comparison_dir)
if 'comparisons_path.csv' in CX:
    CX.remove('comparisons_path.csv')

if 'measurements.csv' in CX:
    CX.remove('measurements.csv')

num_comparisons = 1 if debug else len(CX)
rows_to_skip = 1 if is_vector_field else 0

count = 0
if (verbose) : 
    iou.print_progress_bar(0, num_comparisons, prefix='Progress', suffix='', length=50)
for cx in range(num_comparisons) :
    subdir = CX[cx]
    case_path = iou.fullfile(comparison_dir, subdir)
    mapping_files_dir = iou.fullfile(case_path, 'MAPPING')

    df = pd.read_csv(iou.fullfile(mapping_files_dir, map_name))
    total = len(df)

    case0 = df.columns[0]
    case1 = df.columns[1]
    idx0 = df[case0]
    idx1 = df[case1]

    if max_distance_set : 
        idx0 = idx0[df.distance_manual <= max_distance]
        idx1 = idx1[df.distance_manual <= max_distance]

    arr0 = np.loadtxt(iou.fullfile(case_path, case0, dat_file), skiprows=rows_to_skip)
    arr1 = np.loadtxt(iou.fullfile(case_path, case1, dat_file), skiprows=rows_to_skip)

    if which_name == 'gradlat':
        arr_idx0 = 1/arr0[idx0]
        arr_idx1 = 1/arr1[idx1]
    else : 
        arr_idx0 = arr0[idx0]
        arr_idx1 = arr1[idx1]
    
    if is_vector_field : 
        _, _, r = iou.loadCarpMesh(dat_file[:-4], iou.fullfile(case_path, case0))
        r = r[idx0]
        my_dic = compare_vector_field(arr_idx0, arr_idx1, r) 
    else :
        my_dic = compare_scalar_field(arr_idx0, arr_idx1)
        my_dic[which_name + '_0'] = my_dic.pop('s0')
        my_dic[which_name + '_1'] = my_dic.pop('s1')
    
    my_dic['distance'] = df.distance_manual[idx0]
    
    odf = pd.DataFrame(my_dic)
    odir = iou.mkdirplus(case_path, 'COMPARISONS')
    oname = which_name
    oname += '' if (fibre is None) else '_' + fibre 
    oname += '.csv'

    odf.to_csv(iou.fullfile(odir, oname), index=False)
    if debug : 
        print(odf)
    
    if (verbose) : 
        count += 1
        iou.print_progress_bar(count, num_comparisons, prefix='Progress', suffix='', length=50)