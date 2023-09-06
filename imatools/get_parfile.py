import os
import sys
import json
import argparse 

from common import m3dutils as m3d 

def fill_with_args(args):
    pot = m3d.get_empty_pot()
    pot['segmentation']['seg_dir'] = args.seg_dir
    pot['segmentation']['seg_name'] = args.seg_name
    pot['segmentation']['mesh_from_segmentation'] = args.mesh_from_segmentation
    pot['segmentation']['boundary_relabelling'] = args.boundary_relabelling
    pot['meshing']['facet_angle'] = args.facet_angle
    pot['meshing']['facet_size'] = args.facet_size
    pot['meshing']['facet_distance'] = args.facet_distance
    pot['meshing']['cell_rad_edge_ratio'] = args.cell_rad_edge_ratio
    pot['meshing']['cell_size'] = args.cell_size
    pot['meshing']['rescaleFactor'] = args.rescale_factor
    pot['laplacesolver']['abs_toll'] = args.abs_toll
    pot['laplacesolver']['rel_toll'] = args.rel_toll
    pot['laplacesolver']['itr_max'] = args.itr_max
    pot['laplacesolver']['dimKrilovSp'] = args.dimKrilovSp
    pot['laplacesolver']['verbose'] = args.verbose
    pot['others']['eval_thickness'] = args.eval_thickness
    pot['output']['outdir'] = args.outdir
    pot['output']['name'] = args.name
    pot['output']['out_medit'] = args.out_medit
    pot['output']['out_vtk'] = args.out_vtk
    pot['output']['out_carp'] = args.out_carp
    pot['output']['out_vtk_binary'] = args.out_vtk_binary
    pot['output']['out_carp_binary'] = args.out_carp_binary
    pot['output']['out_potential'] = args.out_potential

    return pot

def error_in_paths(pot: dict): 
    mylist = [pot['segmentation']['seg_dir'], pot['segmentation']['seg_name'], pot['output']['outdir'], pot['output']['name']]
    mynamesl = ['seg_dir', 'seg_name', 'outdir', 'name']
    none_found = (None in mylist)

    if none_found:
        index_none = mylist.index(None)
        print(f'ERROR: missing argument [{mynamesl[index_none]}]')

    return none_found
        
    
def new_parfile(args_pot):
    final_pot = m3d.get_empty_pot()
    if error_in_paths(args_pot) :
        raise ValueError('Missing argument')
    
    final_pot = m3d.update_pot(final_pot, args_pot)

    return final_pot

def main(args):
    output_path = args.filepath
    output_path += '.par' if '.par' not in output_path else '' 
    
    args_pot = fill_with_args(args)

    if args.mode == 'new' :
        final_pot = new_parfile(args_pot)

    elif args.mode == 'json' :
        if args.template is None:
            raise ValueError('Please specify a template file')

        json_pot = m3d.load_from_json(args.template)
        final_pot = m3d.update_pot(json_pot, args_pot) 

        error_in_paths(final_pot)

    elif args.mode == 'par' :
        if args.template is None:
            raise ValueError('Please specify a template file')

        par_pot = m3d.load_from_par(args.template)
        final_pot = m3d.update_pot(par_pot, args_pot) 

        error_in_paths(final_pot)
    
    if args.json:
        json_out = output_path.replace('.par', '.json')
        m3d.save_to_json(final_pot, json_out)

    m3d.save_pot(final_pot, output_path)

if __name__ == '__main__':
    iparse = argparse.ArgumentParser()
    iparse.add_argument('mode', type=str, choices=['new', 'json', 'par'], help='Mode of operation')
    iparse.add_argument('--filepath', '-f', type=str, required=True, help='Output Meshtools3d parameter file')
    iparse.add_argument('--template', '-t', type=str, help='Path to template parameter file (JSON or .par)') 
    iparse.add_argument('--json', '-json', action="store_true", help='If set, saves output as a JSON file')
    iparse.add_argument('--seg_dir', '-seg_dir', type=str)
    iparse.add_argument('--seg_name', '-seg_name', type=str, 'converted.inr')
    iparse.add_argument('--mesh_from_segmentation', '-mesh_from_segmentation', type=bool, metavar='1/0')
    iparse.add_argument('--boundary_relabelling', '-boundary_relabelling', type=bool, metavar='1/0')
    iparse.add_argument('--facet_angle', '-facet_angle', type=float, help='Facet angle')
    iparse.add_argument('--facet_size', '-facet_size', type=float, help='Facet size')
    iparse.add_argument('--facet_distance', '-facet_distance', type=float, help='Facet distance')
    iparse.add_argument('--cell_rad_edge_ratio', '-cell_rad_edge_ratio', type=float, help='Cell radius edge ratio')
    iparse.add_argument('--cell_size', '-cell_size', type=float, help='Cell size')
    iparse.add_argument('--rescale_factor', '-rescaleFactor', type=float, help='Rescale factor')
    iparse.add_argument('--abs_toll', '-abs_toll', type=float, help='Absolute tolerance')
    iparse.add_argument('--rel_toll', '-rel_toll', type=float, help='Relative tolerance')
    iparse.add_argument('--itr_max', '-itr_max', type=int, help='Max iterations')
    iparse.add_argument('--dimKrilovSp', '-dimKrilovSp', type=int, help='Dimension of the Krylov space')
    iparse.add_argument('--verbose', '-verbose', type=bool)
    iparse.add_argument('--eval_thickness', '-eval_thickness', type=bool, metavar='1/0')
    iparse.add_argument('--outdir', '-outdir', type=str, default=os.getcwd(), help='Subdirectory of outputs in parfile')
    iparse.add_argument('--name', '-name', type=str, default='myocardium', help='Name of the output file')
    iparse.add_argument('--out_medit', '-out_medit', type=bool, metavar='1/0')
    iparse.add_argument('--out_vtk', '-out_vtk', type=bool, metavar='1/0')
    iparse.add_argument('--out_carp', '-out_carp', type=bool, metavar='1/0')
    iparse.add_argument('--out_vtk_binary', '-out_vtk_binary', type=bool, metavar='1/0')
    iparse.add_argument('--out_carp_binary', '-out_carp_binary', type=bool, metavar='1/0')
    iparse.add_argument('--out_potential', '-out_potential', type=bool, metavar='1/0')

    args = iparse.parse_args()
    main(args)
