import os
import sys
import json 
import numpy as np

def get_empty_pot() -> dict:
    pot = {
            'segmentation' : {
                'seg_dir': None,
                'seg_name': None,
                'mesh_from_segmentation': True, 
                'boundary_relabeling': False}, # bad spelling is on purpose
            'meshing': {
                'facet_angle': 30,
                'facet_size': 0.8,
                'facet_distance': 0.1,
                'cell_rad_edge_ratio': 2,
                'cell_size': 0.8,
                'rescaleFactor': 1000},
            'laplacesolver': {
                'abs_toll': 1e-6,
                'rel_toll': 1e-6,
                'itr_max': 700,
                'dimKrilovSp': 500,
                'verbose': True},
            'others': {
                'eval_thickness': False},
            'output': {
                'outdir': None,
                'name': None,
                'out_medit': False,
                'out_vtk': True, 
                'out_carp': True,
                'out_vtk_binary': False,
                'out_carp_binary': False,
                'out_potential': False}
            }

    return pot

def save_to_json(pot, out_path):
    with open(out_path, 'w') as f:
        json.dump(pot, f, indent=4)

    return

def load_from_json(in_path):
    in_path += '.json' if '.json' not in in_path else ''
    with open(in_path, 'r') as f:
        pot = json.load(f)
    return pot


# ---------------------------------------------------------------------------
# Re-export shim — functions moved to core/parfile and io/parfile_io (T2e).
# Legacy callers that import from this module continue to work.
# ---------------------------------------------------------------------------
from imatools.core.parfile import update_pot  # noqa: E402,F401
from imatools.io.parfile_io import load_from_par, save_pot  # noqa: E402,F401
