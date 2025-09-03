import os 
import argparse
import numpy as np
from typing import Dict, Any, Optional

import imatools.common.vtktools as vtku 
import imatools.common.itktools as itku

from imatools.common.CommandRunner import CommandRunner
from imatools.common.config import configure_logging

logger = configure_logging("vscar_projection")

class VScarPipeline:
    """Handles the complete ventricular scar projection pipeline."""
    
    def __init__(self, args):
        self.args = args
        self.current_path = args.input
    
    def _update_current_path(self, new_path: str) -> Dict[str, str]:
        """Update current path and return parsed info."""
        self.current_path = new_path
        return parse_input_name(new_path)
    
    def run_pipeline(self) -> str:
        """Execute the complete pipeline."""
        logger.info("Starting complete pipeline")
        
        # Step 1: Scale mesh
        logger.info("Step 1: Scaling mesh")
        info = parse_input_name(self.current_path)
        scaled_path = execute_scale_mesh(info, scale=self.args.scale, convert_format=self.args.convert_format)
        
        # Step 2: Deform mesh
        logger.info("Step 2: Deforming mesh")
        info = self._update_current_path(scaled_path)
        deformed_path = execute_deform_mesh(info, self.args.path_to_mirtk, self.args.path_to_moving, self.args.path_to_fixed)
        
        # Step 3: Calculate COG
        logger.info("Step 3: Calculating center of gravity")
        info = self._update_current_path(deformed_path)
        cog_path = execute_cog_mesh(info)
        
        # Step 4: Project scar
        logger.info("Step 4: Projecting scar")
        final_path = execute_vscar_projection(info, cog_path=cog_path, reference_image=self.args.reference_image, label=self.args.label)
        
        logger.info(f"Pipeline completed successfully. Final output: {final_path}")
        return final_path

def parse_input_name(input_path: str) -> Dict[str, str]:
    """Parse input path into components."""
    base = os.path.basename(input_path)
    dirname = os.path.dirname(input_path)
    name, ext = os.path.splitext(base)
    
    return {
        'base': base, 
        'dirname': dirname, 
        'name': name, 
        'ext': ext[1:]
    }

def execute_scale_mesh(input_info: dict, scale: float = 0.001, convert_format: bool = False) -> str:
    """
    Scale mesh and optionally convert format.

    USAGE: 

    python scripts/vscar_projection.py scale --input <path_to_msh_cine> --scale 0.001 [--format]
    
    """
    ext = input_info['ext']
    input_path = os.path.join(input_info['dirname'], input_info['name']) 
    output_path = os.path.join(input_info['dirname'], f"{input_info['name']}_mm")

    cmd_runner = CommandRunner()
    log_dir = os.path.join(input_info['dirname'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    cmd_runner.set_log_dir(log_dir=log_dir)

    executable = 'meshtool'
    arguments = ['convert',
                 f'-imsh={input_path}', 
                 f'-ifmt={ext}',
                 f'-scale={scale}',
                 f'-omsh={output_path}',
                 f'-ofmt={ext}']
    
    cmd = cmd_runner.build_command(executable, arguments)
    logger.info(f'Executing command: {cmd}')
    cmd_runner.run_command(cmd, expected_outputs=[f'{output_path}.{ext}'])
    
    if convert_format:
        fmt_swap_dict = {'vtk': 'carp_txt', 'pts': 'vtk', 'elem': 'vtk'}
        new_ext = fmt_swap_dict[ext]

        expected_outs = ([f'{output_path}.pts', f'{output_path}.elem'] 
                        if ext == 'vtk' else [f'{output_path}.vtk'])
        
        arguments[-1] = f'-ofmt={new_ext}'
        cmd = cmd_runner.build_command(executable, arguments)
        logger.info(f'Converting format with command: {cmd}')
        cmd_runner.run_command(cmd, expected_outputs=expected_outs)
    
    return f'{output_path}.vtk'

def execute_deform_mesh(input_info: dict, path_to_mirtk: str,  path_to_moving: str, path_to_fixed: str) -> str:
    """
    Register and deform mesh using MIRTK.
    
    USAGE: 

    python scripts/vscar_projection.py deform --input <path_to_msh_cine_mm> -mirtk <mirtk_libraries> -moving <path_to_cine> -fixed <path_to_LGE>
    """
    register_exe = os.path.join(path_to_mirtk, 'register')
    transform_exe = os.path.join(path_to_mirtk, 'transform-points')
    
    # Validate executables exist
    for exe, name in [(register_exe, 'register'), (transform_exe, 'transform-points')]:
        if not os.path.exists(exe):
            raise FileNotFoundError(f"Could not find MIRTK executable '{name}' in {exe}")

    dof_file = os.path.join(input_info['dirname'], 'rigid.dof')
    
    cmd_runner = CommandRunner()
    log_dir = os.path.join(input_info['dirname'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    cmd_runner.set_log_dir(log_dir=log_dir)
    
    # Step 1: Rigid registration
    arguments = [path_to_moving, path_to_fixed, f'-dofout {dof_file}',
                 '-model Rigid', '-verbose 3']
    cmd = cmd_runner.build_command(register_exe, arguments)
    logger.info(f'Executing registration: {cmd}')
    cmd_runner.run_command(cmd, expected_outputs=[dof_file])

    # Step 2: Apply transform to mesh points
    input_mesh_path = os.path.join(input_info['dirname'], input_info['base'])
    output_mesh_path = os.path.join(input_info['dirname'], 
                                  f"{input_info['name']}_on_LGE.{input_info['ext']}")

    arguments = [input_mesh_path, output_mesh_path, f'-dofin {dof_file}']
    cmd = cmd_runner.build_command(transform_exe, arguments)
    logger.info(f'Executing transformation: {cmd}')
    cmd_runner.run_command(cmd, expected_outputs=[output_mesh_path])

    return output_mesh_path

def execute_cog_mesh(input_info: dict) -> str:
    """
    Calculate center of gravity for mesh elements.
    
    USAGE:
    python scripts/vscar_projection.py cog --input <path_to_msh_cine_mm_on_LGE>

    """
    msh_path = os.path.join(input_info['dirname'], input_info['base'])
    output_name = input_info['base'].replace('.vtk', '.pts')
    
    msh = vtku.read_vtk(msh_path, input_type='ugrid') 
    cogs = vtku.cogs_from_ugrid(msh) 

    output_path = os.path.join(input_info['dirname'], output_name)
    logger.info(f'Saving COG file to: {output_path}')
    np.savetxt(output_path, cogs, delimiter=' ') 

    return output_path

def execute_vscar_projection(input_info: dict, cog_path: str, reference_image: str, label: int) -> str:
    """
    Project ventricular scar onto mesh.
    
    USAGE:
    python scripts/vscar_projection.py scar --input <path_to_cog_file> -ref <path_to_LGE_segmentation> -label <SCAR_LABEL>
    """
    msh_path = os.path.join(input_info['dirname'], input_info['base'])
    output_msh_name = f'scar3d_{input_info["base"]}'

    cogs = np.loadtxt(cog_path)
    img = itku.load_image(reference_image)
    _, _, bboxes_dict = itku.get_indices_from_label(img, label, get_voxel_bbox=True)

    msh = vtku.read_vtk(msh_path, input_type='ugrid')
    # outmsh = vtku.tag_mesh_elements_by_growing_from_seed(msh, bboxes_dict['centres'], bboxes_dict['corners'],  cogs=cogs, label_name='scar')
    outmsh = vtku.tag_mesh_elements_by_growing_from_seed_optimized(msh, bboxes_dict['centres'], bboxes_dict['corners'],  cogs=cogs, label_name='scar')
    
    output_path = os.path.join(input_info['dirname'], f'{output_msh_name}')
    vtku.write_vtk(outmsh, input_info['dirname'], output_msh_name, output_type='ugrid')

    return output_path

def validate_pipeline_args(args) -> None:
    """Validate required arguments for pipeline mode."""
    required_for_pipeline = {
        'path_to_mirtk': 'MIRTK path',
        'path_to_moving': 'moving image path',
        'path_to_fixed': 'fixed image path', 
        'reference_image': 'reference image path'
    }
    
    missing = [desc for arg, desc in required_for_pipeline.items() 
               if getattr(args, arg) is None]
    
    if missing:
        raise ValueError(f"Pipeline mode requires: {', '.join(missing)}")
    
def update_arguments_for_cwd(args, input_info) -> None:
    """Update argument paths based on inferred working directory."""
    if args.infer_working_directory:
        logger.info("Inferring working directory from input file path")
        if args.path_to_moving:
            args.path_to_moving = os.path.join(input_info['dirname'], os.path.basename(args.path_to_moving))
        if args.path_to_fixed:
            args.path_to_fixed = os.path.join(input_info['dirname'], os.path.basename(args.path_to_fixed))
        if args.reference_image:
            args.reference_image = os.path.join(input_info['dirname'], os.path.basename(args.reference_image))
    else :
        logger.info("Using provided paths without inferring working directory")

def main(args):
    """
    Main execution function.

    Mode descriptions:
      pipeline  - Run complete pipeline (scale -> deform -> cog -> scar)
      scale     - Scale mesh (input: msh_cine -> output: msh_cine_mm)
      deform    - Deform mesh (input: msh_cine_mm -> output: msh_cine_mm_on_LGE)  
      cog       - Calculate COG (input: msh_cine_mm_on_LGE -> output: *.pts)
      scar      - Project scar (input: COG file -> output: scar3d_*.vtk)

      USAGE: 

      python scripts/vscar_projection.py pipeline --input <path_to_msh_cine> -mirtk <mirtk_libraries> -moving <path_to_cine> -fixed <path_to_LGE> -ref <path_to_LGE_segmentation> -label <SCAR_LABEL>

      CWD_OPTION USAGE: 
      To infer working directory from input file path, use --infer-working-directory or -cwd flag.

      python scripts/vscar_projection.py pipeline --input <PATH_to_msh_cine> -mirtk <mirtk_libraries> -moving <cine_filename> -fixed <LGE_filename> -ref <LGE_segmentation_filename> -label <SCAR_LABEL> -cwd
    """
    input_info = parse_input_name(args.input)
    mode = args.mode
    update_arguments_for_cwd(args, input_info)

    try:
        if mode == 'pipeline':
            validate_pipeline_args(args)
            pipeline = VScarPipeline(args)
            final_output = pipeline.run_pipeline()
            logger.info(f"Pipeline completed. Final output: {final_output}")
            
        elif mode == 'scale':
            execute_scale_mesh(input_info, scale=args.scale, 
                             convert_format=args.convert_format)
            
        elif mode == 'deform':

            execute_deform_mesh(input_info, path_to_mirtk=args.path_to_mirtk, 
                              path_to_moving=args.path_to_moving, 
                              path_to_fixed=args.path_to_fixed)
            
        elif mode == 'cog':
            execute_cog_mesh(input_info)
            
        elif mode == 'scar':
            execute_vscar_projection(input_info, cog_path=args.input, 
                                   reference_image=args.reference_image, label=args.label)
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
    except Exception as e:
        logger.error(f"Error in {mode} mode: {str(e)}")
        raise
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Ventricular Scar from CINE", usage=main.__doc__, add_help=False)
    
    parser.add_argument('mode', type=str,  choices=['pipeline', 'scale', 'deform', 'cog', 'scar'],  help='Mode of operation')
    parser.add_argument('--input', '-in', type=str, required=True, help='Input file path')
    parser.add_argument('--infer-working-directory', '-cwd', action='store_true', help='Infer working directory from input file path')

    # Scale options
    scale_group = parser.add_argument_group('Scale Options')
    scale_group.add_argument('--scale', '-scale', type=float, default=0.001,  help='Scaling factor (default: 0.001 for um to mm)')
    scale_group.add_argument('--convert-format', '-format', action='store_true',  help='Convert mesh format (vtk <-> carp_txt)')

    # Deform options  
    deform_group = parser.add_argument_group('Deform Options')
    deform_group.add_argument('--path-to-mirtk', '-mirtk', type=str,  help='Path to MIRTK executables folder')
    deform_group.add_argument('--path-to-moving', '-moving', type=str,  help='Path to moving image (CINE)')
    deform_group.add_argument('--path-to-fixed', '-fixed', type=str,  help='Path to fixed image (LGE)')

    # Scar projection options
    scar_group = parser.add_argument_group('Scar Projection Options')
    scar_group.add_argument('--reference-image', '-ref', type=str,  help='Path to reference image (LGE Segmentation)')
    scar_group.add_argument('--label', '-label', type=int, default=3,  help='Label for scar region (default: 3)')

    args = parser.parse_args()
    main(args)