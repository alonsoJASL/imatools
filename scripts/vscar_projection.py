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
        final_path = execute_vscar_projection(info, cog_path=cog_path, reference_image=self.args.reference_image, label=self.args.label,
                                              field_name=self.args.field_name, label_value=self.args.label_value)
        
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
    output_name = input_info['name'] + '.pts'

    msh = vtku.read_vtk(msh_path, input_type='ugrid')
    if msh.GetNumberOfCells() == 0:
        raise ValueError(f"Mesh has no cells (failed to read as ugrid?): {msh_path}")

    cogs = vtku.cogs_from_ugrid(msh)

    output_path = os.path.join(input_info['dirname'], output_name)
    logger.info(f'Saving COG file to: {output_path}')
    np.savetxt(output_path, cogs, delimiter=' ')

    return output_path

def execute_vscar_projection(input_info: dict, cog_path: str, reference_image: str, label: int,
                             field_name: str = 'scar', label_value: int = 1) -> str:
    """
    Project ventricular scar onto mesh.

    `label` selects which voxels are scar in the LGE segmentation image;
    `field_name`/`label_value` control what is written onto the mesh: tagged
    cells get `label_value` in the cell field `field_name`, overlaid onto that
    field if it already exists (other cells keep their values), else created.

    USAGE:
    python scripts/vscar_projection.py scar --input <path_to_mesh> --cog <path_to_cog_file> -ref <path_to_LGE_segmentation> -label <SCAR_LABEL>
    """
    msh_path = os.path.join(input_info['dirname'], input_info['base'])
    output_msh_name = f'scar3d_{input_info["name"]}'

    msh = vtku.read_vtk(msh_path, input_type='ugrid')
    if msh.GetNumberOfCells() == 0:
        raise ValueError(f"Mesh has no cells (failed to read as ugrid?): {msh_path}")

    cogs = np.atleast_2d(np.loadtxt(cog_path))
    if len(cogs) != msh.GetNumberOfCells():
        raise ValueError(
            f"COG count ({len(cogs)}) does not match mesh cell count "
            f"({msh.GetNumberOfCells()}); cog file '{cog_path}' likely came from a different mesh."
        )

    img = itku.load_image(reference_image)
    _, _, bboxes_dict = itku.get_indices_from_label(img, label, get_voxel_bbox=True)

    outmsh = vtku.tag_mesh_elements_by_growing_from_seed_optimized(
        msh, bboxes_dict['centres'], bboxes_dict['corners'], cogs=cogs,
        label_name=field_name, label_value=label_value)

    vtku.write_vtk(outmsh, input_info['dirname'], output_msh_name, output_type='ugrid')

    output_path = os.path.join(input_info['dirname'], output_msh_name + '.vtk')
    logger.info(f'Saving scar mesh to: {output_path}')
    return output_path

def update_arguments_for_cwd(args, input_info) -> None:
    """Resolve auxiliary path arguments against the input's directory.

    Only acts when --infer-working-directory/-cwd is set. Tolerant of modes
    that do not declare every auxiliary path (uses getattr defaults).
    """
    if not getattr(args, 'infer_working_directory', False):
        logger.info("Using provided paths without inferring working directory")
        return

    logger.info("Inferring working directory from input file path")
    for attr in ('path_to_moving', 'path_to_fixed', 'reference_image'):
        value = getattr(args, attr, None)
        if value:
            setattr(args, attr, os.path.join(input_info['dirname'], os.path.basename(value)))

# --- Mode handlers (one per subcommand) ------------------------------------

def _run_scale(args) -> None:
    info = parse_input_name(args.input)
    execute_scale_mesh(info, scale=args.scale, convert_format=args.convert_format)

def _run_deform(args) -> None:
    info = parse_input_name(args.input)
    update_arguments_for_cwd(args, info)
    execute_deform_mesh(info, path_to_mirtk=args.path_to_mirtk,
                        path_to_moving=args.path_to_moving,
                        path_to_fixed=args.path_to_fixed)

def _run_cog(args) -> None:
    info = parse_input_name(args.input)
    execute_cog_mesh(info)

def _run_scar(args) -> None:
    info = parse_input_name(args.input)
    update_arguments_for_cwd(args, info)
    execute_vscar_projection(info, cog_path=args.cog,
                             reference_image=args.reference_image, label=args.label,
                             field_name=args.field_name, label_value=args.label_value)

def _run_pipeline(args) -> None:
    info = parse_input_name(args.input)
    update_arguments_for_cwd(args, info)
    pipeline = VScarPipeline(args)
    final_output = pipeline.run_pipeline()
    logger.info(f"Pipeline completed. Final output: {final_output}")

def main(args) -> None:
    """Dispatch to the handler selected by the chosen subcommand."""
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Error in {args.mode} mode: {str(e)}")
        raise

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project ventricular scar from CINE onto LGE.")
    subparsers = parser.add_subparsers(dest='mode', required=True,
                                       metavar='{pipeline,scale,deform,cog,scar}')

    # `-cwd` is identical across the modes that use it; share it via `parents`.
    cwd = argparse.ArgumentParser(add_help=False)
    cwd.add_argument('--infer-working-directory', '-cwd', action='store_true',
                     help="Resolve auxiliary path basenames against the input's directory")

    # scale -----------------------------------------------------------------
    p_scale = subparsers.add_parser(
        'scale',
        help='Scale a mesh (um -> mm) and optionally convert format',
        description='Scale a mesh by a factor (default 0.001, um -> mm). '
                    'Optionally also write it in the swapped format '
                    '(vtk <-> carp_txt). Output: <name>_mm.<ext>.')
    p_scale.add_argument('--input', '-in', type=str, required=True,
                         help='Mesh to scale (vtk or carp_txt)')
    p_scale.add_argument('--scale', '-scale', type=float, default=0.001,
                         help='Scaling factor (default: 0.001 for um to mm)')
    p_scale.add_argument('--convert-format', '-format', action='store_true',
                         help='Also write the mesh in the swapped format (vtk <-> carp_txt)')
    p_scale.set_defaults(func=_run_scale)

    # deform ----------------------------------------------------------------
    p_deform = subparsers.add_parser(
        'deform', parents=[cwd],
        help='Rigidly register CINE->LGE and transform mesh points (MIRTK)',
        description='Rigidly register moving (CINE) to fixed (LGE) with MIRTK '
                    'and apply the transform to the mesh points. '
                    'Output: <name>_on_LGE.<ext>.')
    p_deform.add_argument('--input', '-in', type=str, required=True,
                          help='Scaled mesh (<name>_mm.vtk)')
    p_deform.add_argument('--path-to-mirtk', '-mirtk', type=str, required=True,
                          help='MIRTK executables folder (must contain register, transform-points)')
    p_deform.add_argument('--path-to-moving', '-moving', type=str, required=True,
                          help='Moving image (CINE)')
    p_deform.add_argument('--path-to-fixed', '-fixed', type=str, required=True,
                          help='Fixed image (LGE)')
    p_deform.set_defaults(func=_run_deform)

    # cog -------------------------------------------------------------------
    p_cog = subparsers.add_parser(
        'cog',
        help='Write per-element centres of gravity to a .pts file',
        description='Compute per-element centres of gravity of a deformed '
                    'unstructured-grid mesh and write them as a .pts file '
                    '(one cell centroid per line). Output: <name>.pts.')
    p_cog.add_argument('--input', '-in', type=str, required=True,
                       help='Deformed mesh (<name>_on_LGE.vtk, vtk ugrid)')
    p_cog.set_defaults(func=_run_cog)

    # scar ------------------------------------------------------------------
    p_scar = subparsers.add_parser(
        'scar', parents=[cwd], aliases=['projection'],
        help='Tag mesh elements as scar from an LGE segmentation label',
        description='Tag mesh elements as scar by growing from voxels of the '
                    'given label in the LGE segmentation. Requires both the '
                    'mesh (--input) and its COG file (--cog). '
                    'Output: scar3d_<mesh>.vtk.')
    p_scar.add_argument('--input', '-in', type=str, required=True,
                        help='Deformed mesh (vtk ugrid) to tag')
    p_scar.add_argument('--cog', '-cog', type=str, required=True,
                        help='COG .pts file for that mesh (from `cog` mode)')
    p_scar.add_argument('--reference-image', '-ref', type=str, required=True,
                        help='LGE segmentation image')
    p_scar.add_argument('--label', '-label', type=int, default=3,
                        help='Scar label in the SEGMENTATION image (which voxels are scar; default: 3)')
    p_scar.add_argument('--field-name', '-fname', type=str, default='scar',
                        help='Mesh cell field to modify or create (default: scar)')
    p_scar.add_argument('--label-value', '-lval', type=int, default=1,
                        help='Value written onto tagged mesh cells (default: 1)')
    p_scar.set_defaults(func=_run_scar)

    # pipeline --------------------------------------------------------------
    p_pipe = subparsers.add_parser(
        'pipeline', parents=[cwd],
        help='Run the full pipeline: scale -> deform -> cog -> scar',
        description='Run scale -> deform -> cog -> scar end to end. '
                    '--input is the raw CINE mesh; required paths are enforced '
                    'up front by argparse.')
    p_pipe.add_argument('--input', '-in', type=str, required=True,
                        help='Raw CINE mesh (vtk or carp_txt)')
    p_pipe.add_argument('--path-to-mirtk', '-mirtk', type=str, required=True,
                        help='MIRTK executables folder (must contain register, transform-points)')
    p_pipe.add_argument('--path-to-moving', '-moving', type=str, required=True,
                        help='Moving image (CINE)')
    p_pipe.add_argument('--path-to-fixed', '-fixed', type=str, required=True,
                        help='Fixed image (LGE)')
    p_pipe.add_argument('--reference-image', '-ref', type=str, required=True,
                        help='LGE segmentation image')
    p_pipe.add_argument('--label', '-label', type=int, default=3,
                        help='Scar label in the SEGMENTATION image (which voxels are scar; default: 3)')
    p_pipe.add_argument('--field-name', '-fname', type=str, default='scar',
                        help='Mesh cell field to modify or create (default: scar)')
    p_pipe.add_argument('--label-value', '-lval', type=int, default=1,
                        help='Value written onto tagged mesh cells (default: 1)')
    p_pipe.add_argument('--scale', '-scale', type=float, default=0.001,
                        help='Scaling factor (default: 0.001 for um to mm)')
    p_pipe.add_argument('--convert-format', '-format', action='store_true',
                        help='Also write the mesh in the swapped format (vtk <-> carp_txt)')
    p_pipe.set_defaults(func=_run_pipeline)

    return parser

if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)