"""
CLI entry point for scar quantification and ventricular scar projection tools.

Subcommands:
  lge               Create a synthetic LGE test image (prism) for debugging.
  surf              Create a segmentation surface mesh with MIRTK.
  scar-opts         Write a CEMRG scar-options JSON file.
  scar              Run CEMRG MitkCemrgScarProjectionOptions.
  mask              Blood-pool threshold masking of an image or segmentation.
  vscar-pipeline    Run the full ventricular scar projection pipeline.
  vscar-scale       Scale a mesh with meshtool.
  vscar-deform      Register and deform a mesh with MIRTK.
  vscar-cog         Calculate centre-of-gravity for mesh elements.
  vscar-project     Project ventricular scar onto a mesh.
  enhance           Enhance scar corridor labels by LGE intensity thresholds.
  check             Convert a scar-corridor CSV to a VTK vector field.

Consolidates: scarq_tools.py, vscar_projection.py, enhance_debug_scar.py,
pool_enhance_debug_scar.py, scr_check_scar.py, common/scarqtools.py.

ScarQuantificationTools external-command orchestration (run_scar/run_clip/
create_segmentation_mesh/clip_mitral_valve/check_mirtk/check_cemrg) is inlined
here as module-level helpers using pycemrg.system.CommandRunner.

ScarQuantificationTools persistent state (cemrg/mirtk dirs, cmd names) is
modelled as a ScarConfig dataclass; save/load via io.scar_io.
"""

import argparse
import json
import logging
import multiprocessing
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import SimpleITK as sitk
import vtk
from pycemrg.system import CommandExecutionError, CommandRunner

from imatools.core import image as core_image
from imatools.core import scar as core_scar
from imatools.io import scar_io

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# External tools (MIRTK/CEMRG/meshtool) can emit this benign error on success.
_IGNORE_ERRORS: List[str] = ["unordered_map::at()"]

# ---------------------------------------------------------------------------
# ScarConfig — explicit data contract for state (replaces ScarQuantificationTools
# mutable state, hardcoded paths removed entirely)
# ---------------------------------------------------------------------------


@dataclass
class ScarConfig:
    """Paths and command names for the scar pipeline."""

    cemrg_dir: str = ""
    mirtk_dir: str = ""
    scar_cmd_name: str = "MitkCemrgScarProjectionOptions"
    clip_cmd_name: str = "MitkCemrgApplyExternalClippers"

    def to_state_dict(self) -> dict:
        """Return a platform-keyed state dict compatible with io.scar_io."""
        platform = sys.platform
        return {
            "cemrg": {platform: self.cemrg_dir},
            "mirtk": {platform: self.mirtk_dir},
            "scar_cmd_name": self.scar_cmd_name,
            "clip_cmd_name": self.clip_cmd_name,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "ScarConfig":
        platform = sys.platform
        return cls(
            cemrg_dir=state["cemrg"].get(platform, ""),
            mirtk_dir=state["mirtk"].get(platform, ""),
            scar_cmd_name=state.get("scar_cmd_name", "MitkCemrgScarProjectionOptions"),
            clip_cmd_name=state.get("clip_cmd_name", "MitkCemrgApplyExternalClippers"),
        )

    def save(self, path: str) -> None:
        scar_io.save_scar_state(path, self.to_state_dict())

    @classmethod
    def load(cls, path: str) -> "ScarConfig":
        return cls.from_state_dict(scar_io.load_scar_state(path))


# ---------------------------------------------------------------------------
# External-command helpers (scarqtools orchestration, inlined here)
# ---------------------------------------------------------------------------


def _check_mirtk(mirtk_dir: str, test: str = "close-image") -> bool:
    """Return True if the MIRTK test binary exists in ``mirtk_dir``."""
    test_cmd = os.path.join(mirtk_dir, test)
    return os.path.isfile(test_cmd) or os.path.isfile(f"{test_cmd}.exe")


def _check_cemrg(cemrg_dir: str, test: str = "MitkCemrgScarProjectionOptions") -> bool:
    """Return True if the CEMRG test binary exists in ``cemrg_dir``."""
    return os.path.isfile(os.path.join(cemrg_dir, test))


def _run_cmd(
    script_dir: str,
    cmd_name: str,
    arguments: Sequence[str],
    debug: bool = False,
) -> tuple:
    """Assemble and (unless debug) run an external command.

    Returns ``(status, cmd_str)`` — callers check ``status != 0``.
    ``debug=True`` only logs the command without executing it.
    """
    cmd_path = os.path.join(script_dir, cmd_name) if script_dir else cmd_name
    cmd = [cmd_path, *arguments]
    cmd_str = " ".join(str(t) for t in cmd)
    status = 0

    if debug:
        logger.info(cmd_str)
    else:
        try:
            CommandRunner(logger=logger).run(cmd, ignore_errors=_IGNORE_ERRORS)
        except CommandExecutionError as exc:
            status = exc.returncode or 1

    return status, cmd_str


def _create_segmentation_mesh(
    mirtk_dir: str,
    work_dir: str,
    pveins_file: str = "PVeinsCroppedImage.nii",
    iterations: int = 1,
    isovalue: float = 0.5,
    blur: float = 0.0,
    debug: bool = False,
) -> None:
    """Run MIRTK close-image → extract-surface → smooth-surface."""
    close_args = [
        os.path.join(work_dir, pveins_file),
        os.path.join(work_dir, "segmentation.s.nii"),
        "-iterations",
        str(iterations),
    ]
    st, _ = _run_cmd(mirtk_dir, "close-image", close_args, debug)
    if st != 0:
        logger.error("Error in close-image")

    extract_args = [
        os.path.join(work_dir, "segmentation.s.nii"),
        os.path.join(work_dir, "segmentation.vtk"),
        "-isovalue",
        str(isovalue),
        "-blur",
        str(blur),
    ]
    st, _ = _run_cmd(mirtk_dir, "extract-surface", extract_args, debug)
    if st != 0:
        logger.error("Error in extract-surface")

    smooth_args = [
        os.path.join(work_dir, "segmentation.vtk"),
        os.path.join(work_dir, "segmentation.vtk"),
    ]
    st, _ = _run_cmd(mirtk_dir, "smooth-surface", smooth_args, debug)
    if st != 0:
        logger.error("Error in smooth-surface")


def _clip_mitral_valve(
    cemrg_dir: str,
    clip_cmd_name: str,
    work_dir: str,
    pveins_file: str,
    mvi_name: str = "prodMVI.vtk",
    debug: bool = False,
) -> None:
    """Run CEMRG clip-mitral-valve command."""
    clip_args = [
        "-i",
        os.path.join(work_dir, pveins_file),
        "-mv",
        "-mvname",
        mvi_name,
    ]
    st, _ = _run_cmd(cemrg_dir, clip_cmd_name, clip_args, debug)
    if st != 0:
        logger.error("Error clipping mitral valve")


# ---------------------------------------------------------------------------
# State resolution helper (shared by surf/scar/mask subcommands)
# ---------------------------------------------------------------------------


def _resolve_config(args: argparse.Namespace) -> ScarConfig:
    """Load ScarConfig from state file if it exists, else build from args."""
    state_file = getattr(args, "scarq_state", "")
    if state_file and os.path.isfile(state_file):
        try:
            cfg = ScarConfig.load(state_file)
            logger.info(f"Loaded scar state from {state_file}")
            return cfg
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Could not load state file {state_file}: {exc}. Using args.")

    # Build from explicit CLI args; append platform suffix to cmd names on Win32.
    scar_cmd = getattr(args, "scar_cmd", "MitkCemrgScarProjectionOptions")
    clip_cmd = getattr(args, "clip_cmd", "MitkCemrgApplyExternalClippers")
    if sys.platform == "win32":
        if not scar_cmd.endswith(".bat"):
            scar_cmd += "_release.bat"
        if not clip_cmd.endswith(".bat"):
            clip_cmd += "_release.bat"

    cfg = ScarConfig(
        cemrg_dir=getattr(args, "cemrg_dir", ""),
        mirtk_dir=getattr(args, "mirtk_dir", ""),
        scar_cmd_name=scar_cmd,
        clip_cmd_name=clip_cmd,
    )
    # Persist for subsequent calls.
    if state_file:
        cfg.save(state_file)
    return cfg


# ---------------------------------------------------------------------------
# Shared path helpers
# ---------------------------------------------------------------------------


def _parse_input_name(input_path: str) -> Dict[str, str]:
    """Parse input path into components (mirrors vscar_projection)."""
    base = os.path.basename(input_path)
    dirname = os.path.dirname(input_path)
    name, ext = os.path.splitext(base)
    return {"base": base, "dirname": dirname, "name": name, "ext": ext[1:]}


def _vtk_version() -> float:
    return vtk.vtkVersion().GetVTKMajorVersion() + 0.1 * vtk.vtkVersion().GetVTKMinorVersion()


# ---------------------------------------------------------------------------
# Subcommand handlers — scarq quantification
# ---------------------------------------------------------------------------


def handle_lge(args: argparse.Namespace) -> int:
    """Create synthetic LGE test image (prism)."""
    from imatools.common.itktools import save_image  # noqa: PLC0415

    image_size = tuple(args.lge_image_size)
    prism_size = tuple(args.lge_prism_size)
    origin = tuple(args.lge_origin)
    spacing = tuple(args.lge_spacing)

    im, seg, boundic = core_image.generate_scar_image(
        image_size=image_size,
        prism_size=prism_size,
        origin=origin,
        spacing=spacing,
        mode=args.lge_method,
        simple=args.lge_simple,
    )

    if im is None:
        logger.error("Failed to generate scar image.")
        return 1

    if args.base_dir:
        out_dir = args.base_dir
    elif args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
    else:
        out_dir = os.getcwd()

    out_name = os.path.basename(args.output) if args.output else "dcm-LGE_image_debug.nii"

    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Saving LGE image to {out_dir}")
    save_image(im, out_dir, out_name)
    save_image(seg, out_dir, "LA.nii")
    with open(os.path.join(out_dir, "bounds.json"), "w") as f:
        json.dump(boundic, f)
    return 0


def handle_surf(args: argparse.Namespace) -> int:
    """Create segmentation surface mesh via MIRTK."""
    from imatools.common.itktools import (  # noqa: PLC0415
        check_for_existing_label,
        exchange_labels,
        extract_single_label,
        get_labels,
        load_image,
        save_image,
    )
    from imatools.common.vtktools import (
        join_vtk,
        readVtk,
        set_cell_scalars,
        writeVtk,
    )  # noqa: PLC0415

    cfg = _resolve_config(args)

    mirtk_test = "close-image"
    if sys.platform == "win32":
        mirtk_test += ".exe"
    if not _check_mirtk(cfg.mirtk_dir, mirtk_test):
        logger.error(f"MIRTK not found in {cfg.mirtk_dir}. Exiting.")
        return 1

    if args.input is None:
        logger.error("No input file specified. Exiting.")
        return 1

    pveins_file = os.path.basename(args.input)
    if args.base_dir:
        work_dir = args.base_dir
    else:
        work_dir = os.path.dirname(os.path.abspath(args.input))

    if not work_dir or not pveins_file:
        logger.error("No input file specified. Exiting.")
        return 1

    seg = load_image(os.path.join(work_dir, pveins_file))
    if check_for_existing_label(seg, 100):
        logger.info("Fixing segmentation padding values (label 100 → 0) before meshing")
        seg = exchange_labels(seg, 100, 0)
        save_image(seg, work_dir, pveins_file)

    iterations = args.surf_iterations
    isovalue = args.surf_isovalue
    blur = args.surf_blur
    debug = args.debug

    if args.surf_multilabel:
        logger.info("Using multilabel segmentation")
        labels = get_labels(seg)
        vtkout = None
        for label in labels:
            logger.info(f"Creating mesh for label {label}")
            seg_label = extract_single_label(seg, label)
            label_name = f"segmentation_{label}"
            save_image(seg_label, work_dir, f"{label_name}.nii")
            _create_segmentation_mesh(
                cfg.mirtk_dir, work_dir, f"{label_name}.nii", iterations, isovalue, blur, debug
            )
            os.rename(
                os.path.join(work_dir, "segmentation.vtk"),
                os.path.join(work_dir, f"{label_name}.vtk"),
            )
            vtklabel = readVtk(os.path.join(work_dir, f"{label_name}.vtk"))
            vtklabel = set_cell_scalars(vtklabel, label)
            writeVtk(vtklabel, work_dir, f"{label_name}.vtk")
            if vtkout is None:
                vtkout = readVtk(os.path.join(work_dir, f"{label_name}.vtk"))
            else:
                vtkout = join_vtk(vtkout, readVtk(os.path.join(work_dir, f"{label_name}.vtk")))
        if vtkout is not None:
            writeVtk(vtkout, work_dir, "segmentation.vtk")
    else:
        _create_segmentation_mesh(
            cfg.mirtk_dir, work_dir, pveins_file, iterations, isovalue, blur, debug
        )
        clip_mv = args.clip_mitral_valve
        if clip_mv is not None:
            _clip_mitral_valve(
                cfg.cemrg_dir, cfg.clip_cmd_name, work_dir, pveins_file, clip_mv, debug
            )

    if args.surf_output != "segmentation":
        logger.info(f"Renaming segmentation.vtk to {args.surf_output}.vtk")
        os.rename(
            os.path.join(work_dir, "segmentation.vtk"),
            os.path.join(work_dir, f"{args.surf_output}.vtk"),
        )
    return 0


def handle_scar_opts(args: argparse.Namespace) -> int:
    """Write scar options JSON file."""
    if args.input is None:
        logger.error("No input file specified. Exiting.")
        return 1

    if args.base_dir:
        work_dir = args.base_dir
        opts_file = os.path.basename(args.input)
    else:
        opts_file_path = os.path.abspath(args.input)
        work_dir = os.path.dirname(opts_file_path)
        opts_file = os.path.basename(opts_file_path)

    if args.output:
        output_dir = os.path.basename(args.output)
    else:
        output_dir = "OUTPUT"

    os.makedirs(work_dir, exist_ok=True)
    scar_io.create_scar_options_file(
        dir=work_dir,
        opt_file=opts_file,
        output_dir=output_dir,
        legacy=args.scar_opts_legacy,
    )
    return 0


def handle_scar(args: argparse.Namespace) -> int:
    """Run CEMRG MitkCemrgScarProjectionOptions."""
    if args.input is None:
        logger.error("No input LGE path specified. Exiting.")
        return 1

    cfg = _resolve_config(args)

    if args.base_dir:
        lge_path = os.path.join(args.base_dir, os.path.basename(args.input))
    else:
        lge_path = os.path.abspath(args.input)

    opts_path = args.scar_opts or ""
    if not lge_path or not opts_path:
        logger.error("No input files specified. Exiting.")
        return 1

    svp = args.scar_opts_svp
    with open(opts_path, "r") as f:
        json_opts = json.load(f)

    if "single_voxel_projection" not in json_opts or json_opts["single_voxel_projection"] != svp:
        logger.info(f"Rewriting [{opts_path}] to include single voxel projection option")
        json_opts["single_voxel_projection"] = svp
        with open(opts_path, "w") as f:
            json.dump(json_opts, f)

    arguments = ["-i", lge_path, "-seg", args.scar_seg, "-opts", opts_path]
    _run_cmd(cfg.cemrg_dir, cfg.scar_cmd_name, arguments, debug=args.debug)
    return 0


def handle_mask(args: argparse.Namespace) -> int:
    """Mask image/segmentation with blood-pool threshold."""
    from imatools.common.itktools import load_image, save_image  # noqa: PLC0415

    if args.base_dir:
        base = args.base_dir
        im_path = os.path.join(base, os.path.basename(args.input)) if args.input else ""
        mask_path = os.path.join(base, os.path.basename(args.mask)) if args.mask else ""
        thres_path = (
            os.path.join(base, os.path.basename(args.mask_threshold_file))
            if args.mask_threshold_file
            else ""
        )
        output = os.path.join(base, os.path.basename(args.output)) if args.output else ""
    else:
        im_path = os.path.abspath(args.input) if args.input else ""
        mask_path = os.path.abspath(args.mask) if args.mask else ""
        thres_path = os.path.abspath(args.mask_threshold_file) if args.mask_threshold_file else ""
        output = os.path.abspath(args.output) if args.output else ""

    im = load_image(im_path)
    mask = load_image(mask_path)
    ignore_im = None
    if args.mask_ignore:
        ignore_path = os.path.join(os.path.dirname(im_path), args.mask_ignore)
        ignore_im = load_image(ignore_path)

    meanbp, stdbp = scar_io.get_bloodpool_stats_from_file(thres_path)

    scar_method = args.scar_method

    if not args.mask_seg:
        masked_im = core_scar.mask_voxels_above_threshold(
            im,
            mask,
            meanbp,
            stdbp,
            scar_method,
            args.mask_threshold_value,
            args.mask_value,
            ignore_im,
        )
    else:
        seg2mask_path = os.path.join(os.path.dirname(im_path), args.mask_seg)
        masked_im = core_scar.mask_segmentation_above_threshold(
            seg2mask_path,
            im,
            mask,
            meanbp,
            stdbp,
            scar_method,
            args.mask_threshold_value,
            args.mask_value,
            ignore_im,
        )

    if not output:
        output = os.path.join(os.path.dirname(im_path), "masked.nii")

    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_image(masked_im, output)
    return 0


# ---------------------------------------------------------------------------
# VScar pipeline functions (inlined from vscar_projection.py, M1.6c)
# ---------------------------------------------------------------------------


def execute_scale_mesh(input_info: dict, scale: float = 0.001, convert_format: bool = False) -> str:
    """Scale mesh and optionally convert format via meshtool."""
    ext = input_info["ext"]
    input_path = os.path.join(input_info["dirname"], input_info["name"])
    output_path = os.path.join(input_info["dirname"], f"{input_info['name']}_mm")

    cmd_runner = CommandRunner(logger=logger)

    cmd = [
        "meshtool",
        "convert",
        f"-imsh={input_path}",
        f"-ifmt={ext}",
        f"-scale={scale}",
        f"-omsh={output_path}",
        f"-ofmt={ext}",
    ]
    logger.info(f'Executing command: {" ".join(cmd)}')
    cmd_runner.run(cmd, expected_outputs=[f"{output_path}.{ext}"], ignore_errors=_IGNORE_ERRORS)

    if convert_format:
        fmt_swap_dict = {"vtk": "carp_txt", "pts": "vtk", "elem": "vtk"}
        new_ext = fmt_swap_dict[ext]
        expected_outs = (
            [f"{output_path}.pts", f"{output_path}.elem"]
            if ext == "vtk"
            else [f"{output_path}.vtk"]
        )
        cmd[-1] = f"-ofmt={new_ext}"
        logger.info(f'Converting format with command: {" ".join(cmd)}')
        cmd_runner.run(cmd, expected_outputs=expected_outs, ignore_errors=_IGNORE_ERRORS)

    return f"{output_path}.vtk"


def execute_deform_mesh(
    input_info: dict, path_to_mirtk: str, path_to_moving: str, path_to_fixed: str
) -> str:
    """Register and deform mesh using MIRTK."""
    register_exe = os.path.join(path_to_mirtk, "register")
    transform_exe = os.path.join(path_to_mirtk, "transform-points")

    for exe, name in [(register_exe, "register"), (transform_exe, "transform-points")]:
        if not os.path.exists(exe):
            raise FileNotFoundError(f"Could not find MIRTK executable '{name}' in {exe}")

    dof_file = os.path.join(input_info["dirname"], "rigid.dof")
    cmd_runner = CommandRunner(logger=logger)

    # Rigid registration — each flag/value is a separate token.
    reg_cmd = [
        register_exe,
        path_to_moving,
        path_to_fixed,
        "-dofout",
        dof_file,
        "-model",
        "Rigid",
        "-verbose",
        "3",
    ]
    logger.info(f'Executing registration: {" ".join(reg_cmd)}')
    cmd_runner.run(reg_cmd, expected_outputs=[dof_file], ignore_errors=_IGNORE_ERRORS)

    # Apply transform to mesh points.
    input_mesh_path = os.path.join(input_info["dirname"], input_info["base"])
    output_mesh_path = os.path.join(
        input_info["dirname"], f"{input_info['name']}_on_LGE.{input_info['ext']}"
    )
    xfm_cmd = [transform_exe, input_mesh_path, output_mesh_path, "-dofin", dof_file]
    logger.info(f'Executing transformation: {" ".join(xfm_cmd)}')
    cmd_runner.run(xfm_cmd, expected_outputs=[output_mesh_path], ignore_errors=_IGNORE_ERRORS)

    return output_mesh_path


def execute_cog_mesh(input_info: dict) -> str:
    """Calculate centre-of-gravity for mesh elements."""
    from imatools.io import mesh_io  # noqa: PLC0415

    msh_path = os.path.join(input_info["dirname"], input_info["base"])
    output_name = input_info["base"].replace(".vtk", ".pts")
    msh = mesh_io.read_vtk(msh_path, input_type="ugrid")

    from imatools.core.mesh import cogs_from_ugrid  # noqa: PLC0415

    cogs = cogs_from_ugrid(msh)
    output_path = os.path.join(input_info["dirname"], output_name)
    logger.info(f"Saving COG file to: {output_path}")
    np.savetxt(output_path, cogs, delimiter=" ")
    return output_path


def execute_vscar_projection(
    input_info: dict, cog_path: str, reference_image: str, label: int
) -> str:
    """Project ventricular scar onto mesh."""
    from imatools.common.itktools import get_indices_from_label, load_image  # noqa: PLC0415
    from imatools.core.mesh import tag_mesh_elements_by_growing_from_seed_optimized  # noqa: PLC0415
    from imatools.io import mesh_io  # noqa: PLC0415

    msh_path = os.path.join(input_info["dirname"], input_info["base"])
    output_msh_name = f'scar3d_{input_info["base"]}'

    cogs = np.loadtxt(cog_path)
    img = load_image(reference_image)
    _, _, bboxes_dict = get_indices_from_label(img, label, get_voxel_bbox=True)

    msh = mesh_io.read_vtk(msh_path, input_type="ugrid")
    outmsh = tag_mesh_elements_by_growing_from_seed_optimized(
        msh,
        bboxes_dict["centres"],
        bboxes_dict["corners"],
        cogs=cogs,
        label_name="scar",
    )

    output_path = os.path.join(input_info["dirname"], output_msh_name)
    mesh_io.write_vtk(outmsh, input_info["dirname"], output_msh_name, output_type="ugrid")
    return output_path


def _validate_pipeline_args(args: argparse.Namespace) -> None:
    """Validate required arguments for pipeline mode."""
    required = {
        "path_to_mirtk": "MIRTK path",
        "path_to_moving": "moving image path",
        "path_to_fixed": "fixed image path",
        "reference_image": "reference image path",
    }
    missing = [desc for arg, desc in required.items() if getattr(args, arg, None) is None]
    if missing:
        raise ValueError(f"vscar-pipeline requires: {', '.join(missing)}")


def _update_arguments_for_cwd(args: argparse.Namespace, input_info: dict) -> None:
    """Update argument paths based on inferred working directory."""
    if getattr(args, "infer_working_directory", False):
        logger.info("Inferring working directory from input file path")
        for attr in ("path_to_moving", "path_to_fixed", "reference_image"):
            val = getattr(args, attr, None)
            if val:
                setattr(
                    args,
                    attr,
                    os.path.join(input_info["dirname"], os.path.basename(val)),
                )


class VScarPipeline:
    """Handles the complete ventricular scar projection pipeline."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.current_path = args.input

    def _update_current_path(self, new_path: str) -> dict:
        self.current_path = new_path
        return _parse_input_name(new_path)

    def run_pipeline(self) -> str:
        """Execute the complete scale → deform → cog → project pipeline."""
        logger.info("Starting complete pipeline")

        info = _parse_input_name(self.current_path)
        scaled_path = execute_scale_mesh(
            info, scale=self.args.scale, convert_format=self.args.convert_format
        )

        info = self._update_current_path(scaled_path)
        deformed_path = execute_deform_mesh(
            info, self.args.path_to_mirtk, self.args.path_to_moving, self.args.path_to_fixed
        )

        info = self._update_current_path(deformed_path)
        cog_path = execute_cog_mesh(info)

        final_path = execute_vscar_projection(
            info,
            cog_path=cog_path,
            reference_image=self.args.reference_image,
            label=self.args.label,
        )
        logger.info(f"Pipeline completed. Final output: {final_path}")
        return final_path


# ---------------------------------------------------------------------------
# Subcommand handlers — ventricular scar projection
# ---------------------------------------------------------------------------


def handle_vscar_scale(args: argparse.Namespace) -> int:
    """Scale mesh and optionally convert format."""
    info = _parse_input_name(args.input)
    execute_scale_mesh(info, scale=args.scale, convert_format=args.convert_format)
    return 0


def handle_vscar_deform(args: argparse.Namespace) -> int:
    """Register and deform mesh using MIRTK."""
    info = _parse_input_name(args.input)
    _update_arguments_for_cwd(args, info)
    execute_deform_mesh(
        info,
        path_to_mirtk=args.path_to_mirtk,
        path_to_moving=args.path_to_moving,
        path_to_fixed=args.path_to_fixed,
    )
    return 0


def handle_vscar_cog(args: argparse.Namespace) -> int:
    """Calculate centre-of-gravity for mesh elements."""
    info = _parse_input_name(args.input)
    execute_cog_mesh(info)
    return 0


def handle_vscar_project(args: argparse.Namespace) -> int:
    """Project ventricular scar onto mesh."""
    info = _parse_input_name(args.input)
    execute_vscar_projection(
        info,
        cog_path=args.input,
        reference_image=args.reference_image,
        label=args.label,
    )
    return 0


def handle_vscar_pipeline(args: argparse.Namespace) -> int:
    """Run the full ventricular scar projection pipeline."""
    _validate_pipeline_args(args)
    info = _parse_input_name(args.input)
    _update_arguments_for_cwd(args, info)
    pipeline = VScarPipeline(args)
    final_output = pipeline.run_pipeline()
    logger.info(f"Pipeline completed. Final output: {final_output}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand handler — enhance (collapse of enhance_debug_scar + pool_enhance)
# ---------------------------------------------------------------------------


def handle_enhance(args: argparse.Namespace) -> int:
    """Enhance scar corridor labels using LGE intensity thresholds."""
    im_path = args.input
    debug_scar_path = args.scar_corridor_image
    prod_stats_path = args.image_info_file

    im = sitk.ReadImage(im_path)
    scar = sitk.ReadImage(debug_scar_path)

    if im.GetSize() != scar.GetSize():
        raise ValueError("Image and scar corridor image have different sizes")

    meanbp, stdbp = scar_io.get_bloodpool_stats_from_file(prod_stats_path)
    threshold_values = core_scar.get_threshold_values(
        args.threshold, meanbp, stdbp, args.threshold_method
    )

    im_array = sitk.GetArrayFromImage(im)
    scar_array = sitk.GetArrayFromImage(scar)

    jobs = args.jobs
    if jobs == 1:
        # Serial path (enhance_debug_scar behaviour)
        enhanced_array = core_scar.enhance_scar_array(scar_array, im_array, threshold_values)
    else:
        # Parallel path (pool_enhance_debug_scar behaviour)
        n_workers = jobs if jobs > 0 else multiprocessing.cpu_count()

        def _process_voxel(xyz):
            x, y, z = xyz
            sv = scar_array[x, y, z]
            lv = im_array[x, y, z]
            if sv > 1:
                ev = 2
                for th in threshold_values:
                    ev += 1 if lv > th else 0
                return ev
            return sv

        indices = list(np.ndindex(scar_array.shape))
        with multiprocessing.Pool(processes=n_workers) as pool:
            enhanced_values = pool.map(_process_voxel, indices)

        enhanced_array = np.copy(scar_array)
        for idx, (x, y, z) in enumerate(indices):
            enhanced_array[x, y, z] = enhanced_values[idx]

    enhanced_scar = sitk.GetImageFromArray(enhanced_array)
    enhanced_scar.SetOrigin(scar.GetOrigin())
    enhanced_scar.SetSpacing(scar.GetSpacing())
    enhanced_scar.SetDirection(scar.GetDirection())

    debug_scar_dir = os.path.dirname(os.path.abspath(debug_scar_path))
    threshold_str = list(map(str, np.multiply(args.threshold, 100)))
    output_name = f'enhanced_debug_{"_".join(threshold_str).replace(".", "")}'

    sitk.WriteImage(enhanced_scar, os.path.join(debug_scar_dir, f"{output_name}.nii"))
    logger.info(f"Enhanced scar image saved to {debug_scar_dir}/{output_name}.nii")

    # Optional label-count side output (from enhance_debug_scar)
    if args.label_counts:
        enhanced_labels = np.unique(enhanced_array).tolist()
        if 0 in enhanced_labels:
            enhanced_labels.remove(0)
        if 1 in enhanced_labels:
            enhanced_labels.remove(1)

        label_counter = [0] * len(enhanced_labels)
        total_counter = 0
        for x in range(enhanced_array.shape[0]):
            for y in range(enhanced_array.shape[1]):
                for z in range(enhanced_array.shape[2]):
                    value = enhanced_array[x, y, z]
                    if value > 1:
                        total_counter += 1
                    if value in enhanced_labels:
                        label_counter[enhanced_labels.index(value)] += 1

        counts_path = os.path.join(debug_scar_dir, f"{output_name}_label_counts.txt")
        with open(counts_path, "w") as f:
            f.write(f"Total voxels: {total_counter}\n")
            for i, label in enumerate(enhanced_labels):
                f.write(f"{label}: {label_counter[i]}\n")
        logger.info(f"Label counts written to {counts_path}")

    return 0


# ---------------------------------------------------------------------------
# Subcommand handler — check (scr_check_scar.py)
# ---------------------------------------------------------------------------


def handle_check(args: argparse.Namespace) -> int:
    """Convert scar-corridor CSV to a VTK vector field."""
    default_output = args.output == ""
    inname = args.input
    # Fix: backwards extension check was 'outname in ".vtk"' — corrected to endswith.
    outname = inname[:-4] if default_output else args.output
    if not outname.endswith(".vtk"):
        outname += ".vtk"

    data = np.loadtxt(os.path.join(args.dir, inname), skiprows=3, delimiter=",")
    centres = data[:, 1:4]
    normals = data[:, -3:]
    normals_n = np.divide(normals.T, np.linalg.norm(normals, axis=1)).T

    min_step = -1
    max_step = 3
    min_step_positions = centres + min_step * normals_n
    max_step_positions = centres + max_step * normals_n

    if args.verbose:
        print(f"min step: {min_step}")
        print(min_step_positions)
        print(f"max step: {max_step}")
        print(max_step_positions)
        print("normals")
        print(normals_n)

    polydata = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    num_points = len(min_step_positions)

    # Fix: pts was never populated before SetPoints(pts) — add the points.
    for i in range(num_points):
        pts.InsertNextPoint(*min_step_positions[i])

    vf = vtk.vtkDoubleArray()
    vf.SetName("scar_corridor")
    vf.SetNumberOfComponents(3)
    vf.SetNumberOfTuples(num_points)
    for i in range(num_points):
        pt1 = min_step_positions[i]
        pt2 = max_step_positions[i]
        vf.SetTuple3(i, pt2[0] - pt1[0], pt2[1] - pt1[1], pt2[2] - pt1[2])

    polydata.SetPoints(pts)
    polydata.GetPointData().SetVectors(vf)

    if args.verbose:
        arrow_source = vtk.vtkArrowSource()
        glyph = vtk.vtkGlyph3D()
        glyph.SetSourceConnection(arrow_source.GetOutputPort())
        glyph.SetInputData(polydata)
        glyph.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)

        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)

        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)

        interactor.Initialize()
        render_window.Render()
        interactor.Start()

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polydata)
    writer.SetFileName(os.path.join(args.dir, outname))

    if _vtk_version() >= 9.1:
        writer.SetFileVersion(42)

    writer.Write()
    logger.info(f"VTK vector field written to {args.dir}/{outname}")
    return 0


# ---------------------------------------------------------------------------
# Argument-parser construction
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="imatools-scar",
        description="Scar quantification and ventricular scar projection tools.",
    )
    sub = p.add_subparsers(dest="command")

    # ---  Shared argument groups (reused via parents) ---

    _base = argparse.ArgumentParser(add_help=False)
    _base.add_argument("-dir", "--base-dir", type=str, default=None, help="Base directory")
    _base.add_argument("-i", "--input", type=str, default=None, help="Input file path")
    _base.add_argument("-o", "--output", type=str, default=None, help="Output file path")
    _base.add_argument("-d", "--debug", action="store_true", help="Debug mode (log only, no exec)")

    _scarq = argparse.ArgumentParser(add_help=False)
    _scarq.add_argument("--scarq-state", type=str, default="", help="State file path")
    _scarq.add_argument("--cemrg-dir", type=str, default="", help="CEMRG directory")
    _scarq.add_argument("--mirtk-dir", type=str, default="", help="MIRTK directory")
    _scarq.add_argument(
        "--scar-cmd",
        type=str,
        default="MitkCemrgScarProjectionOptions",
        help="Scar command name",
    )
    _scarq.add_argument(
        "--clip-cmd",
        type=str,
        default="MitkCemrgApplyExternalClippers",
        help="Clip command name",
    )

    # --- lge ---
    p_lge = sub.add_parser("lge", help="Create synthetic LGE test image (prism)", parents=[_base])
    p_lge.add_argument(
        "--lge-image-size", nargs=3, type=int, default=[300, 300, 100], help="Image size"
    )
    p_lge.add_argument(
        "--lge-prism-size", nargs=3, type=int, default=[80, 80, 80], help="Prism size"
    )
    p_lge.add_argument(
        "--lge-method", type=str, choices=["iir", "msd"], default="iir", help="Method"
    )
    p_lge.add_argument(
        "--lge-origin", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Image origin"
    )
    p_lge.add_argument(
        "--lge-spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0], help="Image spacing"
    )
    p_lge.add_argument("--lge-simple", action="store_true", help="Use simple generation method")
    p_lge.set_defaults(func=handle_lge)

    # --- surf ---
    p_surf = sub.add_parser(
        "surf",
        help="Create segmentation surface mesh with MIRTK",
        parents=[_base, _scarq],
    )
    p_surf.add_argument("--surf-multilabel", action="store_true")
    p_surf.add_argument("--surf-iterations", type=int, default=1)
    p_surf.add_argument("--surf-isovalue", type=float, default=0.5)
    p_surf.add_argument("--surf-blur", type=float, default=0.0)
    p_surf.add_argument("--clip-mitral-valve", type=str, default=None)
    p_surf.add_argument("--surf-output", type=str, default="segmentation")
    p_surf.set_defaults(func=handle_surf)

    # --- scar-opts ---
    p_so = sub.add_parser("scar-opts", help="Write CEMRG scar-options JSON file", parents=[_base])
    p_so.add_argument("--scar-opts-legacy", action="store_true", help="Use legacy ROI projection")
    p_so.set_defaults(func=handle_scar_opts)

    # --- scar ---
    p_scar = sub.add_parser(
        "scar",
        help="Run CEMRG MitkCemrgScarProjectionOptions",
        parents=[_base, _scarq],
    )
    p_scar.add_argument(
        "--scar-seg",
        type=str,
        default="PVeinsCroppedImage.nii",
        help="Segmentation file name",
    )
    p_scar.add_argument("--scar-opts", type=str, default=None, help="Options JSON path")
    p_scar.add_argument("--scar-opts-svp", action="store_true", help="Single voxel projection")
    p_scar.set_defaults(func=handle_scar)

    # --- mask ---
    p_mask = sub.add_parser(
        "mask",
        help="Blood-pool threshold masking",
        parents=[_base],
    )
    p_mask.add_argument(
        "--scar-method",
        type=str,
        choices=["iir", "msd"],
        default="iir",
        help="Scar quantification method",
    )
    p_mask.add_argument("--mask-seg", type=str, default="", help="Segmentation to mask")
    p_mask.add_argument("--mask", type=str, default="", help="Mask image")
    p_mask.add_argument("--mask-threshold-file", type=str, default="", help="prodStats.txt path")
    p_mask.add_argument(
        "--mask-threshold-value", type=float, default=0.0, help="Threshold multiplier"
    )
    p_mask.add_argument("--mask-value", type=float, default=0.0, help="Value for masked voxels")
    p_mask.add_argument("--mask-ignore", type=str, default="", help="Image to exclude from mask")
    p_mask.set_defaults(func=handle_mask)

    # --- vscar-pipeline ---
    _vscar_base = argparse.ArgumentParser(add_help=False)
    _vscar_base.add_argument("--input", "-in", type=str, required=True, help="Input file path")
    _vscar_base.add_argument(
        "--infer-working-directory",
        "-cwd",
        action="store_true",
        help="Infer working directory from input file",
    )

    p_vp = sub.add_parser(
        "vscar-pipeline",
        help="Run full ventricular scar projection pipeline",
        parents=[_vscar_base],
    )
    p_vp.add_argument("--scale", "-scale", type=float, default=0.001)
    p_vp.add_argument("--convert-format", "-format", action="store_true")
    p_vp.add_argument("--path-to-mirtk", "-mirtk", type=str, default=None)
    p_vp.add_argument("--path-to-moving", "-moving", type=str, default=None)
    p_vp.add_argument("--path-to-fixed", "-fixed", type=str, default=None)
    p_vp.add_argument("--reference-image", "-ref", type=str, default=None)
    p_vp.add_argument("--label", "-label", type=int, default=3)
    p_vp.set_defaults(func=handle_vscar_pipeline)

    # --- vscar-scale ---
    p_vs = sub.add_parser(
        "vscar-scale",
        help="Scale a mesh with meshtool",
        parents=[_vscar_base],
    )
    p_vs.add_argument("--scale", "-scale", type=float, default=0.001)
    p_vs.add_argument("--convert-format", "-format", action="store_true")
    p_vs.set_defaults(func=handle_vscar_scale)

    # --- vscar-deform ---
    p_vd = sub.add_parser(
        "vscar-deform",
        help="Register and deform mesh with MIRTK",
        parents=[_vscar_base],
    )
    p_vd.add_argument("--path-to-mirtk", "-mirtk", type=str, required=True)
    p_vd.add_argument("--path-to-moving", "-moving", type=str, required=True)
    p_vd.add_argument("--path-to-fixed", "-fixed", type=str, required=True)
    p_vd.set_defaults(func=handle_vscar_deform)

    # --- vscar-cog ---
    p_vc = sub.add_parser(
        "vscar-cog",
        help="Calculate centre-of-gravity for mesh elements",
        parents=[_vscar_base],
    )
    p_vc.set_defaults(func=handle_vscar_cog)

    # --- vscar-project ---
    p_vpr = sub.add_parser(
        "vscar-project",
        help="Project ventricular scar onto mesh",
        parents=[_vscar_base],
    )
    p_vpr.add_argument("--reference-image", "-ref", type=str, required=True)
    p_vpr.add_argument("--label", "-label", type=int, default=3)
    p_vpr.set_defaults(func=handle_vscar_project)

    # --- enhance ---
    p_en = sub.add_parser(
        "enhance",
        help="Enhance scar corridor labels by LGE intensity thresholds",
    )
    p_en.add_argument("--input", "-in", type=str, required=True, help="LGE image path")
    p_en.add_argument(
        "--scar-corridor-image",
        "-scar-im",
        type=str,
        required=True,
        help="DEBUG scar corridor image",
    )
    p_en.add_argument(
        "--image-info-file", "-info", type=str, required=True, help="prodStats.txt file"
    )
    p_en.add_argument(
        "--threshold-method",
        "-m",
        choices=["iir", "msd"],
        required=True,
        help="Threshold method",
    )
    p_en.add_argument(
        "--threshold", "-thres", nargs="+", type=float, required=True, help="Threshold value(s)"
    )
    p_en.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel workers (1=serial, -1=all CPUs)",
    )
    p_en.add_argument(
        "--label-counts",
        action="store_true",
        help="Write label count side file (serial-mode behaviour from enhance_debug_scar)",
    )
    p_en.set_defaults(func=handle_enhance)

    # --- check ---
    p_chk = sub.add_parser(
        "check",
        help="Convert scar-corridor CSV to VTK vector field",
    )
    p_chk.add_argument("-d", "--dir", type=str, required=True, help="Folder with data")
    p_chk.add_argument("-i", "--input", type=str, required=True, help="Input CSV filename")
    p_chk.add_argument("-o", "--output", type=str, default="", help="Output filename")
    p_chk.add_argument("-v", "--verbose", action="store_true", help="Render arrow glyphs")
    p_chk.set_defaults(func=handle_check)

    return p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(args=None):
    parser = _build_parser()
    parsed = parser.parse_args(args)

    if parsed.command is None:
        parser.print_help()
        return 1

    if not hasattr(parsed, "func"):
        parser.print_help()
        return 1

    return parsed.func(parsed)


if __name__ == "__main__":
    sys.exit(main())
