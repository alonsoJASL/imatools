"""
Mesh file I/O operations.

Handles loading and saving of 3D meshes in various formats.
Dev functions (load_mesh, save_mesh, clean_stl_file) return/accept MeshContract
objects.  The remaining functions are ported verbatim from master's
``common/vtktools.py`` and operate on bare VTK objects.
"""

import logging
import os
from pathlib import Path
from typing import Literal, Union

import numpy as np
import vtk
import vtk.util.numpy_support as vtknp

from imatools.contracts import MeshContract, MeshType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy accessor — used by callers that may import before vtktools is ready.
# ---------------------------------------------------------------------------


def _vtk():  # noqa: ANN201
    """Return the imatools.common.vtktools module (lazy, avoids circular import)."""
    import imatools.common.vtktools as _vtkmod

    return _vtkmod


def load_mesh(
    path: Union[str, Path], mesh_type: MeshType = "polydata", return_contract: bool = True
) -> Union[MeshContract, vtk.vtkPolyData, vtk.vtkUnstructuredGrid]:
    """
    Load mesh from file.

    Supports: .vtk (polydata/ugrid), .stl

    Args:
        path: Path to mesh file
        mesh_type: Type of mesh ('polydata', 'ugrid', 'stl')
        return_contract: If True, return MeshContract; if False, return VTK object

    Returns:
        MeshContract or VTK object depending on return_contract flag

    Raises:
        ValueError: If mesh_type is invalid
        FileNotFoundError: If file doesn't exist

    Example:
        >>> contract = load_mesh("mesh.vtk", mesh_type="polydata")
        >>> polydata = contract.get_polydata()
    """
    path = Path(path)
    logger.info(f"Loading mesh from {path} (type: {mesh_type})")

    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    # Select appropriate reader
    if mesh_type == "ugrid":
        reader = vtk.vtkUnstructuredGridReader()
    elif mesh_type == "polydata":
        reader = vtk.vtkPolyDataReader()
    elif mesh_type == "stl":
        reader = vtk.vtkSTLReader()
    else:
        raise ValueError(f"Invalid mesh type: {mesh_type}. Must be 'polydata', 'ugrid', or 'stl'")

    # Read mesh
    reader.SetFileName(str(path))
    reader.Update()
    output = reader.GetOutput()

    # Check for errors
    if reader.GetErrorCode() != vtk.vtkErrorCode.NoError:
        raise IOError(f"Error reading VTK file: {path}")

    if return_contract:
        # Create appropriate contract based on mesh type
        if mesh_type == "ugrid":
            return MeshContract.from_ugrid(output, path=path)
        else:
            return MeshContract.from_polydata(output, path=path)
    else:
        return output


def save_mesh(
    contract: Union[MeshContract, vtk.vtkPolyData, vtk.vtkUnstructuredGrid],
    output_path: Union[str, Path],
    mesh_type: MeshType = "polydata",
    file_format: Literal["ascii", "binary"] = "ascii",
    vtk_version: int = 42,
    overwrite: bool = False,
) -> None:
    """
    Save mesh to disk.

    Args:
        contract: MeshContract or VTK object to save
        output_path: Path where mesh will be saved
        mesh_type: Type of mesh output ('polydata', 'ugrid', 'stl')
        file_format: File format ('ascii' or 'binary')
        vtk_version: VTK file version (42 for legacy compatibility)
        overwrite: If True, overwrite existing file

    Raises:
        FileExistsError: If file exists and overwrite=False
        ValueError: If mesh_type is invalid

    Example:
        >>> save_mesh(contract, "output.vtk", mesh_type="polydata")
    """
    output_path = Path(output_path)
    logger.info(f"Saving mesh to {output_path} (type: {mesh_type})")

    # Ensure .vtk extension
    if not str(output_path).endswith(".vtk"):
        output_path = output_path.with_suffix(".vtk")

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {output_path}. Use overwrite=True to replace.")

    # Get VTK object from contract or use directly
    if isinstance(contract, MeshContract):
        if mesh_type == "ugrid":
            mesh_data = contract.get_ugrid()
        else:
            mesh_data = contract.get_polydata()
    else:
        mesh_data = contract

    # Select appropriate writer
    if mesh_type == "ugrid":
        writer = vtk.vtkUnstructuredGridWriter()
    elif mesh_type == "stl":
        writer = vtk.vtkSTLWriter()
    else:  # polydata
        writer = vtk.vtkPolyDataWriter()

    # Configure writer
    writer.WriteArrayMetaDataOff()
    writer.SetInputData(mesh_data)
    writer.SetFileName(str(output_path))

    if file_format == "ascii":
        writer.SetFileTypeToASCII()
    else:
        writer.SetFileTypeToBinary()

    # Set VTK version for compatibility (VTK 9.1+)
    if hasattr(writer, "SetFileVersion"):
        writer.SetFileVersion(vtk_version)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write mesh
    writer.Update()

    if not output_path.exists():
        raise IOError(f"Failed to save mesh: {output_path}")


def clean_stl_file(input_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    """
    Clean STL file by removing content after endsolid marker.

    Some STL files have garbage data after the endsolid tag that causes parsing errors.

    Args:
        input_path: Path to input STL file
        output_path: Path to cleaned output file

    Example:
        >>> clean_stl_file("dirty.stl", "clean.stl")
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    logger.info(f"Cleaning STL file: {input_path} -> {output_path}")

    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            f_out.write(line)
            if line.strip().startswith("endsolid"):
                break  # Stop after endsolid marker


# ---------------------------------------------------------------------------
# Master vtktools file-I/O functions — ported verbatim from common/vtktools.py
# ---------------------------------------------------------------------------

_DATA_TYPES = ["polydata", "ugrid", "stl"]
_EXPORT_DATA_TYPES = ["vtp", "vtk", "ply", "stl", "obj", "ugrid"]


def read_vtk(fname, input_type="polydata"):
    """
    Read VTK file
    """
    if input_type not in _DATA_TYPES:
        logger.error(f"Invalid input type: {input_type}")
        raise ValueError(f"Invalid input type: {input_type}")

    try:
        if input_type == "ugrid":
            reader = vtk.vtkUnstructuredGridReader()
        elif input_type == "polydata":
            reader = vtk.vtkPolyDataReader()
        else:  # stl
            reader = vtk.vtkSTLReader()

        logger.info(f"Reading VTK [{input_type}] file: {fname}")
        reader.SetFileName(fname)
        reader.Update()
        output = reader.GetOutput()

        if reader.GetErrorCode() != vtk.vtkErrorCode.NoError:
            raise ValueError(f"Error reading VTK file: {fname}")

        return output
    except Exception as e:
        logger.error(f"Failed to read VTK file: {fname}. Error: {e}")
        raise


def readVtk(fname, input_type="polydata"):
    logger.warning("This function is deprecated. Please use read_vtk instead.")
    return read_vtk(fname, input_type)


def write_vtk(mesh, directory, outname="output", output_type="polydata"):
    filename = os.path.join(directory, outname)
    filename += ".vtk" if not filename.endswith(".vtk") else ""

    if output_type not in _DATA_TYPES:
        logger.error(f"Invalid output type: {output_type}")
        raise ValueError(f"Invalid output type: {output_type}")

    if output_type == "ugrid":
        writer = vtk.vtkUnstructuredGridWriter()
    else:
        writer = vtk.vtkPolyDataWriter()
    writer.WriteArrayMetaDataOff()
    writer.SetInputData(mesh)
    writer.SetFileName(filename)
    writer.SetFileTypeToASCII()
    # check for vtk version
    if vtk.vtkVersion().GetVTKMajorVersion() >= 9 and vtk.vtkVersion().GetVTKMinorVersion() >= 1:
        writer.SetFileVersion(42)
    writer.Update()


def writeVtk(mesh, directory, outname="output", output_type="polydata"):
    logger.warning("This function is deprecated. Please use write_vtk instead.")
    return write_vtk(mesh, directory, outname, output_type)


def export_as(input_mesh, output_file: str, export_as="ply") -> None:
    """
    Export a vtkPolyData object to a file
    """

    if export_as not in _EXPORT_DATA_TYPES:
        raise ValueError(f"Invalid export type {export_as}. Choose from: {_EXPORT_DATA_TYPES}")

    if export_as == "ply":
        writer = vtk.vtkPLYWriter()
    elif export_as == "stl":
        writer = vtk.vtkSTLWriter()
    elif export_as == "obj":
        writer = vtk.vtkOBJWriter()
    elif export_as == "vtp":
        writer = vtk.vtkXMLPolyDataWriter()
    elif export_as == "vtk" or export_as == "ugrid":
        export_as = "polydata" if export_as == "vtk" else "ugrid"
        write_vtk(
            input_mesh,
            os.path.dirname(output_file),
            os.path.basename(output_file),
            output_type=export_as,
        )
        return

    writer.SetFileName(output_file)
    writer.SetInputData(input_mesh)
    writer.Write()


def saveCarpAsVtk(pts, el, dir, name, dat=None):
    nodes = vtk.vtkPoints()
    for ix in range(len(pts)):
        nodes.InsertPoint(ix, pts[ix, 0], pts[ix, 1], pts[ix, 2])

    elems = vtk.vtkCellArray()
    for ix in range(len(el)):
        elIdList = vtk.vtkIdList()
        elIdList.InsertNextId(el[ix][0])
        elIdList.InsertNextId(el[ix][1])
        elIdList.InsertNextId(el[ix][2])
        elems.InsertNextCell(elIdList)

    pd = vtk.vtkPolyData()
    pd.SetPoints(nodes)
    pd.SetPolys(elems)
    if dat is not None:
        pd.GetPointData().SetScalars(vtknp.numpy_to_vtk(dat))
        p2c = vtk.vtkPointDataToCellData()
        p2c.SetInputData(pd)
        p2c.Update()
        outpd = p2c.GetOutput()
    else:
        outpd = pd

    writeVtk(outpd, dir, name)


def vtk_from_points_file(file_path: str, mydelim=","):
    """
    Creates a vtkPolyData object from a points file
    """
    points_read = np.loadtxt(file_path, delimiter=mydelim)
    points = vtk.vtkPoints()

    for pt in points_read:
        points.InsertNextPoint(pt[0], pt[1], pt[2])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    return polydata
