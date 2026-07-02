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


# ---------------------------------------------------------------------------
# Moved from imatools.common.vtktools (M2a-2; zero-caller-but-KEEP functions)
# ---------------------------------------------------------------------------


def convert_5_to_4(imsh, omsh):
    """
    Input: msh paths
        imsh (input)
        omsh (output)
    """
    # Read the VTK legacy file in version 5 format
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(imsh)
    reader.Update()
    print(reader.GetFileVersion())

    # Convert the file to VTK legacy format
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(reader.GetOutput())
    writer.SetFileName(omsh)
    writer.SetFileTypeToASCII()
    # writer.SetHeader("vtk version 4.0")
    writer.SetFileVersion(42)
    writer.Update()


def convertToCarto(vtkpoly_path: str, cell_scalar_field: str, output_file: str) -> None:
    """
    Convert a vtkPolyData object to a Carto object

    Adapted (M2a-2): originally called ``normalise_vtk_values`` (DELETE-
    category, removed from ``common.vtktools`` elsewhere in M2) — its 3-line
    min-max body is inlined below instead of being preserved as a shared
    utility (Jose's call, see MIGRATION_M2.md).
    """
    from imatools.core.mesh import (  # noqa: PLC0415
        convertPointDataToNpArray,
        np_to_vtk_array,
        set_cell_to_point_data,
    )

    try:
        vtkpoly = read_vtk(vtkpoly_path)
        working_msh = set_cell_to_point_data(vtkpoly, cell_scalar_field)

        # normalise_vtk_values inlined (that shared helper was dropped — trivial, not worth
        # preserving as a named utility elsewhere; M2 disposition)
        array = convertPointDataToNpArray(working_msh, cell_scalar_field)
        array = (array - np.min(array)) / (np.max(array) - np.min(array))
        scalars = np_to_vtk_array(array, cell_scalar_field)
        norm_working_msh = vtk.vtkPolyData()
        norm_working_msh.DeepCopy(working_msh)
        norm_working_msh.GetPointData().SetScalars(scalars)

        # save
        odir = os.path.dirname(output_file)

        write_vtk(norm_working_msh, odir, f"normalised_{cell_scalar_field}.vtk")

        ## change lookup table for norm_working_msh

        lut = vtk.vtkColorTransferFunction()
        lut.SetColorSpaceToRGB()
        lut.AddRGBPoint(0.0, 0.04, 0.21, 0.25)
        lut.AddRGBPoint(0.5, 0.94, 0.47, 0.12)
        lut.AddRGBPoint(1.0, 0.90, 0.11, 0.14)
        lut.SetScaleToLinear()

    except Exception as e:
        print(f"Error: {e}")
        return

    with open(output_file, "w") as cartoFile:
        # Header
        cartoFile.write("# vtk DataFile Version 3.0\n")
        cartoFile.write("PatientData Anon Anon 00000000\n")
        cartoFile.write("ASCII\n")
        cartoFile.write("DATASET POLYDATA\n")

        # Points
        cartoFile.write(f"POINTS\t{working_msh.GetNumberOfPoints()} float\n")
        points = working_msh.GetPoints()
        for ix in range(working_msh.GetNumberOfPoints()):
            pt = points.GetPoint(ix)
            cartoFile.write(f"{pt[0]} {pt[1]} {pt[2]}\n")

        cartoFile.write("\n")

        # Cells
        cartoFile.write(
            f"POLYGONS\t{working_msh.GetNumberOfCells()}\t{working_msh.GetNumberOfCells()*4}\n"
        )
        for ix in range(working_msh.GetNumberOfCells()):
            cell = working_msh.GetCell(ix)
            cell_type = cell.GetCellType()
            num_points = cell.GetNumberOfPoints()
            cartoFile.write(f"{num_points}\n")
            for jx in range(num_points):
                cartoFile.write(f"{cell.GetPointId(jx)}\n")

        cartoFile.write("\n")

        # Scalars
        cartoFile.write(f"POINT_DATA\tSCALARS {cell_scalar_field} float\n")
        cartoFile.write("LOOKUP_TABLE lookup_table\n")

        scalars = working_msh.GetPointData().GetScalars()
        max_scalar = np.max(scalars)
        min_scalar = np.min(scalars)

        for kx in range(working_msh.GetNumberOfPoints()):
            value = scalars.GetTuple1(kx)
            normalized_value = (value - min_scalar) / (max_scalar - min_scalar)
            # set precision to 2 decimal places
            cartoFile.write(f"{normalized_value:.2f}\n")

        cartoFile.write("\n")

        # LUT
        numCols = 256
        cartoFile.write(f"LOOKUP_TABLE lookup_table {numCols}\n")
        lut = vtk.vtkColorTransferFunction()
        lut.SetColorSpaceToRGB()
        lut.AddRGBPoint(0.0, 0.04, 0.21, 0.25)
        lut.AddRGBPoint((numCols - 1.0) / 2.0, 0.94, 0.47, 0.12)
        lut.AddRGBPoint((numCols - 1.0), 0.90, 0.11, 0.14)
        lut.SetScaleToLinear()
        for i in range(numCols):
            color = lut.GetColor(i)
            cartoFile.write(f"{color[0]} {color[1]} {color[2]} 1.0\n")
