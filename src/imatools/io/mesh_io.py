"""
Mesh file I/O operations.

Handles loading and saving of 3D meshes in various formats.
All functions return/accept MeshContract objects.
"""

import os
from pathlib import Path
from typing import Union, Literal
import logging

import vtk

from imatools.contracts import MeshContract, MeshMetadata, MeshType

logger = logging.getLogger(__name__)


def load_mesh(
    path: Union[str, Path],
    mesh_type: MeshType = "polydata",
    return_contract: bool = True
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
    overwrite: bool = False
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
        raise FileExistsError(
            f"File already exists: {output_path}. Use overwrite=True to replace."
        )
    
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
    if hasattr(writer, 'SetFileVersion'):
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