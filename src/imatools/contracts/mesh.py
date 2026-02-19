"""
Mesh data contracts.

Defines structured interfaces for 3D mesh data (VTK polydata, unstructured grids).
"""

from dataclasses import dataclass
from typing import Optional, Literal
from pathlib import Path
import numpy as np
import vtk


MeshType = Literal["polydata", "ugrid", "stl"]


@dataclass
class MeshMetadata:
    """
    Metadata for mesh data.
    
    Attributes:
        num_points: Number of vertices in mesh
        num_cells: Number of cells (triangles, tetrahedra, etc.)
        bounds: (min_x, max_x, min_y, max_y, min_z, max_z)
        mesh_type: Type of mesh (polydata, ugrid, stl)
    """
    num_points: int = 0
    num_cells: int = 0
    bounds: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mesh_type: MeshType = "polydata"
    
    @classmethod
    def from_polydata(cls, polydata: vtk.vtkPolyData) -> "MeshMetadata":
        """Extract metadata from VTK polydata."""
        return cls(
            num_points=polydata.GetNumberOfPoints(),
            num_cells=polydata.GetNumberOfCells(),
            bounds=polydata.GetBounds(),
            mesh_type="polydata",
        )
    
    @classmethod
    def from_ugrid(cls, ugrid: vtk.vtkUnstructuredGrid) -> "MeshMetadata":
        """Extract metadata from VTK unstructured grid."""
        return cls(
            num_points=ugrid.GetNumberOfPoints(),
            num_cells=ugrid.GetNumberOfCells(),
            bounds=ugrid.GetBounds(),
            mesh_type="ugrid",
        )


@dataclass
class MeshContract:
    """
    Contract for mesh data exchange between layers.
    
    Attributes:
        path: Source file path (optional)
        metadata: Mesh metadata
        polydata: Optional VTK polydata object
        ugrid: Optional VTK unstructured grid object
        points: Optional numpy array of vertices (N, 3)
        elements: Optional numpy array of element connectivity
    """
    metadata: MeshMetadata
    path: Optional[Path] = None
    polydata: Optional[vtk.vtkPolyData] = None
    ugrid: Optional[vtk.vtkUnstructuredGrid] = None
    points: Optional[np.ndarray] = None
    elements: Optional[np.ndarray] = None
    
    @classmethod
    def from_polydata(cls, polydata: vtk.vtkPolyData, path: Optional[Path] = None) -> "MeshContract":
        """Create contract from VTK polydata."""
        return cls(
            path=path,
            metadata=MeshMetadata.from_polydata(polydata),
            polydata=polydata,
        )
    
    @classmethod
    def from_ugrid(cls, ugrid: vtk.vtkUnstructuredGrid, path: Optional[Path] = None) -> "MeshContract":
        """Create contract from VTK unstructured grid."""
        return cls(
            path=path,
            metadata=MeshMetadata.from_ugrid(ugrid),
            ugrid=ugrid,
        )
    
    @classmethod
    def from_path(cls, path: Path, mesh_type: MeshType = "polydata") -> "MeshContract":
        """
        Create contract with path only (data loaded by I/O layer).
        """
        return cls(
            path=path,
            metadata=MeshMetadata(mesh_type=mesh_type),
        )
    
    def get_polydata(self) -> vtk.vtkPolyData:
        """Get VTK polydata object."""
        if self.polydata is not None:
            return self.polydata
        raise ValueError("No polydata available. Load mesh first.")
    
    def get_ugrid(self) -> vtk.vtkUnstructuredGrid:
        """Get VTK unstructured grid object."""
        if self.ugrid is not None:
            return self.ugrid
        raise ValueError("No unstructured grid available. Load mesh first.")
    
    def validate(self) -> bool:
        """Validate that contract has minimum required data."""
        has_vtk_data = self.polydata is not None or self.ugrid is not None
        has_array_data = self.points is not None
        has_metadata = self.metadata is not None
        
        return (has_vtk_data or has_array_data) and has_metadata