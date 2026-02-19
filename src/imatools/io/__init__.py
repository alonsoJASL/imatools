"""
I/O layer for imatools.

Handles all file system operations for images and meshes.
Functions in this layer:
- Accept file paths as input
- Return contracts as output
- Handle format-specific details (NRRD, NIfTI, DICOM, VTK, STL)

Core logic should NEVER import from this module.
"""

from .image_io import (
    load_image,
    save_image,
    load_nrrd,
    save_nrrd,
    get_nrrd_header,
)

from .mesh_io import (
    load_mesh,
    save_mesh,
)

__all__ = [
    # Image I/O
    "load_image",
    "save_image",
    "load_nrrd",
    "save_nrrd",
    "get_nrrd_header",
    # Mesh I/O
    "load_mesh",
    "save_mesh",
]