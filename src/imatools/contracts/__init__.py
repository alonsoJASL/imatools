"""
Data contracts for imatools.

Contracts define explicit interfaces between layers:
- Orchestrators pass contracts to engines
- Engines pass contracts to core logic
- I/O layer converts files to/from contracts

This enforces the separation of concerns defined in the Developer's Manifest.
"""

from imatools.contracts.image import ImageContract, ImageMetadata
from imatools.contracts.mesh import MeshContract, MeshMetadata, MeshType
from imatools.contracts.operations import (
    LabelOperation,
    MorphOperation,
    TransformOperation,
)
from imatools.contracts.report import MeshReportInputs, RenderParams

__all__ = [
    "ImageContract",
    "ImageMetadata",
    "MeshContract",
    "MeshMetadata",
    "MeshType",
    "LabelOperation",
    "MorphOperation",
    "TransformOperation",
    "MeshReportInputs",
    "RenderParams",
]
