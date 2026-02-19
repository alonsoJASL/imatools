"""
Image data contracts.

Defines structured interfaces for medical image data (NIfTI, NRRD, DICOM).
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import SimpleITK as sitk


@dataclass
class ImageMetadata:
    """
    Metadata for medical images.
    
    Attributes:
        origin: Physical coordinates of the image origin (x, y, z)
        spacing: Voxel dimensions in physical units (x, y, z)
        direction: 3x3 direction cosine matrix (flattened to 9 elements)
        size: Image dimensions in voxels (x, y, z)
    """
    origin: Tuple[float, float, float]
    spacing: Tuple[float, float, float]
    direction: Tuple[float, ...] = field(default_factory=lambda: (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    size: Tuple[int, int, int] = (0, 0, 0)
    
    @classmethod
    def from_sitk_image(cls, image: sitk.Image) -> "ImageMetadata":
        """Extract metadata from a SimpleITK image."""
        return cls(
            origin=image.GetOrigin(),
            spacing=image.GetSpacing(),
            direction=image.GetDirection(),
            size=image.GetSize(),
        )
    
    def to_sitk_metadata(self, image: sitk.Image) -> sitk.Image:
        """Apply this metadata to a SimpleITK image (modifies in place)."""
        image.SetOrigin(self.origin)
        image.SetSpacing(self.spacing)
        image.SetDirection(self.direction)
        return image


@dataclass
class ImageContract:
    """
    Contract for image data exchange between layers.
    
    This contract serves as the boundary between:
    - I/O layer (loads from disk, creates contract)
    - Core logic (operates on contract data)
    - Engines (orchestrate operations via contracts)
    
    Attributes:
        path: Source file path (optional, for tracking provenance)
        metadata: Physical image metadata
        data: Optional numpy array (lazy-loaded when needed)
        sitk_image: Optional SimpleITK image object (for compatibility)
    """
    metadata: ImageMetadata
    path: Optional[Path] = None
    data: Optional[np.ndarray] = None
    sitk_image: Optional[sitk.Image] = None
    
    @classmethod
    def from_sitk_image(cls, image: sitk.Image, path: Optional[Path] = None) -> "ImageContract":
        """Create contract from a SimpleITK image."""
        return cls(
            path=path,
            metadata=ImageMetadata.from_sitk_image(image),
            sitk_image=image,
            data=None,  # Lazy load on demand
        )
    
    @classmethod
    def from_path(cls, path: Path) -> "ImageContract":
        """
        Create contract with path only (data loaded by I/O layer).
        Used for deferred loading.
        """
        return cls(
            path=path,
            metadata=ImageMetadata(origin=(0, 0, 0), spacing=(1, 1, 1)),
            data=None,
            sitk_image=None,
        )
    
    def get_array(self) -> np.ndarray:
        """
        Get image data as numpy array.
        Lazy-loads from sitk_image if needed.
        """
        if self.data is not None:
            return self.data
        
        if self.sitk_image is not None:
            self.data = sitk.GetArrayFromImage(self.sitk_image)
            return self.data
        
        raise ValueError("No image data available. Load image first.")
    
    def get_sitk_image(self) -> sitk.Image:
        """
        Get SimpleITK image object.
        Creates from data if needed.
        """
        if self.sitk_image is not None:
            return self.sitk_image
        
        if self.data is not None:
            image = sitk.GetImageFromArray(self.data)
            self.metadata.to_sitk_metadata(image)
            self.sitk_image = image
            return image
        
        raise ValueError("No image data available. Load image first.")
    
    def validate(self) -> bool:
        """Validate that contract has minimum required data."""
        has_data = self.data is not None or self.sitk_image is not None
        has_metadata = self.metadata is not None
        return has_data and has_metadata