"""
Operation contracts.

Defines structured interfaces for image/mesh processing operations.
"""

from dataclasses import dataclass
from typing import Literal, Any, Optional
from enum import Enum


class MorphOperationType(str, Enum):
    """Morphological operation types."""
    DILATE = "dilate"
    ERODE = "erode"
    OPEN = "open"
    CLOSE = "close"
    FILL = "fill"
    SMOOTH = "smooth"


class KernelType(str, Enum):
    """Morphological kernel types."""
    BALL = "ball"
    BOX = "box"
    CROSS = "cross"


@dataclass
class MorphOperation:
    """
    Contract for morphological operations.
    
    Attributes:
        operation: Type of morphological operation
        kernel: Structuring element shape
        radius: Kernel radius in voxels
        iterations: Number of times to apply operation
    """
    operation: MorphOperationType
    kernel: KernelType = KernelType.BALL
    radius: int = 3
    iterations: int = 1
    
    def validate(self) -> bool:
        """Validate operation parameters."""
        if self.radius <= 0:
            raise ValueError(f"Radius must be positive, got {self.radius}")
        if self.iterations <= 0:
            raise ValueError(f"Iterations must be positive, got {self.iterations}")
        return True


@dataclass
class LabelOperation:
    """
    Contract for label manipulation operations.
    
    Attributes:
        operation: Type of label operation (extract, merge, remove, etc.)
        labels: Label values to operate on
        binarise: Whether to binarise output (1/0 instead of original labels)
        background_value: Value to use for background
        output_label: Label value for result (if applicable)
    """
    operation: Literal["extract", "merge", "remove", "relabel", "exchange"]
    labels: list[int]
    binarise: bool = False
    background_value: int = 0
    output_label: Optional[int] = None
    
    def validate(self) -> bool:
        """Validate operation parameters."""
        if not self.labels:
            raise ValueError("Labels list cannot be empty")
        if any(label < 0 for label in self.labels):
            raise ValueError("Label values must be non-negative")
        return True


@dataclass
class TransformOperation:
    """
    Contract for spatial transformations.
    
    Attributes:
        operation: Type of transform (resample, flip, rotate, etc.)
        reference_metadata: Target image metadata for resampling
        interpolation: Interpolation method
        parameters: Additional operation-specific parameters
    """
    operation: Literal["resample", "flip", "rotate", "translate", "crop"]
    interpolation: Literal["nearest", "linear", "bspline"] = "linear"
    parameters: dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
    
    def validate(self) -> bool:
        """Validate operation parameters."""
        if self.operation not in ["resample", "flip", "rotate", "translate", "crop"]:
            raise ValueError(f"Unknown operation: {self.operation}")
        return True