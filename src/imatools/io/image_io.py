"""
Image file I/O operations.

Handles loading and saving of medical images in various formats.
All functions return/accept ImageContract objects.
"""

import os
from pathlib import Path
from typing import Union, Optional, Tuple
import logging

import SimpleITK as sitk
import nrrd
import numpy as np

from imatools.contracts import ImageContract, ImageMetadata

logger = logging.getLogger(__name__)


def load_image(
    path: Union[str, Path],
    return_contract: bool = True
) -> Union[ImageContract, sitk.Image]:
    """
    Load medical image from file.
    
    Supports: .nii, .nii.gz, .nrrd, .mha, .mhd
    
    Args:
        path: Path to image file
        return_contract: If True, return ImageContract; if False, return sitk.Image
    
    Returns:
        ImageContract or sitk.Image depending on return_contract flag
    
    Example:
        >>> contract = load_image("image.nii.gz")
        >>> sitk_img = contract.get_sitk_image()
    """
    path = Path(path)
    logger.info(f"Loading image from {path}")
    
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    # Use SimpleITK to load (handles most formats)
    sitk_image = sitk.ReadImage(str(path))
    
    if return_contract:
        return ImageContract.from_sitk_image(sitk_image, path=path)
    else:
        return sitk_image


def save_image(
    contract: Union[ImageContract, sitk.Image],
    output_path: Union[str, Path],
    overwrite: bool = False
) -> None:
    """
    Save image to disk.
    
    Args:
        contract: ImageContract or sitk.Image to save
        output_path: Path where image will be saved
        overwrite: If True, overwrite existing file
    
    Raises:
        FileExistsError: If file exists and overwrite=False
    
    Example:
        >>> save_image(contract, "output.nii.gz")
    """
    output_path = Path(output_path)
    logger.info(f"Saving image to {output_path}")
    
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {output_path}. Use overwrite=True to replace."
        )
    
    # Get SimpleITK image from contract or use directly
    if isinstance(contract, ImageContract):
        sitk_image = contract.get_sitk_image()
    else:
        sitk_image = contract
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write image
    sitk.WriteImage(sitk_image, str(output_path))
    
    if not output_path.exists():
        raise IOError(f"Failed to save image: {output_path}")


def load_nrrd(
    path: Union[str, Path],
    return_contract: bool = True
) -> Union[ImageContract, Tuple[np.ndarray, dict]]:
    """
    Load NRRD file with full header access.
    
    Args:
        path: Path to .nrrd file
        return_contract: If True, return ImageContract; if False, return (data, header)
    
    Returns:
        ImageContract or (numpy array, header dict)
    
    Example:
        >>> contract = load_nrrd("image.nrrd")
        >>> # For header manipulation:
        >>> data, header = load_nrrd("image.nrrd", return_contract=False)
    """
    path = Path(path)
    logger.info(f"Loading NRRD from {path}")
    
    if not path.exists():
        raise FileNotFoundError(f"NRRD file not found: {path}")
    
    data, header = nrrd.read(str(path))
    
    if return_contract:
        # Convert to SimpleITK image for contract
        sitk_image = sitk.GetImageFromArray(data)
        sitk_image.SetOrigin(header.get('space origin', (0, 0, 0)))
        sitk_image.SetSpacing(header.get('spacings', (1, 1, 1)))
        
        return ImageContract.from_sitk_image(sitk_image, path=path)
    else:
        return data, header


def save_nrrd(
    data: np.ndarray,
    header: dict,
    output_path: Union[str, Path],
    overwrite: bool = False
) -> None:
    """
    Save NRRD file with custom header.
    
    Args:
        data: Numpy array to save
        header: NRRD header dictionary
        output_path: Path where file will be saved
        overwrite: If True, overwrite existing file
    
    Example:
        >>> data, header = load_nrrd("input.nrrd", return_contract=False)
        >>> # Modify header
        >>> header['space origin'] = (0, 0, 0)
        >>> save_nrrd(data, header, "output.nrrd")
    """
    output_path = Path(output_path)
    logger.info(f"Saving NRRD to {output_path}")
    
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {output_path}. Use overwrite=True to replace."
        )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nrrd.write(str(output_path), data, header)


def get_nrrd_header(path: Union[str, Path]) -> dict:
    """
    Read NRRD header without loading full image data.
    
    Args:
        path: Path to .nrrd file
    
    Returns:
        NRRD header dictionary
    
    Example:
        >>> header = get_nrrd_header("image.nrrd")
        >>> print(header['space origin'])
    """
    path = Path(path)
    logger.info(f"Reading NRRD header from {path}")
    
    if not path.exists():
        raise FileNotFoundError(f"NRRD file not found: {path}")
    
    _, header = nrrd.read(str(path))
    return header


def fix_nrrd_header_and_save(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    fix_function: callable
) -> None:
    """
    Load NRRD, apply header fix function, and save.
    
    Args:
        input_path: Path to input .nrrd file
        output_path: Path to output .nrrd file
        fix_function: Function that takes header dict and returns fixed header dict
    
    Example:
        >>> def align_header(hdr):
        >>>     # Fix logic here
        >>>     return hdr
        >>> fix_nrrd_header_and_save("in.nrrd", "out.nrrd", align_header)
    """
    data, header = load_nrrd(input_path, return_contract=False)
    fixed_header = fix_function(header)
    save_nrrd(data, fixed_header, output_path, overwrite=True)