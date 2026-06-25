"""
Image file I/O operations.

Handles loading and saving of medical images in various formats.
Dev's existing functions (load_image, save_image, get_nrrd_header, load_nrrd,
save_nrrd, fix_nrrd_header_and_save) use ImageContract and are the canonical
dev implementations.

Master itktools file-I/O functions added by T2a4 (7 functions):
    load_image_as_np, load_nrrd_base, load_nrrd_image, convert_to_inr,
    convert_from_inr, pointfile_to_image, fix_header_and_save.
These are verbatim from master except for the Cat-A fix in load_nrrd_image
(see that function's docstring).  ``imatools.common.itktools`` re-exports
all 7 via a bottom shim.
"""

import json
import logging
from pathlib import Path
from typing import Tuple, Union

import nrrd
import numpy as np
import SimpleITK as sitk  # noqa: N813

from imatools.contracts import ImageContract

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy-helper accessor — avoids circular import at module load time.
# After itktools finishes loading (including its bottom shims for core.*),
# all helper names (imarray, imview, points_to_image) are available.
# ---------------------------------------------------------------------------


def _itk():
    """Return the itktools module (always already loaded when an io fn is called)."""
    import imatools.common.itktools as _m  # noqa: PLC0415

    return _m


def _spatial():
    """Return the core.spatial module for fix_header_to_axis_aligned."""
    import imatools.core.spatial as _s  # noqa: PLC0415

    return _s


def load_image(
    path: Union[str, Path], return_contract: bool = True
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
    overwrite: bool = False,
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
        raise FileExistsError(f"File already exists: {output_path}. Use overwrite=True to replace.")

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
    path: Union[str, Path], return_contract: bool = True
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
        sitk_image.SetOrigin(header.get("space origin", (0, 0, 0)))
        sitk_image.SetSpacing(header.get("spacings", (1, 1, 1)))

        return ImageContract.from_sitk_image(sitk_image, path=path)
    else:
        return data, header


def save_nrrd(
    data: np.ndarray, header: dict, output_path: Union[str, Path], overwrite: bool = False
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
        raise FileExistsError(f"File already exists: {output_path}. Use overwrite=True to replace.")

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
    input_path: Union[str, Path], output_path: Union[str, Path], fix_function: callable
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


# ---------------------------------------------------------------------------
# Master itktools file-I/O functions (added by T2a4, verbatim from master
# except load_nrrd_image which has a Cat-A SetSpacing fix).
# imatools.common.itktools re-exports all 7 via a bottom shim.
# ---------------------------------------------------------------------------


def fix_header_and_save(path_to_file, out_path):
    """
    Reads a NRRD file, modifies its header to make space directions axis-aligned,
    and saves the modified header back to a new NRRD file.

    Authoritative implementation (T2a4 migration from ``imatools.common.itktools``).
    ``imatools.common.itktools.fix_header_and_save`` is a re-export shim for this.
    """
    logger.info(f"Fixing header for {path_to_file} and saving to {out_path}")
    data, hdr = nrrd.read(path_to_file)

    # Fix the header — delegates to core.spatial (migrated in T2a3)
    fixed_header = _spatial().fix_header_to_axis_aligned(hdr)

    # Save the modified data and header
    nrrd.write(out_path, data, fixed_header)
    logger.info(f"Saved fixed NRRD file to {out_path}")


def load_image_as_np(path_to_file):
    """Reads image into numpy array.

    Returns:
        tuple: (array, origin, im_size)
    """
    sitk_t1 = sitk.ReadImage(path_to_file)

    t1 = _itk().imarray(sitk_t1)
    origin = sitk_t1.GetOrigin()
    im_size = sitk_t1.GetSize()

    return t1, origin, im_size


def load_nrrd_base(path_to_file):
    """
    Loads a NRRD file and returns the image data and header.

    Returns:
        tuple: (data ndarray, header dict)
    """
    logger.info(f"Loading NRRD file from {path_to_file}")
    data, header = nrrd.read(path_to_file)

    return data, header


def load_nrrd_image(path_to_file):
    """
    Loads a NRRD file and returns the image as a SimpleITK object.

    Cat-A fix (T2a4): master's implementation passed the ``'space directions'``
    value (a (3,3) ndarray written by SimpleITK) directly to
    ``sitk.Image.SetSpacing``, which requires a flat sequence of scalars.  This
    caused a ``TypeError`` crash on any NRRD written by SimpleITK.

    Fix: compute per-axis voxel spacings as the L2 norm of each direction-matrix
    row::

        dirs     = np.asarray(header["space directions"], dtype=float)
        spacings = np.linalg.norm(dirs, axis=1)

    and pass ``[float(s) for s in spacings]`` to ``SetSpacing``.  A
    ``(1.0, 1.0, 1.0)`` fallback is used when the header key is absent.

    NOTE: this fix replaces a crash — master output is unreachable so there is
    NO golden for this function.  The test (``test_load_nrrd_image_intent_stub``)
    is a structural intent-stub that asserts the call succeeds and returns a
    ``sitk.Image`` of the expected size.
    """
    logger.info(f"Loading NRRD image from {path_to_file}")
    data, header = load_nrrd_base(path_to_file)

    # Convert the numpy array to a SimpleITK image
    sitk_image = sitk.GetImageFromArray(data)

    # Set the image properties from the header
    sitk_image.SetOrigin(header.get("space origin", (0, 0, 0)))

    # Cat-A fix: 'space directions' is a (3,3) ndarray; compute per-axis spacings
    # from the row norms instead of passing the matrix directly to SetSpacing.
    if "space directions" in header:
        dirs = np.asarray(header["space directions"], dtype=float)
        spacings = np.linalg.norm(dirs, axis=1)
        sitk_image.SetSpacing([float(s) for s in spacings])
    else:
        sitk_image.SetSpacing((1.0, 1.0, 1.0))

    return sitk_image


def convert_to_inr(image, out_path):
    """
    Converts a SimpleITK image to an INR file.
    """
    print(f"Converting image to {out_path}")
    # Get the image data as a NumPy array
    data = _itk().imview(image)
    spacing = image.GetSpacing()
    # make sure elements less than 1 are 0
    dtype = data.dtype
    bitlen = data.dtype.itemsize * 8
    if dtype == bool or dtype == np.uint8:  # noqa: E721
        btype = "unsigned fixed"
    elif dtype == np.uint16:  # noqa: E721
        btype = "unsigned fixed"
    elif dtype == np.int16:  # noqa: E721
        btype = "signed fixed"
    elif dtype == np.float32:  # noqa: E721
        btype = "float"
    elif dtype == np.float64:  # noqa: E721
        btype = "float"
    else:
        raise ValueError("Volume format not supported")

    logger.info(f"Data type: {dtype}. TYPE:{btype} PIXSIZE:{bitlen}")
    xdim, ydim, zdim = data.shape
    header = f"#INRIMAGE-4#{{\nXDIM={xdim}\nYDIM={ydim}\nZDIM={zdim}\nVDIM=1\nVX={spacing[0]:.4f}\nVY={spacing[1]:.4f}\nVZ={spacing[2]:.4f}\n"
    header += "SCALE=2**0\n" if btype == "unsigned fixed" or btype == "signed fixed" else ""
    header += f"TYPE={btype}\nPIXSIZE={bitlen} bits\nCPU=decm"
    header += "\n" * (252 - len(header))  # Fill remaining space with newlines
    header += "##}\n"  # End of header

    # Write to binary file
    with open(out_path, "wb") as file:
        file.write(header.encode(encoding="utf-8"))  # Write header as bytes
        file.write(data.tobytes())  # Write data as bytes


def convert_from_inr(inr_path):
    """
    Converts an INR file to a SimpleITK image.
    """
    with open(inr_path, "rb") as file:
        header = ""
        while True:
            line = file.readline().decode("utf-8")
            header += line
            if line.strip() == "##}":
                break

        # Parse header
        header_dict = {}
        for line in header.split("\n"):
            if "=" in line:
                key, value = line.split("=")
                header_dict[key.strip()] = value.strip()

        xdim = int(header_dict["XDIM"])
        ydim = int(header_dict["YDIM"])
        zdim = int(header_dict["ZDIM"])
        spacing = [float(header_dict["VX"]), float(header_dict["VY"]), float(header_dict["VZ"])]
        pixsize = int(header_dict["PIXSIZE"].split()[0])
        dtype = header_dict["TYPE"]

        if dtype == "unsigned fixed":
            if pixsize == 8:
                np_dtype = np.uint8
            elif pixsize == 16:
                np_dtype = np.uint16
        elif dtype == "signed fixed":
            if pixsize == 16:
                np_dtype = np.int16
        elif dtype == "float":
            if pixsize == 32:
                np_dtype = np.float32
            elif pixsize == 64:
                np_dtype = np.float64
        else:
            raise ValueError("Volume format not supported")

        # Read image data
        data = np.frombuffer(file.read(), dtype=np_dtype)
        data = data.reshape((xdim, ydim, zdim), order="F")  # Fortran order to match INR format

        # Convert to SimpleITK image
        image = sitk.GetImageFromArray(data)
        image.SetSpacing(spacing)

        return image


def pointfile_to_image(path_to_image, path_to_points, label=1, girth=2, points_are_indices=False):
    """
    Set the closest voxels to the given points to the specified label in the input image.

    Args:
        path_to_image (str): Path to the input 3D image.
        path_to_points (str): Path to the text file containing the points.
        label (int): The label value to set for the closest voxels to the points.

    Returns:
        SimpleITK.Image: The modified image with the closest voxels set to the label.
    """
    # Load the image
    image = sitk.ReadImage(path_to_image)

    # Load the points
    points = []
    if path_to_points.endswith(".json"):
        with open(path_to_points, "r") as file:
            points_json = json.load(file)
        for _, coordinates in points_json.items():
            points.append(coordinates)

    else:
        with open(path_to_points, "r") as file:
            for line in file:
                # Split the line into a list of strings
                point = line.split()
                # Convert the strings to floats
                point = [float(x) for x in point]
                # Add the point to the list
                points.append(point)

    modified_image = _itk().points_to_image(image, points, label, girth, points_are_indices)

    return modified_image
