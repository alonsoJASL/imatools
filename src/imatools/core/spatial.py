# src/imatools/core/spatial.py
"""Plane / orientation functions migrated from ``imatools.common.itktools`` (T2a3).

The 5 public functions here are the authoritative implementations; the old
``imatools.common.itktools`` module re-exports them via a shim at its bottom.

Functions imported lazily at call time via ``_itk()`` to avoid circular imports:
``imview`` remains in itktools and is accessed as ``_itk().imview(...)``.

Bug notes
---------
- ``create_normal_vector_for_plane``: computes ``angle_rad = np.radians(angle)``
  but then uses the raw ``angle`` in cos/sin (Cat-B latent bug). Preserved VERBATIM
  — the 4 ``normal_vector`` golden tests lock this behaviour.
- ``create_image_at_plane_from_vector``: master's
  ``transform.SetMatrix(normal_vector + [0, 0, 1])`` passes a 3-element sequence
  where SimpleITK's ``SetMatrix`` requires 9 elements → always crashes with
  ``sitk::ERROR: Length of input (3) does not match matrix dimensions (3, 3)``.
  This is a Cat-A crash (master unreachable) so it is fixed here.  The fix uses a
  proper Rodrigues rotation construction (``_rotation_z_to_vector``) to build a
  valid 3×3 rotation matrix aligning the z-axis to the given normal vector, then
  flattens it to a 9-element list for ``SetMatrix``.  **This replacement is NOT
  validated against master output — master crashed, so no golden file exists.  The
  tests only assert the result is a 2-D ndarray (structural).**
"""

from __future__ import annotations

import nrrd
import numpy as np
import SimpleITK as sitk  # noqa: N813

from imatools.common.config import configure_logging

logger = configure_logging(log_name=__name__)


# ---------------------------------------------------------------------------
# Lazy-helper accessor — avoids circular import at module load time.
# After itktools finishes loading (including its bottom shim), all helper
# names are available in sys.modules and these lookups resolve instantly.
# ---------------------------------------------------------------------------
def _itk():
    """Return the itktools module (always already loaded when a spatial fn is called)."""
    import imatools.common.itktools as _m  # noqa: PLC0415

    return _m


# ---------------------------------------------------------------------------
# Private geometry helper (Cat-A fix for create_image_at_plane_from_vector)
# ---------------------------------------------------------------------------


def _rotation_z_to_vector(n):
    """Return a 3×3 rotation matrix that rotates [0,0,1] onto unit vector *n*.

    Uses the Rodrigues / cross-product construction.  When *n* is already
    aligned (or anti-aligned) with z a special-case identity (or 180-degree
    flip) is returned so that ``SetMatrix`` always receives 9 valid elements.

    NOTE: This helper implements the geometry fix for the Cat-A crash in
    ``create_image_at_plane_from_vector``.  It has NO corresponding master
    implementation — master always crashed before reaching this logic.
    """
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, n)
    s = float(np.linalg.norm(v))
    c = float(np.dot(z, n))
    if s < 1e-8:  # already aligned / anti-aligned
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


# ---------------------------------------------------------------------------
# Public functions — 5 migrated from itktools
# ---------------------------------------------------------------------------


def fix_header_to_axis_aligned(hdr: nrrd.NRRDHeader):
    """Modify NRRD header to make space directions axis-aligned."""
    hdr = hdr.copy()
    dirs = np.asarray(hdr["space directions"], dtype=float)

    # Compute voxel sizes (norms of direction vectors)
    spacings = np.linalg.norm(dirs, axis=1)
    if np.any(spacings <= 0):
        raise ValueError(f"Invalid spacing values: {spacings}")

    # Replace space directions with a diagonal matrix
    aligned_dirs = np.diag(spacings)
    hdr["space directions"] = aligned_dirs

    # Update srow_* fields for ITK/NIfTI compatibility
    origin = hdr["space origin"]
    hdr["srow_x"] = f"{aligned_dirs[0,0]:.6f} 0.000000 0.000000 {origin[0]:.6f}"
    hdr["srow_y"] = f"0.000000 {aligned_dirs[1,1]:.6f} 0.000000 {origin[1]:.6f}"
    hdr["srow_z"] = f"0.000000 0.000000 {aligned_dirs[2,2]:.6f} {origin[2]:.6f}"

    return hdr


def set_direction_as(im: sitk.Image, ref: sitk.Image):
    im.SetDirection(ref.GetDirection())
    return im


def create_normal_vector_for_plane(axis, angle):
    """
    Returns a normal vector for a plane rotated around the given axis by the given angle

    NOTE (Cat-B preserved bug): ``angle_rad = np.radians(angle)`` is computed
    but the raw ``angle`` value (not ``angle_rad``) is used in the cos/sin calls
    below.  This mismatch is intentional — the golden files lock this behaviour
    and it must NOT be corrected.
    """
    AXES = ["x", "y", "z"]  # noqa: N806
    if axis not in AXES:
        raise ValueError(f"Axis {axis} not recognised")

    vector = np.zeros(3)
    vector[AXES.index(axis)] = 1

    angle_rad = np.radians(angle)  # noqa: F841  (computed but unused — Cat-B preserved bug)

    if axis == "x":
        rotation_matrix = np.array(
            [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
        )
    elif axis == "y":
        rotation_matrix = np.array(
            [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        )
    elif axis == "z":
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )

    # Apply the rotation matrix to the initial vector
    normal_vector = np.dot(rotation_matrix, vector)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    return normal_vector


def create_image_at_plane(image: sitk.Image, point_on_plane: np.array, axis: str, angle: float):
    normal_vector = create_normal_vector_for_plane(axis, angle)
    return create_image_at_plane_from_vector(image, point_on_plane, normal_vector)


def create_image_at_plane_from_vector(
    image: sitk.Image, point_on_plane: np.array, normal_vector: np.array
):
    """Return a 2-D numpy array slice of *image* at the plane defined by *normal_vector*.

    Cat-A fix: master's ``transform.SetMatrix(normal_vector + [0, 0, 1])`` always
    crashed because it passed 3 elements to a 3×3 matrix (requires 9).  Replaced
    with ``_rotation_z_to_vector(normal_vector)`` which builds a proper 3×3
    Rodrigues rotation matrix aligned from z to the given normal and flattens it
    to 9 elements.  This replacement is NOT validated against master output (master
    crashed → no golden); tests only verify the result is a 2-D ndarray.
    """
    transform = sitk.AffineTransform(3)
    # Cat-A fix: use proper 9-element rotation matrix instead of crashing 3-element concat
    transform.SetMatrix(_rotation_z_to_vector(normal_vector).flatten().tolist())

    i_transform = transform.GetInverse()

    im_size = image.GetSize()
    spacing = image.GetSpacing()

    # Transform the point on the plane to the image's coordinate system
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([0, 0, -1, 0, -1, 0, 1, 0, 0])
    resampler.SetOutputOrigin(point_on_plane)
    resampler.SetSize(im_size)
    resampler.SetOutputSpacing(spacing)
    resampler.SetTransform(i_transform)

    resampled_im = resampler.Execute(image)

    # Convert the 3D image to a 2D array
    array = _itk().imview(resampled_im)

    # Select the middle slice along the third dimension
    slice_index = array.shape[2] // 2
    slice_2d = array[:, :, slice_index]

    return slice_2d
