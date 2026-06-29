"""Pure scar-quantification logic migrated from ``imatools.common.scarqtools`` (M1.6a).

All functions here are stateless and accept explicit arguments — no class state, no
singletons. ``imatools.common.scarqtools.ScarQuantificationTools`` keeps the CLI/state
layer (M1.6c); this module holds the deterministic numeric core that is golden-backed.

``get_threshold_values`` consolidates the byte-identical duplicate that existed in both
``enhance_debug_scar.py`` and ``pool_enhance_debug_scar.py``.

``enhance_scar_array`` extracts the inline triple-loop kernel from
``enhance_debug_scar.py::main`` into a pure, testable function; the serial/parallel
dispatch is a M1.6c concern.

``mask_segmentation_above_threshold`` preserves master's arg-order bug in the
``get_threshold`` call verbatim (documented below).
"""

import numpy as np

from imatools.common.config import configure_logging

logger = configure_logging(log_name=__name__)


def get_scar_method(scar_method: str) -> int:
    """Return the integer method code for a scar-method string.

    Args:
        scar_method: ``"iir"`` or ``"msd"``.

    Returns:
        1 for IIR, 2 for MSD.

    Raises:
        KeyError: if ``scar_method`` is not recognised.
    """
    midic = {
        "iir": 1,
        "msd": 2,
    }
    return midic[scar_method]


def get_threshold(method: int, value, mean_bp, std_bp):
    """Compute a threshold value from blood-pool statistics.

    Args:
        method:  1 for IIR (``value * mean_bp``), 2 for MSD (``value * std_bp + mean_bp``).
        value:   multiplier / number of standard deviations.  If <= 0, returns 0.
        mean_bp: blood-pool mean intensity.
        std_bp:  blood-pool standard deviation.

    Returns:
        Numeric threshold, or 0 if ``value <= 0``.
    """
    method_dict = {
        1: lambda x, mbp, stdb: x * mbp,
        2: lambda x, mbp, stdb: x * stdb + mbp,
    }
    output = method_dict[method](value, mean_bp, std_bp) if value > 0 else 0
    return output


def get_threshold_values(thresholds, mean_bp, std_bp, method: str):
    """Convert a list of raw threshold multipliers into intensity thresholds.

    Consolidates the byte-identical implementation duplicated across
    ``enhance_debug_scar.py`` and ``pool_enhance_debug_scar.py``.

    Args:
        thresholds:  list of raw threshold values (e.g. ``[0.97, 1.2, 1.32]``).
        mean_bp:     blood-pool mean intensity.
        std_bp:      blood-pool standard deviation.
        method:      ``"iir"`` or ``"msd"``.

    Returns:
        List of intensity threshold values.
    """
    if method == "iir":
        threshold_values = [th * mean_bp for th in thresholds]
    else:
        threshold_values = [mean_bp + std_bp * th for th in thresholds]
    return threshold_values


def enhance_scar_array(
    scar_array: np.ndarray, im_array: np.ndarray, threshold_values
) -> np.ndarray:
    """Voxel-wise scar corridor enhancement kernel.

    Extracted from the inline triple-loop in ``enhance_debug_scar.py::main``.
    Voxels with ``scar_value > 1`` are re-labelled starting at 2, incrementing by 1 for
    each threshold the LGE intensity exceeds.  Voxels with ``scar_value <= 1`` are
    unchanged.

    Args:
        scar_array:        3-D integer array (background=0, corridor=1, scar-corridor=2+).
        im_array:          3-D float array of LGE intensities (same shape).
        threshold_values:  list of intensity thresholds (ascending, from
                           ``get_threshold_values``).

    Returns:
        Enhanced array (same shape and dtype as ``scar_array``).
    """
    enhanced_array = np.copy(scar_array)
    for x in range(scar_array.shape[0]):
        for y in range(scar_array.shape[1]):
            for z in range(scar_array.shape[2]):
                scar_value = scar_array[x, y, z]
                lge_value = im_array[x, y, z]
                if scar_value > 1:
                    enhanced_value = 2
                    for th in threshold_values:
                        enhanced_value += 1 if lge_value > th else 0
                    enhanced_array[x, y, z] = enhanced_value
    return enhanced_array


def mask_voxels_above_threshold(
    im,
    mask,
    thres_mean,
    thres_std,
    scar_method: str,
    thres_value=0,
    mask_value=0,
    ignore_im=None,
):
    """Mask image voxels that exceed a blood-pool threshold.

    Args:
        im:           SimpleITK image to mask.
        mask:         Binary mask image.
        thres_mean:   Blood-pool mean intensity.
        thres_std:    Blood-pool standard deviation.
        scar_method:  ``"iir"`` or ``"msd"``.
        thres_value:  Raw threshold multiplier (default 0 → no threshold).
        mask_value:   Value to assign to masked voxels.
        ignore_im:    Optional image; voxels > 0 here are excluded from masking.

    Returns:
        Masked SimpleITK image.
    """
    from imatools.core.image import mask_image  # noqa: PLC0415

    method = get_scar_method(scar_method)
    thres = get_threshold(method, thres_value, thres_mean, thres_std)
    masked_im = mask_image(
        im=im, mask=mask, mask_value=mask_value, threshold=thres, ignore_im=ignore_im
    )
    return masked_im


def mask_segmentation_above_threshold(
    seg_path,
    im,
    mask,
    thres_mean,
    thres_std,
    scar_method: str,
    thres_value=0,
    mask_value=0,
    ignore_im=None,
):
    """Mask a segmentation image using blood-pool thresholding.

    Loads the segmentation from ``seg_path``, computes a threshold from the
    blood-pool statistics, derives a restriction mask, and applies it.

    # BUG (M1.6 Q3): master passes get_threshold args in wrong order; preserved
    # under golden, deliberate re-baseline deferred.
    # Master call: get_threshold(method, thres_mean, thres_std, thres_value)
    # Correct call: get_threshold(method, thres_value, thres_mean, thres_std)
    # The golden locks the buggy behaviour.

    Args:
        seg_path:    Path to the segmentation image file.
        im:          SimpleITK LGE image.
        mask:        Binary mask image.
        thres_mean:  Blood-pool mean intensity.
        thres_std:   Blood-pool standard deviation.
        scar_method: ``"iir"`` or ``"msd"``.
        thres_value: Raw threshold multiplier (default 0).
        mask_value:  Value to assign to masked voxels.
        ignore_im:   Optional image; voxels > 0 here are excluded from masking.

    Returns:
        Masked segmentation SimpleITK image.
    """
    from imatools.common.itktools import load_image  # noqa: PLC0415
    from imatools.core.image import get_mask_with_restrictions, simple_mask  # noqa: PLC0415

    method = get_scar_method(scar_method)
    # BUG preserved verbatim: master passes (method, thres_mean, thres_std, thres_value)
    # instead of (method, thres_value, thres_mean, thres_std).
    thres = get_threshold(method, thres_mean, thres_std, thres_value)
    mask_from_im = get_mask_with_restrictions(im, mask, thres, ignore_im=ignore_im)

    seg = load_image(seg_path)
    masked_seg = simple_mask(im=seg, mask=mask_from_im, mask_value=mask_value)
    return masked_seg
