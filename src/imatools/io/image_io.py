# src/imatools/io/image_io.py

import json 
import logging 
import SimpleITK as sitk

from pathlib import Path

from imatools.core.io import check_file_exists

logger = logging.getLogger(__name__)

def load_image(
    path_to_image: Path,
) -> sitk.Image:
    """
    Reads Image into SimpleITK format.
    
    :param path_to_image: Path to the image file
    :type path_to_image: Path
    :return: Image in SimpleITK format
    :rtype: sitk.Image
    """

    check_file_exists(path_to_image)
    
    image = sitk.ReadImage(str(path_to_image))

    logger.info(f'Image loaded: {path_to_image}')
    
    return image
