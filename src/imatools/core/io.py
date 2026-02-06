# src/imatools/core/io.py

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def check_file_exists(path_to_file: Path) -> None:
    """
    Checks if the file exists at the given path.

    :param path_to_file: Path to the file
    :type path_to_file: Path
    :return: True if the file exists, False otherwise
    :rtype: bool
    """
    if not path_to_file.exists():
        logger.error(f'File not found: {path_to_file}')
        raise FileNotFoundError(f'File not found: {path_to_file}')
    logger.info(f'File found: {path_to_file}')