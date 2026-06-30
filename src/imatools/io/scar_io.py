"""Scar-quantification file I/O migrated from ``imatools.common.scarqtools`` (M1.6a/c).

Functions here read/write the data formats used by the scar pipeline:
- ``prodStats.txt`` — blood-pool statistics and threshold scores.
- ``options.json`` — scar options for CEMRG MitkCemrgScarProjectionOptions.
- ``state.json`` — ScarQuantificationTools persistent state (paths + cmd names).
"""

import json
import os
import sys

from imatools.common.config import configure_logging

logger = configure_logging(log_name=__name__)


def read_stats_from_file(file_path):
    """Parse a ``prodStats.txt`` file produced by CEMRG.

    File format (line by line):
    - Line 0: method identifier starting with ``IIR_`` or ``MSD_``.
    - Line 1: ``mean_bp`` (float).
    - Line 2: ``std_bp`` (float).
    - Subsequent lines: ``V=<threshold>, SCORE=<score>`` pairs.

    Args:
        file_path: Path to the ``prodStats.txt`` file.

    Returns:
        Tuple of ``(method, mean_bp, std_bp, thresholds, scores)`` where
        ``method`` is 1 (IIR) or 2 (MSD), ``thresholds`` and ``scores`` are
        ``list[float]``.
    """
    mean_bp = None
    std_bp = None
    method = None

    with open(file_path, "r") as f:
        lines = f.readlines()

    thresholds = []
    scores = []
    for ix, line in enumerate(lines):
        line = line.strip()
        if ix == 0:
            method = 1 if line.startswith("IIR_") else 2
        elif line.startswith("V="):
            parts = line.split(", ")
            if len(parts) == 2:
                if parts[0].startswith("V="):
                    threshold = float(parts[0].split("=")[1])
                    thresholds.append(threshold)
                if parts[1].startswith("SCORE="):
                    score = float(parts[1].split("=")[1])
                    scores.append(score)
        elif mean_bp is None:
            mean_bp = float(line)
        elif std_bp is None:
            std_bp = float(line)

    return method, mean_bp, std_bp, thresholds, scores


def get_bloodpool_stats_from_file(file_path):
    """Return ``(mean_bp, std_bp)`` from a ``prodStats.txt`` file.

    Convenience wrapper around :func:`read_stats_from_file` when only the
    blood-pool statistics are needed.

    Args:
        file_path: Path to the ``prodStats.txt`` file.

    Returns:
        Tuple of ``(mean_bp, std_bp)`` floats.
    """
    _, mean_bp, std_bp, _, _ = read_stats_from_file(file_path)
    return mean_bp, std_bp


def create_scar_options_file(
    dir,
    opt_file="options.json",
    output_dir="OUTPUT",
    legacy=False,
    limits=None,
    radius=False,
    method=1,
    threshold_values=None,
):
    """Write a CEMRG scar-options JSON file.

    Args:
        dir:               Directory in which to write ``opt_file``.
        opt_file:          Filename for the options file (default ``options.json``).
        output_dir:        Value for ``output_dir`` key (default ``"OUTPUT"``).
        legacy:            Whether to use legacy ROI projection (default False).
        limits:            ROI limits as a list (default ``[-1, 3]``).
        radius:            Whether to use radius projection (default False).
        method:            Threshold method: 1 (IIR) or 2 (MSD) (default 1).
        threshold_values:  List of threshold values (default ``[0.97, 1.2, 1.32]``).

    Returns:
        None.  Raises ValueError if ``method`` is not 1 or 2.
    """
    if limits is None:
        limits = [-1, 3]
    if threshold_values is None:
        threshold_values = [0.97, 1.2, 1.32]

    if method != 1 and method != 2:
        logger.error("Error: Method must be 1 (IIR) or 2 (Msd)")
        raise ValueError("Error: Method must be 1 (IIR) or 2 (Msd)")

    dic = {
        "output_dir": output_dir,
        "roi_legacy_projection": legacy,
        "roi_limits": ",".join(map(str, limits)),
        "roi_radius": (radius or legacy),
        "threshold_values": ",".join(map(str, threshold_values)),
        "thresholds_method": method,
    }

    output_path = os.path.join(dir, opt_file)
    logger.info(f"Creating options file: {output_path}")

    with open(output_path, "w") as f:
        json.dump(dic, f)


def save_scar_state(path: str, config_dict: dict) -> None:
    """Write a ``ScarConfig`` dict to a JSON state file.

    The dict is expected to contain platform-keyed paths under ``"cemrg"`` and
    ``"mirtk"`` keys plus flat string keys ``"scar_cmd_name"`` and
    ``"clip_cmd_name"``.

    Args:
        path:        Full path to the state file (e.g. ``scarq_state.json``).
        config_dict: Serialisable dict produced by :class:`cli.scar.ScarConfig`.

    Raises:
        OSError: if the file cannot be written.
    """
    logger.info(f"Saving scar state to {path}")
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)


def load_scar_state(path: str) -> dict:
    """Load a scar state JSON file produced by :func:`save_scar_state`.

    Args:
        path: Full path to the state file.

    Returns:
        Dict with keys ``"cemrg"``, ``"mirtk"``, ``"scar_cmd_name"``,
        ``"clip_cmd_name"``.

    Raises:
        OSError:           if the file cannot be read.
        json.JSONDecodeError: if the file is not valid JSON.
    """
    with open(path, "r") as f:
        state = json.load(f)
    return state


def _current_platform() -> str:
    """Return ``sys.platform`` (thin helper for testability)."""
    return sys.platform
