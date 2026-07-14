# src/imatools/core/parfile.py
"""Pure dict-manipulation functions for Meshtools3d parameter files.

Migrated from ``imatools.common.m3dutils`` (T2e); that shim module was deleted
in M2 — this is now the sole home.

M3-C3: ``update_pot`` deep-copies its base pot so it no longer mutates the
caller's input in place. This is golden-neutral — the golden captures the
returned merged dict (unchanged); only the input-mutation side effect is fixed.
"""

from __future__ import annotations

import copy


def update_pot(pot: dict, new_pot: dict):
    merged = copy.deepcopy(pot)
    for section in new_pot.keys():
        for key, value in new_pot[section].items():
            if value is not None:
                merged[section][key] = value

    return merged
