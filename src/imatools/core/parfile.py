# src/imatools/core/parfile.py
"""Pure dict-manipulation functions for Meshtools3d parameter files.

Migrated from ``imatools.common.m3dutils`` (T2e).  Legacy callers that import
from ``imatools.common.m3dutils`` continue to work via the re-export shim at
the bottom of that module.

Cat-B bug preserved verbatim (per Wave-2 bug policy):
  - ``update_pot`` does a **shallow** ``dict.copy()`` of the base pot, so
    calling ``update_pot(base, patch)`` mutates ``base``'s sub-dicts in place.
    The golden locks this behaviour; do NOT change to deepcopy.
"""

from __future__ import annotations


def update_pot(pot: dict, new_pot: dict):
    update_pot = pot.copy()
    for section in new_pot.keys():
        for key, value in new_pot[section].items():
            if value is not None:
                update_pot[section][key] = value

    return update_pot
