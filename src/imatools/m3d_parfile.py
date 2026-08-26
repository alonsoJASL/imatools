"""Parked: mesh3d parameter-file generation.

Duplicates pycemrg-meshing, which is not yet an optional dependency, and the
original script imported the flat ``common.m3dutils`` module that M2 deleted.
The body was removed once it became unrunnable; recover it from git history if
this is revived here instead of in pycemrg-meshing.  See ``docs/future_work.md``.
"""

raise ImportError(
    "imatools.m3d_parfile is parked: duplicates pycemrg-meshing and is not yet "
    "packaged as an optional dependency. See future_work.md."
)
