"""Parked: quLATi mesh-pair downsampling.

Not packaged, quLATi is not yet an optional dependency, and the original script
imported the flat ``common.ioutils`` / ``common.vtktools`` modules that M2
deleted.  The body was removed once it became unrunnable; recover it from git
history if quLATi is revived.  See ``docs/future_work.md``.
"""

raise ImportError(
    "imatools.qulati_downsample_pair is parked as quLATi is not yet "
    "packaged as an optional dependency. See future_work.md."
)
