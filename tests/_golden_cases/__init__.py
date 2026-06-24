"""Golden-capture case providers.

Each submodule exposes ``CASES: list[CaptureCase]`` describing master functions to
record. Submodules import master code (``imatools.common.*`` / bare ``common.*``) and
the shared builders (``_fixtures``), so they are imported ONLY by the capture harness
running in the master env — never by pytest. The Wave-1 characterization tasks add one
submodule per concern (label, image, mesh, metrics, ...) so they stay merge-disjoint.
"""
