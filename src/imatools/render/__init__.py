"""
Rendering layer for imatools.

Stateless data-in/figure-or-image-out primitives for turning meshes and
scalar fields into visual output (PNG snapshots, PDF pages, pyvista plots).
No file-path resolution, argument parsing, or hardcoded-path I/O lives here
— that belongs to `cli/` (offscreen VTK->PNG, pyvista mesh views) or to the
higher-level anatomical report functions in `mesh_report.py` (decoupled
separately).
"""

from .mesh_views import (
    pts_elem_to_pyvista,
    visualise_fibres,
    visualise_mesh,
    visualise_pericardium,
    visualise_two_meshes,
    visualise_vtx,
)
from .plots import (
    append_scar_to_pandas_dataframe,
    extract_scar_stats_from_file,
    plot_dict,
)
from .vtk_png import (
    render_vtk_to_png,
    render_vtk_to_single_png,
)

__all__ = [
    # vtk_png
    "render_vtk_to_png",
    "render_vtk_to_single_png",
    # mesh_views
    "pts_elem_to_pyvista",
    "visualise_mesh",
    "visualise_two_meshes",
    "visualise_vtx",
    "visualise_fibres",
    "visualise_pericardium",
    # plots
    "plot_dict",
    "extract_scar_stats_from_file",
    "append_scar_to_pandas_dataframe",
]
