"""
Rendering layer for imatools.

Stateless data-in/figure-or-image-out primitives for turning meshes and
scalar fields into visual output (PNG snapshots, PDF pages, pyvista plots).
No file-path resolution, argument parsing, or hardcoded-path I/O lives here
— that belongs to `cli/` (offscreen VTK->PNG, pyvista mesh views, and the
mesh-report composition functions in `report_views.py`).
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
from .report_views import (
    render_eas_views,
    render_endocardia_views,
    render_epicardium_views,
    render_fibres_views,
    render_mesh_views,
    render_pericardium_views,
    render_veins_views,
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
    # report_views
    "render_mesh_views",
    "render_fibres_views",
    "render_pericardium_views",
    "render_epicardium_views",
    "render_endocardia_views",
    "render_veins_views",
    "render_eas_views",
]
