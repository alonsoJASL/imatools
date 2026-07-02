import numpy as np

from imatools.common.config import configure_logging

logger = configure_logging(__name__)


# Deprecated — use project_cell_data and project_point_data (from core.mesh) instead.
# These legacy names are now aliases defined at the bottom of this file.


norm2 = lambda a: np.linalg.norm(a)
norm_vec = lambda a: a / norm2(a)


def create_mapping(msh_left_name, msh_right_name, left_id, right_id, map_type="elem"):
    """Thin path-loading wrapper around ``core.mesh.create_mapping``.

    Reads both mesh files and delegates all logic to the object-based pure
    function.  The old path-based signature is preserved so that the deferred
    ``create_mapping_fibres.py`` (M1.6) can keep importing this symbol unchanged.
    """
    from imatools.core.mesh import create_mapping as _core_create_mapping  # noqa: PLC0415

    msh_left = readVtk(msh_left_name)
    msh_right = readVtk(msh_right_name)
    return _core_create_mapping(msh_left, msh_right, left_id, right_id, map_type)


# ---------------------------------------------------------------------------
# Re-export shim — geometry functions now live in imatools.core.geometry (T2b1).
# These names are injected into this module's namespace so that all existing
# ``from imatools.common.vtktools import …`` call sites keep working unchanged.
# ---------------------------------------------------------------------------
from imatools.core.geometry import (  # noqa: E402,F401
    compute_mesh_size,
    dot_prod_vec,
    get_bounding_box,
    get_cog_per_element,
    l2_norm,
    point_in_aabb,
    point_in_aabb_vectorized,
    precompute_valid_cells,
)

# ---------------------------------------------------------------------------
# Re-export shim — mesh transform functions now live in imatools.core.mesh (T2b2).
# These names are injected into this module's namespace so that all existing
# ``from imatools.common.vtktools import …`` call sites keep working unchanged.
# ---------------------------------------------------------------------------
from imatools.core.mesh import (  # noqa: E402,F401
    cell_to_point_data,
    clean_mesh,
    cogs_from_ugrid,
    compare_mesh_sizes,
    convertCellDataToNpArray,
    convertPointDataToNpArray,
    extractPointsAndElemsFromVtk,
    fibrorisScore,
    fibrosis_overlap,
    fibrosis_overlap_cells,
    fibrosis_overlap_points,
    fibrosis_score,
    fibrosis_score_cell,
    fibrosis_score_point,
    fibrosisOverlapCell,
    flip_xy,
    genericThreshold,
    get_element_cogs,
    getElemPermutation,
    getSurfaceArea,
    global_centre_of_mass,
    indices_at_scalar,
    join_vtk,
    map_cells,
    map_points,
    mask_cell_scalars,
    np_to_vtk_array,
    point_to_cell_data,
    project_cell_data,
    project_point_data,
    set_cell_scalars,
    set_cell_to_point_data,
    set_vtk_scalars,
    setCellDataToPointData,
    tag_elements_by_voxel_boxes,
    tag_mesh_elements_by_growing_from_seed,
    tag_mesh_elements_by_growing_from_seed_optimized,
    tag_mesh_elements_by_voxel_boxes,
    tag_mesh_elements_parallel_regions,
    thresholdExactValue,
    translate_to_point,
    ugrid2polydata,
    verify_cell_indices,
)
from imatools.io.mesh_io import (  # noqa: E402,F401
    clean_stl_file,
    export_as,
    read_vtk,
    readVtk,
    saveCarpAsVtk,
    vtk_from_points_file,
    write_vtk,
    writeVtk,
)
from imatools.parsers.dotmesh import parse_dotmesh_file  # noqa: E402,F401

# ---------------------------------------------------------------------------
# Legacy camelCase aliases for backward compatibility (M1.2-A).
# ---------------------------------------------------------------------------
projectCellData = project_cell_data  # noqa: N816
projectPointData = project_point_data  # noqa: N816
from imatools.core.mesh import get_mesh_volume  # noqa: E402,F401

# ---------------------------------------------------------------------------
# Re-export shim — render_vtk_to_png/render_vtk_to_single_png now live in
# imatools.render.vtk_png (render-layer extraction, M1.7/8 #1).
# create_vtk_reader/create_vtk_mapper/create_vtk_actor/center_vtk_data also
# now live there (M2a-1); re-exported here so that compute_global_bounds
# (a zero-caller dead function, slated for deletion in M2b) still resolves
# create_vtk_reader by name at call time.
# ---------------------------------------------------------------------------
from imatools.render.vtk_png import (  # noqa: E402,F401
    center_vtk_data,
    create_vtk_actor,
    create_vtk_mapper,
    create_vtk_reader,
    render_vtk_to_png,
    render_vtk_to_single_png,
)
