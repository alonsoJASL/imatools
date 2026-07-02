import numpy as np
import vtk

from imatools.common.config import configure_logging

logger = configure_logging(__name__)


def clean_stl_file(input_path, output_path):
    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            f_out.write(line)
            if line.strip().startswith("endsolid"):  # Stop writing after endsolid
                break


def getBooleanOperationFilter(msh0, msh1, operation_str="union"):
    opts = {"union": 0, "intersection": 1, "difference": 2}
    bopd = vtk.vtkBooleanOperationPolyDataFilter()
    bopd.SetOperation(opts[operation_str])
    bopd.SetInputData(0, msh0)
    bopd.SetInputData(1, msh1)
    bopd.Update()

    return bopd


def getBooleanOperation(msh0, msh1, operation_str="union"):
    bopd = getBooleanOperationFilter(msh0, msh1, operation_str)
    return bopd.GetOutput()


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


def verify_cell_indices_from_mesh(msh1, msh_test, test_indices):
    """
    Verifies that the test_indices in the mesh (msh)
    are the same as the test_locations.

    test_locations is linked to test_indices via the
    centre of gravity of the mesh elements.
    """

    cog_test = get_cog_per_element(msh_test)
    test_cog = cog_test[test_indices, :]

    return verify_cell_indices(msh1, test_indices, test_cog)


def compute_global_bounds(vtk_files, input_type="ugrid"):
    """Compute the global bounding box (min/max for x, y, z) for a list of VTK files."""
    global_bounds = [
        float("inf"),
        -float("inf"),
        float("inf"),
        -float("inf"),
        float("inf"),
        -float("inf"),
    ]

    for vtk_file in vtk_files:
        data = create_vtk_reader(input_type, vtk_file, centered=True)
        bounds = data.GetBounds()

        # Update global min/max for each axis
        global_bounds[0] = min(global_bounds[0], bounds[0])  # xmin
        global_bounds[1] = max(global_bounds[1], bounds[1])  # xmax
        global_bounds[2] = min(global_bounds[2], bounds[2])  # ymin
        global_bounds[3] = max(global_bounds[3], bounds[3])  # ymax
        global_bounds[4] = min(global_bounds[4], bounds[4])  # zmin
        global_bounds[5] = max(global_bounds[5], bounds[5])  # zmax

    print(f"Global bounds: {global_bounds}")
    return global_bounds


def normalise_vtk_values(imsh, fieldname="scalars"):
    """
    Normalise the values of a vtkPolyData object
    """
    array = convertPointDataToNpArray(imsh, fieldname)
    array = (array - np.min(array)) / (np.max(array) - np.min(array))
    scalars = np_to_vtk_array(array, fieldname)
    omsh = vtk.vtkPolyData()
    omsh.DeepCopy(imsh)
    omsh.GetPointData().SetScalars(scalars)
    # .GetPointData().SetScalars(scalars)
    return omsh


# def detect_bridges_combined(polydata: vtk.vtkPolyData, max_distance=5.0, thickness_threshold=1.5) -> vtk.vtkIntArray:
#     """
#     Combine graph-based and thickness-based methods. A cell is flagged only if both
#     methods flag it as a bridge.
#     """
#     # Compute the individual flags.
#     graphFlag = detect_bridges_with_graph(polydata)
#     thicknessFlag = detect_bridges_with_thickness(polydata, max_distance, thickness_threshold)

#     numCells = polydata.GetNumberOfCells()
#     combinedFlag = vtk.vtkIntArray()
#     combinedFlag.SetName("CombinedBridgeFlag")
#     combinedFlag.SetNumberOfComponents(1)
#     combinedFlag.SetNumberOfTuples(numCells)

#     for i in range(numCells):
#         # Logical AND: flag cell if both methods flag it.
#         if graphFlag.GetTuple1(i) == 1 and thicknessFlag.GetTuple1(i) == 1:
#             combinedFlag.SetTuple1(i, 1)
#         else:
#             combinedFlag.SetTuple1(i, 0)


#     return combinedFlag


def get_combined_bounds(meshes):
    """
    Given a list of vtkPolyData meshes, computes and returns the combined bounds.

    Parameters:
        meshes (list of vtk.vtkPolyData): List of meshes.

    Returns:
        tuple: (xmin, xmax, ymin, ymax, zmin, zmax) representing the overall bounds.
    """
    if not meshes:
        raise ValueError("The list of meshes is empty!")

    # Initialize combined bounds with the bounds of the first mesh.
    combined = list(meshes[0].GetBounds())  # [xmin, xmax, ymin, ymax, zmin, zmax]

    # Iterate over the remaining meshes and update the combined bounds.
    for mesh in meshes[1:]:
        b = mesh.GetBounds()
        combined[0] = min(combined[0], b[0])  # xmin
        combined[1] = max(combined[1], b[1])  # xmax
        combined[2] = min(combined[2], b[2])  # ymin
        combined[3] = max(combined[3], b[3])  # ymax
        combined[4] = min(combined[4], b[4])  # zmin
        combined[5] = max(combined[5], b[5])  # zmax

    return tuple(combined)


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
