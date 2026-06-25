import numpy as np
import pyvista as pv
import vtk


def pts_elem_to_pyvista(pts, elem, add_tags=False, el_type="Tt"):

    tmp_elem = elem

    if el_type == "Tt":
        final_elem = tmp_elem[:, :4]
        tets = np.column_stack(
            (np.ones((final_elem.shape[0],), dtype=int) * 4, final_elem)
        ).flatten()
        cell_type = np.ones((final_elem.shape[0],), dtype=int) * vtk.VTK_TETRA
    elif el_type == "Tr":
        final_elem = tmp_elem[:, :3]
        tets = np.column_stack(
            (np.ones((final_elem.shape[0],), dtype=int) * 3, final_elem)
        ).flatten()
        cell_type = np.ones((final_elem.shape[0],), dtype=int) * vtk.VTK_TRIANGLE

    plt_msh = pv.UnstructuredGrid(tets, cell_type, pts)
    if add_tags:
        tags = tmp_elem[:, -1]
        plt_msh.cell_data["ID"] = tags

    return plt_msh


# ---------------------------------------------------------------------------
# Re-export shim — rotation_matrix now lives in imatools.core.geometry (T2b1).
# ---------------------------------------------------------------------------
from imatools.core.geometry import rotation_matrix  # noqa: E402,F401

# ---------------------------------------------------------------------------
# Re-export shim — rotate_mesh now lives in imatools.core.mesh (T2b2).
# ---------------------------------------------------------------------------
from imatools.core.mesh import rotate_mesh  # noqa: E402,F401
