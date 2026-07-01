# ---------------------------------------------------------------------------
# Re-export shim — pts_elem_to_pyvista now lives in imatools.render.mesh_views
# (render-layer extraction, M1.7/8 #1).
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Re-export shim — rotation_matrix now lives in imatools.core.geometry (T2b1).
# ---------------------------------------------------------------------------
from imatools.core.geometry import rotation_matrix  # noqa: E402,F401

# ---------------------------------------------------------------------------
# Re-export shim — rotate_mesh now lives in imatools.core.mesh (T2b2).
# ---------------------------------------------------------------------------
from imatools.core.mesh import rotate_mesh  # noqa: E402,F401
from imatools.render.mesh_views import pts_elem_to_pyvista  # noqa: E402,F401
