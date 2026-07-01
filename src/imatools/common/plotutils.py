# ---------------------------------------------------------------------------
# Re-export shim — the plotutils family now lives in imatools.render
# (render-layer extraction, M1.7/8 #1). Existing importers of
# imatools.common.plotutils keep working unchanged.
# ---------------------------------------------------------------------------
from imatools.render.mesh_views import (  # noqa: F401
    visualise_fibres,
    visualise_mesh,
    visualise_pericardium,
    visualise_two_meshes,
    visualise_vtx,
)
from imatools.render.plots import (  # noqa: F401
    append_scar_to_pandas_dataframe,
    extract_scar_stats_from_file,
    plot_dict,
)
