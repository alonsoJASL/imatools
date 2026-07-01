"""
Mesh-report composition functions.

Each ``render_<anatomy>_views`` is a straight port of the matching
``mesh_report.py`` ``visualise_*_all_views`` function, composed over the
stateless primitives in ``render/mesh_views.py``. Data comes from an already
populated ``MeshReportInputs`` (no file I/O here) and page/rendering options
from ``RenderParams``. Each function is self-gating: it no-ops if its
required ``inputs.*`` field(s) are ``None`` — a missing anatomy is skipped,
not a crash.
"""

from imatools.contracts.report import MeshReportInputs, RenderParams
from imatools.render.mesh_views import (
    pts_elem_to_pyvista,
    visualise_fibres,
    visualise_mesh,
    visualise_pericardium,
    visualise_two_meshes,
    visualise_vtx,
)


def render_mesh_views(inputs: MeshReportInputs, params: RenderParams, pdf) -> None:
    """Whole-mesh anterior/posterior views, opaque and translucent."""
    options = [
        ("Anterior view", 0, 1.0),
        ("Posterior view", 180, 1.0),
        ("Anterior view - Translucent", 0, 0.7),
        ("Posterior view - Translucent", 180, 0.7),
    ]

    for title, az_incr, op in options:
        visualise_mesh(
            plt_msh=inputs.mesh,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            colormap=params.colormap,
            zoom=params.zoom,
            pdf_object=pdf,
            camera_azimuth_increment=az_incr,
            title=title,
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            opacity=op,
            dpi=params.dpi,
        )


def render_fibres_views(inputs: MeshReportInputs, params: RenderParams, pdf) -> None:
    """Per-chamber fibre views. No-op if `inputs.lon` is None."""
    if inputs.lon is None:
        return

    options = [
        # title, tag list, [roll, azimuth, elevation], num_fibres
        ("LV fibres (anterior view)", [1], [0, 0, 0], 10000),
        ("LV fibres (basal view)", [1], [0, 0, 89], 10000),
        ("LV fibres (apical view)", [1], [0, 0, -89], 20000),
        ("RV fibres (lateral view)", [2], [0, -90, 0], 20000),
        ("LA fibres (anterior view)", [3, 26], [0, 0, 0], 20000),
        ("LA fibres (posterior view)", [3, 26], [0, 180, 0], 20000),
        ("LA fibres (roof view)", [3, 26], [0, 0, 89], 20000),
        ("RA fibres (roof view)", [4, 26], [0, 0, 89], 20000),
        ("RA fibres (lateral view)", [4, 26], [0, -90, 0], 20000),
    ]

    for mytitle, tag, camera, num_fibres in options:
        visualise_fibres(
            plt_msh=inputs.mesh,
            lon=inputs.lon,
            tag=tag,
            color=params.colormap,
            camera_roll_increment=camera[0],
            camera_azimuth_increment=camera[1],
            camera_elevation_increment=camera[2],
            title=mytitle,
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            pdf_object=pdf,
            zoom=params.zoom,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            num_fibres=num_fibres,
            dpi=params.dpi,
        )


def render_pericardium_views(inputs: MeshReportInputs, params: RenderParams, pdf) -> None:
    """Pericardium penalty-map views. No-op if `inputs.pericardium_scale` is None."""
    if inputs.pericardium_scale is None:
        return

    options = [
        # title, azimuth, elevation
        ("Pericardium scale (anterior view)", 0, 0),
        ("Pericardium scale (posterior view)", 180, 0),
        ("Pericardium scale (top view)", 0, 89),
        ("Pericardium scale (apical view)", 0, -89),
    ]

    for mytitle, az_incr, el_incr in options:
        visualise_pericardium(
            plt_msh=inputs.mesh,
            pdf_object=pdf,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            colormap=params.colormap,
            camera_azimuth_increment=az_incr,
            camera_elevation_increment=el_incr,
            title=mytitle,
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            pericardium_scale=inputs.pericardium_scale,
            dpi=params.dpi,
            zoom=params.zoom,
        )


def render_epicardium_views(inputs: MeshReportInputs, params: RenderParams, pdf) -> None:
    """Epicardium surface views. No-op if `inputs.epicardium_surf` is None."""
    if inputs.epicardium_surf is None:
        return

    plt_msh_surf = pts_elem_to_pyvista(
        pts=inputs.mesh.points, elem=inputs.epicardium_surf, add_tags=False, el_type="Tr"
    )

    visualise_mesh(
        plt_msh=plt_msh_surf,
        fig_w=params.fig_w,
        fig_h=params.fig_h,
        color="lightgrey",
        zoom=params.zoom,
        pdf_object=pdf,
        title="Epicardium (anterior view)",
        title_fontsize=params.title_fontsize,
        title_position=params.title_position,
        dpi=params.dpi,
    )

    visualise_mesh(
        plt_msh=plt_msh_surf,
        fig_w=params.fig_w,
        fig_h=params.fig_h,
        color="lightgrey",
        zoom=params.zoom,
        camera_azimuth_increment=180,
        pdf_object=pdf,
        title="Epicardium (posterior view)",
        title_fontsize=params.title_fontsize,
        title_position=params.title_position,
        dpi=params.dpi,
    )


def render_endocardia_views(inputs: MeshReportInputs, params: RenderParams, pdf) -> None:
    """Per-chamber endocardium surface views. Each chamber is independently gated."""
    if inputs.lv_endo_surf is not None:
        plt_msh_surf = pts_elem_to_pyvista(
            pts=inputs.mesh.points, elem=inputs.lv_endo_surf, add_tags=False, el_type="Tr"
        )

        visualise_mesh(
            plt_msh=plt_msh_surf,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            color="lightgrey",
            zoom=params.zoom,
            pdf_object=pdf,
            title="LV endocardium (anterior view)",
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            dpi=params.dpi,
        )

        visualise_mesh(
            plt_msh=plt_msh_surf,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            color="lightgrey",
            zoom=params.zoom,
            camera_azimuth_increment=180,
            pdf_object=pdf,
            title="LV endocardium (posterior view)",
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            dpi=params.dpi,
        )

        visualise_mesh(
            plt_msh=plt_msh_surf,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            color="lightgrey",
            zoom=params.zoom,
            camera_elevation_increment=89,
            pdf_object=pdf,
            title="LV endocardium (top view)",
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            dpi=params.dpi,
        )

    if inputs.rv_endo_surf is not None:
        plt_msh_surf = pts_elem_to_pyvista(
            pts=inputs.mesh.points, elem=inputs.rv_endo_surf, add_tags=False, el_type="Tr"
        )

        visualise_mesh(
            plt_msh=plt_msh_surf,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            color="lightgrey",
            zoom=params.zoom,
            pdf_object=pdf,
            title="RV endocardium (septal view)",
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            dpi=params.dpi,
            camera_azimuth_increment=90,
        )

        visualise_mesh(
            plt_msh=plt_msh_surf,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            color="lightgrey",
            zoom=params.zoom,
            camera_azimuth_increment=-90,
            pdf_object=pdf,
            title="RV endocardium (lateral view)",
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            dpi=params.dpi,
        )

        visualise_mesh(
            plt_msh=plt_msh_surf,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            color="lightgrey",
            zoom=params.zoom,
            camera_elevation_increment=89,
            camera_azimuth_increment=90,
            pdf_object=pdf,
            title="RV endocardium (basal view)",
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            dpi=params.dpi,
        )

    if inputs.la_endo_surf is not None:
        plt_msh_surf = pts_elem_to_pyvista(
            pts=inputs.mesh.points, elem=inputs.la_endo_surf, add_tags=False, el_type="Tr"
        )

        visualise_mesh(
            plt_msh=plt_msh_surf,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            color="lightgrey",
            zoom=params.zoom,
            pdf_object=pdf,
            title="LA endocardium (anterior view)",
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            dpi=params.dpi,
        )

        visualise_mesh(
            plt_msh=plt_msh_surf,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            color="lightgrey",
            zoom=params.zoom,
            camera_azimuth_increment=180,
            pdf_object=pdf,
            title="LA endocardium (posterior view)",
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            dpi=params.dpi,
        )

        visualise_mesh(
            plt_msh=plt_msh_surf,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            color="lightgrey",
            zoom=params.zoom,
            camera_elevation_increment=-89,
            pdf_object=pdf,
            title="LA endocardium (basal view)",
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            dpi=params.dpi,
        )

    if inputs.ra_endo_surf is not None:
        plt_msh_surf = pts_elem_to_pyvista(
            pts=inputs.mesh.points, elem=inputs.ra_endo_surf, add_tags=False, el_type="Tr"
        )

        visualise_mesh(
            plt_msh=plt_msh_surf,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            color="lightgrey",
            zoom=params.zoom,
            pdf_object=pdf,
            title="RA endocardium (septal view)",
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            dpi=params.dpi,
            camera_azimuth_increment=90,
        )

        visualise_mesh(
            plt_msh=plt_msh_surf,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            color="lightgrey",
            zoom=params.zoom,
            pdf_object=pdf,
            title="RA endocardium (lateral view)",
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            dpi=params.dpi,
            camera_azimuth_increment=-90,
        )

        visualise_mesh(
            plt_msh=plt_msh_surf,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            color="lightgrey",
            zoom=params.zoom,
            pdf_object=pdf,
            title="RA endocardium (basal view)",
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            dpi=params.dpi,
            camera_azimuth_increment=-90,
            camera_elevation_increment=-89,
        )


def render_veins_views(inputs: MeshReportInputs, params: RenderParams, pdf) -> None:
    """Right pulmonary veins + superior vena cava views. Each vein is independently gated."""
    if inputs.rpvs_surf is not None:
        plt_msh_surf = pts_elem_to_pyvista(
            pts=inputs.mesh.points, elem=inputs.rpvs_surf, add_tags=False, el_type="Tr"
        )

        visualise_two_meshes(
            plt_msh_1=inputs.mesh,
            plt_msh_2=plt_msh_surf,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            color_1="lightgrey",
            color_2="red",
            zoom=params.zoom,
            pdf_object=pdf,
            title="Right pulmonary veins (location)",
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            dpi=params.dpi,
            camera_elevation_increment=89,
            opacity_mesh_1=0.3,
            opacity_mesh_2=1,
        )

        visualise_mesh(
            plt_msh=plt_msh_surf,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            color="lightgrey",
            zoom=params.zoom + 3,
            pdf_object=pdf,
            title="Right pulmonary veins (septal view)",
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            dpi=params.dpi,
            camera_azimuth_increment=-90,
        )

        visualise_mesh(
            plt_msh=plt_msh_surf,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            color="lightgrey",
            zoom=params.zoom + 3,
            pdf_object=pdf,
            title="Right pulmonary veins (basal view)",
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            dpi=params.dpi,
            camera_elevation_increment=-90,
        )

    if inputs.svc_surf is not None:
        plt_msh_surf = pts_elem_to_pyvista(
            pts=inputs.mesh.points, elem=inputs.svc_surf, add_tags=False, el_type="Tr"
        )

        visualise_two_meshes(
            plt_msh_1=inputs.mesh,
            plt_msh_2=plt_msh_surf,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            color_1="lightgrey",
            color_2="red",
            zoom=params.zoom,
            pdf_object=pdf,
            title="Superior vena cava (location)",
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            dpi=params.dpi,
            camera_elevation_increment=89,
            opacity_mesh_1=0.3,
            opacity_mesh_2=1,
        )

        visualise_mesh(
            plt_msh=plt_msh_surf,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            color="lightgrey",
            zoom=params.zoom + 3,
            pdf_object=pdf,
            title="Superior vena cava (top view)",
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            dpi=params.dpi,
            camera_elevation_increment=89,
        )

        visualise_mesh(
            plt_msh=plt_msh_surf,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            color="lightgrey",
            zoom=params.zoom + 3,
            pdf_object=pdf,
            title="Superior vena cava (basal view)",
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            dpi=params.dpi,
            camera_elevation_increment=-89,
        )


def render_eas_views(inputs: MeshReportInputs, params: RenderParams, pdf) -> None:
    """Early activation site views. Each site is independently gated."""
    if inputs.san_vtx is not None:
        visualise_vtx(
            plt_msh=inputs.mesh,
            vtx=inputs.san_vtx,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            color_mesh="lightgrey",
            color_vtx="red",
            zoom=params.zoom,
            pdf_object=pdf,
            title="Sino-atrial node (top view)",
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            dpi=params.dpi,
            camera_elevation_increment=89,
            opacity=0.3,
        )

    if inputs.fascicles_lv_vtx is not None:
        visualise_vtx(
            plt_msh=inputs.mesh,
            vtx=inputs.fascicles_lv_vtx,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            color_mesh="lightgrey",
            color_vtx="red",
            zoom=params.zoom,
            pdf_object=pdf,
            title="LV fascicles (anterior view)",
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            dpi=params.dpi,
            opacity=0.3,
        )

    if inputs.fascicles_rv_vtx is not None:
        visualise_vtx(
            plt_msh=inputs.mesh,
            vtx=inputs.fascicles_rv_vtx,
            fig_w=params.fig_w,
            fig_h=params.fig_h,
            color_mesh="lightgrey",
            color_vtx="red",
            zoom=params.zoom,
            pdf_object=pdf,
            title="RV fascicles (anterior view)",
            title_fontsize=params.title_fontsize,
            title_position=params.title_position,
            dpi=params.dpi,
            opacity=0.3,
        )
