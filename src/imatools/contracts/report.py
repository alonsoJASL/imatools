"""
Mesh-report data contracts.

Back the `imatools-report` CLI: a mesh plus OPTIONAL per-anatomy data, so a
partial input renders only the anatomies actually present (a missing field is
a first-class "absent", not a crash), and rendering parameters kept separate
from the data being rendered.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyvista as pv


@dataclass
class MeshReportInputs:
    """
    Data inputs for a mesh quality-assessment report.

    Attributes:
        mesh: The whole-heart volumetric mesh (required).
        lon: Optional fibre orientation array, aligned with `mesh` cells.
        pericardium_scale: Optional per-node pericardium penalty-map scalar array.
        epicardium_surf: Optional epicardium surface element array (Tr).
        lv_endo_surf: Optional LV endocardium surface element array (Tr).
        rv_endo_surf: Optional RV endocardium surface element array (Tr).
        la_endo_surf: Optional LA endocardium surface element array (Tr).
        ra_endo_surf: Optional RA endocardium surface element array (Tr).
        rpvs_surf: Optional right pulmonary veins surface element array (Tr).
        svc_surf: Optional superior vena cava surface element array (Tr).
        san_vtx: Optional sino-atrial node vertex index array.
        fascicles_lv_vtx: Optional LV fascicles vertex index array.
        fascicles_rv_vtx: Optional RV fascicles vertex index array.
    """

    mesh: pv.UnstructuredGrid
    lon: Optional[np.ndarray] = None
    pericardium_scale: Optional[np.ndarray] = None
    epicardium_surf: Optional[np.ndarray] = None
    lv_endo_surf: Optional[np.ndarray] = None
    rv_endo_surf: Optional[np.ndarray] = None
    la_endo_surf: Optional[np.ndarray] = None
    ra_endo_surf: Optional[np.ndarray] = None
    rpvs_surf: Optional[np.ndarray] = None
    svc_surf: Optional[np.ndarray] = None
    san_vtx: Optional[np.ndarray] = None
    fascicles_lv_vtx: Optional[np.ndarray] = None
    fascicles_rv_vtx: Optional[np.ndarray] = None


@dataclass
class RenderParams:
    """
    Page-layout and rendering parameters for a mesh report.

    Attributes:
        fig_w: Page width in pixels.
        fig_h: Page height in pixels.
        colormap: Matplotlib colormap name for tags/scalars.
        zoom: Camera zoom magnitude.
        dpi: Dots per inch (resolution).
        title_fontsize: Fontsize of the title of each page.
        title_position: Title vertical position (1 = top of page).
    """

    fig_w: int = 2480
    fig_h: int = 3508
    colormap: str = "RdBu"
    zoom: float = 1
    dpi: float = 100
    title_fontsize: float = 44
    title_position: float = 0.9
