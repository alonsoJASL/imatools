# API Reference

The toolbox map. `imatools` is **stateless and contract-driven**: `core` holds pure
workflows, `io` reads/writes formats and returns typed contracts, and `contracts`
defines what passes between layers. For the design rationale and the canonical
command flow, start with [Architecture](overview.md).

!!! note "Import name"
    The package is imported as `imatools` (e.g. `from imatools.core import mesh`).
    To browse a symbol's signature, use `imatools-<tool> --help` or read the source
    under `src/imatools/`.

## Core — pure workflows (`imatools.core`)

| Module | What it does |
|---|---|
| `geometry` | Geometric primitives and calculations on points/cells. |
| `image` | Image-array workflows (indexing, sampling, coordinate conversion). |
| `io` | In-memory helpers shared across the core workflows. |
| `label` | Label-map operations (binarise, relabel, merge, dtype-safe combine). |
| `mesh` | Mesh workflows — mapping, scalar projection, `fibrosis_score`, quality metrics. |
| `mesh_topology` | Connectivity / topology utilities extracted during the refactor. |
| `metrics` | Characterization and comparison metrics for meshes and fields. |
| `parfile` | In-memory parfile (`pot`) construction and manipulation. |
| `scar` | Scar thresholding (`get_threshold`), `CEMRGAPP_IGNORE`, scar-score logic. |
| `segmentation` | Segmentation workflows behind the `imatools-segmentation` subcommands. |
| `spatial` | Spatial transforms and coordinate math. |

## IO — format load/save (`imatools.io`)

| Module | What it does |
|---|---|
| `carp_io` | CARP `.pts`/`.elem` read/write (incl. triangle-mesh element detection). |
| `image_io` | NIfTI / medical-image load and save (SimpleITK-backed). |
| `mesh_io` | VTK polydata load/save and mesh export. |
| `parfile_io` | Parfile (`pot`) read/write, e.g. `get_empty_pot`. |
| `paths` | Path helpers used by the IO layer. |
| `scar_io` | Scar-specific inputs/outputs (stats files, corridors). |

## Contracts — data passed between layers (`imatools.contracts`)

| Module | What it does |
|---|---|
| `image` | Image contract (array + metadata). |
| `mesh` | Mesh contract (points, cells, scalar arrays). |
| `operations` | Operation-parameter contracts shared across workflows. |
| `report` | Inputs/parameters for the report and render pipeline. |

## Supporting packages

| Module | What it does |
|---|---|
| `parsers.dotmesh` | Parser for `.mesh` (dotmesh) files → CARP. |
| `render.mesh_views` / `render.plots` / `render.report_views` / `render.vtk_png` | VTK-to-PNG rendering and report views used by `imatools-report`. |

!!! info "Looking for the orchestration pattern?"
    The high-level command flow and the core types are documented on the
    [Architecture](overview.md) page.
