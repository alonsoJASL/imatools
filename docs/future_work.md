# imatools, Task File

> Canonical backlog for `imatools`. Tracks work intentionally parked out of the
> master→development refactor plus maintenance surfaced since. Lives on `development`,
> rides into `main` on release.

## Decisions

<!-- Standing rules a reader must know to make sense of the tasks below. -->

- **Scope:** every item here is something to *design* or *decide*, not a behaviour to preserve, all are excluded from the golden-master characterization net.
- **Golden = oracle:** never re-capture/re-baseline a golden just to make a test pass. A deliberate re-baseline (e.g. `fullfile`) must be its own explicit, reviewed commit with the diff explained.
- **`imview` lifetime rule:** `core/image.imview` (`GetArrayViewFromImage`) is a non-owning view, the `sitk.Image` must outlive it. If an array or slice escapes the function, or the source may be a temporary, use the copying `imarray` (`GetArrayFromImage`) instead. Audited 2026-07-21: all current callers safe.
- **`io/parfile_io.py::load_from_par` reads `encoding="utf-8"`** (master uses locale-default `open`), intentional Cat-A env-crash fix, golden-neutral. To restore exact master parity, revert and make the fixture ASCII.
- **Milestone legend:** M1 = CLI migration, M2 = drop `common/` shims, M3 = quality debt, M4 = land into `development`; Wave 2 = itk/vtk/io migration wave; T1x/T2x = characterization task codes.
- **Category legend:** Cat-A = master-crashing paths (no golden exists, cannot auto-verify); Cat-B = latent master bugs preserved on purpose and golden-locked.
- **`legacy` branch** = old flat pre-refactor code = the behaviour oracle.

## Open

### `io/carp_io.py::save_to_carp_txt`, hardcoded element tag
- **Done when:** honours a tag column/arg instead of always writing `fmt='Tr %d %d %d 1'`; `carp/saveToCarpTxt_elem_lines.json` regenerated from fixed output.
- **Blocked on:** a real live consumer (currently none, only the parked `qulati_downsample_pair.py` + its own goldens)
- **Size:** S
- **Refs:** `io/carp_io.py::save_to_carp_txt` (Cat-B, from T1j/T2c2)
- **Notes:** deferred deliberately, do not churn a golden for a function no live code calls.

### Validate `load_nrrd_image` fix against real data
- **Done when:** the `SetSpacing` 3×3 → per-axis row-norms fix is checked against a real NRRD with non-trivial space directions.
- **Blocked on:** a real NRRD input file + domain review
- **Size:** S
- **Refs:** `io/image_io.py::load_nrrd_image` (Cat-A, was M3 P1.1), no golden exists (master crashed).

### Validate `create_image_at_plane_from_vector` fix against real data
- **Done when:** the `_rotation_z_to_vector` Rodrigues construction is checked against a real oblique-plane slice case (only known to return a 2-D ndarray structurally).
- **Blocked on:** a real oblique-plane input + domain review
- **Size:** S
- **Refs:** `core/spatial.py::create_image_at_plane_from_vector` (Cat-A, was M3 P1.2), no golden exists (master crashed).

### SegmentationGraph, spectral smoothing of label maps
- **Done when:** spectral label-map smoothing is implemented (graph Laplacian → low-frequency eigenvectors) and validated on a small synthetic case (e.g. noisy 8³ cube).
- **Blocked on:** nothing (parked, never worked, no behaviour to preserve)
- **Size:** L
- **Refs:** `imatools/common/SegmentationGraph.py` (legacy, source); target home `core/graph.py` (needs `networkx` + `scipy`)
- **Notes:** rewrite, don't port, known broken (`.asfptype()` typo, missing `array2im`, `U_k@U_k.T` should be `U_k@(U_k.T@f)`, per-voxel graph intractable, use a region-adjacency graph or downsample).

### Add `point2cell` / `cell2point` to `imatools-mesh`
- **Done when:** point↔cell data exchange exists as general `imatools-mesh` subcommands over VTK's `vtkPointDataToCellData` / `vtkCellDataToPointData`, golden-backed against a synthetic mesh.
- **Blocked on:** nothing
- **Size:** M
- **Refs:** dead `scarq_tools.py` modes (from M1.6 scar audit), nothing to migrate; implement fresh.

### Promote generic mesh ops out of `imatools-scar` → `imatools-mesh`
- **Done when:** `vscar-cog` is offered as `imatools-mesh cog` reusing `core/mesh.cogs_from_ugrid`; optionally alias the old name.
- **Blocked on:** nothing
- **Size:** S
- **Refs:** `cli/scar.py` (`vscar-cog`→`execute_cog_mesh`, `core/mesh.py:333`)
- **Notes:** `vscar-scale` / `vscar-deform` are also non-scar but shell out to external binaries, left in the scar pipeline by M1.6 Q2, lower priority.

## Done

<!-- Append-only. One line each. Newest at top. -->

- 2026-07-21 removed `fullfile`/`mkdirplus`/`searchFileByType` from `io/paths.py` (no live consumer in tree, notebooks, or pycemrg suite); retired their 5 golden cases
- 2026-07-21 isolated `mkdirplus` temp-dir pollution: `get_subfolders_sorted.json` re-baselined to `["alpha","beta"]` (subsumed by the `fullfile` removal above)
- 2026-07-21 imview use-after-free: `create_image_at_plane_from_vector` copies its slice out, @`54fcdda`
- 2026-07-21 imview use-after-free: `dice_score` rebinds params to `true_view`/`pred_view`, @`54fcdda`
- 2026-07-21 rename `saveToCarpTxt` → `save_to_carp_txt` (`Union[str,Path]`, deprecated alias kept), @`678d2df`
- 2026-07-15 `io/carp_io.py` off `fullfile` (`loadCarpMesh` builds paths with `Path`), `_paths()` accessor deleted, no more `io/paths` dep, @`9912d7c`
- 2026-07-02 M3-B1 `relabel_image` → `binarise(image, background, foreground)` (widened output dtype; alias kept)
- 2026-07-02 M3-C4 `create_normal_vector_for_plane`, removed dead `angle_rad` line
- 2026-07-02 M3-C3 `update_parfile::update_pot` deep-copies the base (was shallow)
- 2026-07-02 M3-C1 `carp_io` `readParseElem`/`loadCarpMesh`, element type now detected from the `.elem` file
