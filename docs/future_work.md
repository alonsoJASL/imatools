# imatools — Future Work

Items intentionally parked out of the master→development reconciliation refactor.
Each is something to *design*, not a behaviour to *preserve* (so it is excluded
from the golden-master characterization net).
---

## SegmentationGraph — spectral smoothing of label maps (never working)

**Source:** `imatools/common/SegmentationGraph.py` (master). Pulled from the refactor
plan at Jose's request — it never worked properly, so there is no behaviour to
characterize. Outlined here for a possible future implementation.

### Intended use (as the code reads)
Treat a segmentation image as a graph (one node per voxel, edges to neighbouring
voxels), build the **graph Laplacian**, take its eigendecomposition, and use the
low-frequency eigenvectors to **spectrally smooth** the segmentation — i.e.
denoise/regularize a label map using the geometry of the voxel grid rather than a
fixed convolution kernel.

Public surface as written:
- `SegmentationGraph(im: sitk.Image)` — builds the voxel graph on construction.
- `get_graph()`, `num_nodes()` — accessors.
- `laplacian()` → `nx.laplacian_matrix(self.graph)`.
- `eigen(k=-1, which_ord='LM')` → `(evals, evects)` via `scipy.sparse.linalg.eigsh`.
- `smooth(k: int)` → reconstruct a smoothed image from the smallest-k eigenvectors.

### Why it does not run (bugs found 2026-06-23)
1. `eigen()` line 39 calls `lap_mat.asftype()` — typo for scipy sparse `.asfptype()`
   ⇒ `AttributeError`. So `eigen()` and everything downstream is dead.
2. `smooth()` line 46 calls `itku.arrayview(...)` — no such function in `itktools`
   (it exposes `array2im` / `imarray` / `imview`) ⇒ `AttributeError`.
3. `smooth()` math is off: `eigenvectors @ eigenvectors.T` is the N×N projection
   matrix, not the image-shaped projection of the label signal onto the low-frequency
   subspace. It would not reshape back to the image.
4. `_create_graph()` adds a self-loop (the all-zero offset) and is O(N·27) in both
   time and memory — a node per voxel makes this intractable for real volumes
   (a 128³ image ≈ 2.1M nodes, ~57M edges).

### If implemented later — sketch
- **Scale:** never build a node-per-voxel graph for full volumes. Either operate on a
  downsampled/ROI image, or on a region-adjacency graph (one node per label/supervoxel)
  instead of per-voxel.
- **Correct spectral smoothing:** flatten the label signal `f` (N-vector), project onto
  the first `k` Laplacian eigenvectors `U_k`: `f_smooth = U_k @ (U_k.T @ f)`, then
  reshape `f_smooth` back to the image grid and re-quantize to labels. (Current code
  computes `U_k @ U_k.T` and never multiplies by `f`.)
- **Fixes:** `.asfptype()`; replace `arrayview` with `array2im(smooth_array, self.image)`.
- **Validate** against a known small case (e.g. a noisy 8³ cube) before trusting it.

**Target home if revived:** `core/graph.py` (a `core/` module reserved for this; not
created by the current refactor). Needs `networkx` + `scipy` as deps.

---

## Wave-2 Cat-B bugs — mostly RESOLVED in M3; one deferred

Three of the four preserved Cat-B bugs were fixed in **M3** (all golden-neutral / assertion-tested,
no golden re-baselined):
- ~~`io/carp_io.py::readParseElem` / `loadCarpMesh` triangle break~~ **DONE (M3-C1)** — element type
  now detected from the `.elem` file; the 2 intent tests flipped green (gate 483→ 0 xfail).
- ~~`core/parfile::update_pot` shallow copy~~ **DONE (M3-C3)** — now `copy.deepcopy`s the base;
  golden-neutral (return value unchanged), input-mutation guard test added.
- ~~`core/spatial::create_normal_vector_for_plane` `angle_rad` unused~~ **DONE (M3-C4)** — the code
  already used the raw (radian) `angle` correctly; removed the dead `angle_rad` line. Golden-neutral.

Still deferred (the ONE genuine golden re-baseline in the set):
- **`io/carp_io.py::saveToCarpTxt` — hardcoded element tag (from T1j/T2c2).** Always writes
  `fmt='Tr %d %d %d 1'` (tag = 1) regardless of input. Fix: honour a tag column/arg — this changes
  the written elem lines, so `carp/saveToCarpTxt_elem_lines.json` must be deliberately regenerated
  from the fixed dev output. **Deferred out of M3 (Jose, 2026-07-02):** `saveToCarpTxt` has NO live
  caller (only the deferred `qulati_downsample_pair.py` + its own golden tests), so this would churn
  a golden for an unused-by-live-code function. Do it if/when `saveToCarpTxt` gets a real consumer.

## Characterization-harness flaw to tidy
- **`tests/_golden_cases/paths.py` shares a module-level `_TMPDIR` across cases (from T2c3).** The
  `mkdirplus` case creates `newdir/` before the `get_subfolders` case runs, so the `get_subfolders`
  golden (`paths/get_subfolders_sorted.json`) baked in the `newdir` artifact (`["alpha","beta","newdir"]`),
  and the dev test deliberately reproduces it. Cleaner: give the `mkdirplus` case its own isolated temp
  dir. **Deferred out of M3 (2026-07-02):** the proper fix regenerates `get_subfolders_sorted` to
  `["alpha","beta"]` — a golden re-baseline, which M3 was scoped NOT to do. Master-faithful (the
  function behaviour is unchanged; only the test fixture is cleaned), so safe to do in a later pass.

## Cat-A fixes shipped with no golden — validate against real data (from M3 P1 / Track D)
Master crashed on these paths, so there is no golden and no way to auto-verify — validation needs a
real input file + a domain eye on the output. Deferred out of M3 (need real data).
- **`io/image_io.py::load_nrrd_image` (was P1.1).** The `SetSpacing` 3×3 → per-axis row-norms fix is a
  structural stub (master passed the whole `'space directions'` matrix to `SetSpacing`, which crashed).
  Validate against a real NRRD with non-trivial space directions.
- **`core/spatial.py::create_image_at_plane_from_vector` (was P1.2).** The `_rotation_z_to_vector`
  Rodrigues construction replaced master's always-crashing 3-element `SetMatrix` call. Only known to
  return a 2-D ndarray structurally; validate against a real oblique-plane slice case.

## API naming cleanups (surfaced during M1 — cli migration)

These are clarity refactors of golden-locked `core/` functions. They preserve behaviour;
the golden net stays the verification (re-run the gate, do NOT re-baseline). Tracked as
`BACKLOG.md` M3 §P4.

- ~~**`core/label.py::relabel_image` is misnamed**~~ **DONE (M3-B1).** Unified into
  `binarise(image, background=0, foreground=1)`; `relabel_image` is now a deprecated alias
  delegating to it. The dtype trap was handled by widening the output dtype from uint8 to fit
  `foreground` (so `foreground > 255` no longer overflows — the latent bug the
  `cli/segmentation collapse-to-value` handler could hit). Golden-neutral (the label golden
  reduction is `tolist()`, dtype-agnostic); alias-equivalence + overflow tests added.

## `io/paths.py` rework — drop the MATLAB-era path helpers for `pathlib.Path` (Jose, 2026-07-15)

**Jose's call:** `fullfile` is a port of MATLAB's `fullfile` from his MATLAB days. It has no
business in a modern Python package — `pathlib.Path` does this properly. The whole `io/paths.py`
surface wants a rework, not just a rename.

- **`io/paths.py::fullfile(*paths)`** is literally `"/".join(paths)`. Problems: hardcoded `/`
  separator (not platform-aware — ironic, given the codebase also carries `chooseplatform`), no
  normalization (a trailing slash in any part yields `a//b`), no `..`/symlink resolution, and it
  silently accepts non-str parts only to fail deep in `join`. `Path(a, b, c)` handles all of it.
- **Footprint (updated 2026-07-15, after the `carp_io` pass):** `fullfile` now has only **two**
  remaining users:
  1. **`io/paths.py` itself** — `mkdirplus` (line ~95) and `searchFileByType` (line ~104). This is
     the golden-locked part and the whole of the remaining work.
  2. **`qulati_downsample_pair.py`** (4 refs) — parked behind an `ImportError` guard; dead code,
     ignore it.
- **DONE — `io/carp_io.py` no longer needs it** (@`9912d7c`). `loadCarpMesh` was its only caller
  (2 refs); it now builds paths with `Path`. That also let the module's `_paths()` lazy-import
  accessor be **deleted** — its justifying comment cited a circular import via the `common.ioutils`
  shim that M2 had already removed, i.e. the workaround outlived its problem by two milestones.
  `carp_io` now has **no dependency on `io/paths` at all**. Golden-neutral: the path string never
  escapes `loadCarpMesh` (it returns arrays), so the join method is invisible to callers —
  equivalence was verified across 8 edge cases (all resolve to the same file; `Path` only
  normalizes `//` and `./`).
- **The catch — the remainder IS golden-locked.** `tests/test_io_paths.py::test_fullfile_3parts`
  asserts the returned *string*. Unlike the `carp_io` case, `fullfile`'s output **is** its return
  value, so `Path` normalization (collapsing `//`) can change it. That makes this a **deliberate
  golden re-baseline**, not a silent swap — the one case where re-capturing is legitimate, and it
  must be an explicit, reviewed decision (see the standing "never re-baseline to make a test pass"
  rule).
- **Suggested shape:** repoint `mkdirplus`/`searchFileByType` to `Path` directly; keep `fullfile`
  as a thin deprecated shim (`return str(Path(*paths))`) for one release, then delete. Re-baseline
  its golden in the same commit, with the diff explained. Follow the pattern established in
  `carp_io`: **write the test net first** (the `loadCarpMesh` directory branch had zero coverage
  and was refactored blind until a test was added — @`7913443`).
- **Note:** the deleted pre-0.2.0 notebooks called `fullfile` **200 times** — by far their most-used
  function. It was house style, not a real need. The replacement tutorial notebooks must use
  `pathlib` and must not teach `fullfile`.

## Deferred general capabilities (surfaced during M1.6)
- **`point2cell` / `cell2point` for `imatools-mesh` (from M1.6 scar audit, 2026-06-29).** `scarq_tools.py`'s
  `point2cell`/`cell2point` modes call `vtku.exchange_point_data_to_cell_data`, `exchange_cell_data_to_point_data`,
  `save_vtk`, `vtku.read`, `vtku.write` — **none of these functions exist** in master or refactor, so the modes
  have never run (instant `AttributeError`). There is nothing to migrate. Jose's call (2026-06-29): **defer** —
  if the capability is wanted, implement it FRESH as general `imatools-mesh` subcommands over VTK's
  `vtkPointDataToCellData` / `vtkCellDataToPointData` filters (new product work, golden-backed against a small
  synthetic mesh). The dead modes are simply NOT carried forward when `scarq_tools.py` is deleted in M1.6c.

- **Promote generic mesh ops out of `imatools-scar` → `imatools-mesh` (from M1.6c, Jose 2026-06-29).** The
  `vscar-*` subcommands in `cli/scar.py` include genuinely general ops mis-shelved under "scar":
  - **`vscar-cog`** — its handler (`execute_cog_mesh`) is a thin wrapper over `core/mesh.cogs_from_ugrid`
    (`core/mesh.py:333`, already general/migrated): read mesh → cogs → `np.savetxt`. Promote to a general
    `imatools-mesh cog` subcommand reusing that core fn; optionally keep `vscar-cog` as a thin alias. Trivial,
    non-breaking (core logic already in the right layer — only CLI placement is scar-bound).
  - **`vscar-scale`** (meshtool convert/scale) and **`vscar-deform`** (MIRTK register/transform) — also not
    scar-specific, but they shell out to external binaries; Jose's M1.6 **Q2** decision left them in the scar
    pipeline for now. Lower priority than `cog`.
  Jose's call (2026-06-29): **leave as-is for now**, do it in a dedicated "promote generic mesh ops" pass (e.g.
  alongside the deferred `point2cell`/`cell2point` `imatools-mesh` work above, or during M3 quality). Low-cost to
  defer — CLI-surface only, no behaviour change, no lock-in.

## Intentional master-vs-dev deviations (Wave 2)
- **`io/parfile_io.py::load_from_par` reads with `encoding="utf-8"`** (T2e) — master uses bare `open(filename, 'r')`
  (locale-default encoding). Under the test suite's VTK/ITK-forced ASCII locale, master's version crashes with
  `UnicodeDecodeError` on a UTF-8 em-dash in a `.par` comment line. The explicit encoding is a Cat-A-style env-crash
  fix and is golden-neutral (the comment line is skipped by the parser). `save_pot` still uses bare `open(..., 'w')`
  (writes ASCII pot content). If exact master parity is ever required, revert + make the fixture ASCII instead.
