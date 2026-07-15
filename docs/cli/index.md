# CLI Reference

`imatools` installs eight command-line entry points. Each is a subcommand group —
run `imatools-<tool> --help`, or `imatools-<tool> <subcommand> --help`, for the
authoritative flags.

| Tool | Purpose |
|---|---|
| [`imatools-segmentation`](#imatools-segmentation) | Label-map and segmentation operations |
| [`imatools-mesh`](#imatools-mesh) | Mesh conversion, projection, and mapping |
| [`imatools-volume`](#imatools-volume) | Surface area, volume, and label volumes |
| [`imatools-image`](#imatools-image) | Coordinate conversion and test-image generation |
| [`imatools-dicom`](#imatools-dicom) | DICOM patient tables and slice stacking |
| [`imatools-scar`](#imatools-scar) | Scar/LGE thresholding, projection, and scoring |
| [`imatools-comparisons`](#imatools-comparisons) | Cohort comparisons driven by a manifest CSV |
| [`imatools-report`](#imatools-report) | Mesh-quality PDF reports and VTK rendering |

---

## `imatools-segmentation`

Operations on segmentation label maps.

`add` · `cc-extract` · `cc-identify` · `cc-regionprops` · `chain` · `combine` ·
`compare` · `extract-label` · `fill` · `gaps` · `inr` · `largest` · `mask` ·
`merge-labels` · `morph-label` · `op` · `resample` · `sharp-regions` · `show` ·
`show-voxels` · `split-label` · `swap`

```bash
imatools-segmentation extract-label -in image.nii -l 1 -out label1.nii
imatools-segmentation mask -in image.nii -mask mask.nii
```

---

## `imatools-mesh`

| Subcommand | What it does |
|---|---|
| `export` | Export a VTK polydata file to another mesh format |
| `flip-xy` | Flip X/Y coordinates of a VTK polydata mesh |
| `from-dotmesh` | Convert a `.mesh` (dotmesh) file to CARP `.pts`/`.elem` |
| `points-to-image` | Paint image voxels near a point-cloud with a label |
| `project-scalars` | Project scalar arrays onto a mesh |
| `map` | Build a mapping between two meshes |
| `map-stats` | Report statistics from a mapping directory |
| `fibrosis-overlap` | Compare fibrosis overlap between two meshes |

```bash
imatools-mesh export input.vtk -o output.vtp
imatools-mesh from-dotmesh input.mesh -o output
imatools-mesh map -in1 left.vtk -in2 right.vtk -map elem
```

---

## `imatools-volume`

| Subcommand | What it does |
|---|---|
| `mesh-props` | Print surface area and volume of a VTK mesh |
| `label-volumes` | Print or save volumes of labels in a segmentation image |

```bash
imatools-volume mesh-props /path/to/mesh.vtk
imatools-volume label-volumes -in seg.nii
```

---

## `imatools-image`

| Subcommand | What it does |
|---|---|
| `coords-to-index` | Convert world coordinates to image indices |
| `gen-circle` | Generate a circle/sphere test image |
| `gen-cube` | Generate a cube test image |

```bash
imatools-image coords-to-index -im image.nii -xyz points.mps -o indices.txt
imatools-image gen-circle -out circle.nii -r 80 -c 150 150 50
```

---

## `imatools-dicom`

| Subcommand | What it does |
|---|---|
| `patient-table` | Build a patient metadata table from DICOM (`folder` or `single`) |
| `stack-slices` | Stack individual DICOM slices into a volume |

```bash
imatools-dicom patient-table folder -in /data/dicom -out table.csv
imatools-dicom stack-slices -d /data -i slices -o volume.nii.gz
```

---

## `imatools-scar`

| Subcommand | What it does |
|---|---|
| `lge` | Create synthetic LGE test image (prism) |
| `surf` | Create segmentation surface mesh with MIRTK |
| `scar-opts` | Write CEMRG scar-options JSON file |
| `scar` | Run CEMRG `MitkCemrgScarProjectionOptions` |
| `mask` | Apply a mask with a threshold file |
| `vscar-pipeline` | Full ventricular-scar projection pipeline |
| `vscar-scale` | Scale a mesh (e.g. µm → mm) |
| `vscar-deform` | Deform a mesh onto the LGE image with MIRTK |
| `vscar-cog` | Compute cell centres of gravity |
| `vscar-project` | Project scar onto the mesh from a reference image |
| `enhance` | Enhance/debug a scar corridor image (`--jobs N` for parallel) |
| `check` | Check a scar corridor CSV |
| `score` | Report the scar score for a mesh from custom blood-pool stats |

```bash
# Scar score from custom blood-pool statistics (no stats file needed)
imatools-scar score --mesh mesh.vtk --mean-bp 120 --stdev-bp 15 \
  --method iir --value 0.97 1.2 1.32
```

`score` prints a `value / threshold / score` table — one row per `--value`. Use
`--field` to select a named cell array (otherwise the active scalars are used) and
`--scalar-type {cell,point}` to choose the array association.

---

## `imatools-comparisons`

Both subcommands read a **manifest CSV** with columns
`comparison_dir,case_left,case_right`.

| Subcommand | What it does |
|---|---|
| `map-fibres` | Build fibre mappings across the cohort |
| `compare` | Compare fields across the mapped cohort |

```bash
imatools-comparisons map-fibres --manifest manifest.csv -n in -map pts
imatools-comparisons compare --manifest manifest.csv -n lat -f 1
```

---

## `imatools-report`

| Subcommand | What it does |
|---|---|
| `report` | Create a report with images for mesh quality assessment |
| `render-single` | Render a folder of VTK files into a single grid PNG |
| `render-multi` | Render a folder of VTK files into individual PNGs |

```bash
imatools-report report --sims-folder /data/case1 --report-name report.pdf --print-all
imatools-report render-single --base-dir /data/vtks --output grid.png
```
