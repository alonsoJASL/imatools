# Upgrading from 0.1.x

Version 0.2.0 replaced the flat top-level scripts with eight `imatools-*` command-line
tools. **The old scripts were removed** — `python -m imatools.<script>` no longer works.

Everything they did is still here; it moved behind a CLI. This page maps every old
command to its replacement.

!!! info "The behaviour is the same"
    The refactor was verified against the 0.1.x code by a golden-master test suite, so
    the commands below produce the same results. Where behaviour deliberately changed,
    it is called out.

!!! tip "Still need the old code?"
    It is preserved on the [`legacy`](https://github.com/alonsoJASL/imatools/tree/legacy)
    branch and pinned by the `v0.1.x-final` tag.

## At a glance

| I used to run | Now I run |
|---|---|
| `patient_table_from_dicom.py`, `stack_individual_dicoms.py` | `imatools-dicom` |
| `create_mapping.py`, `compare_fibrosis_overlap.py` | `imatools-mesh` |
| `scarq_tools.py`, `vscar_projection.py`, `enhance_debug_scar.py`, `pool_enhance_debug_scar.py`, `scr_check_scar.py` | `imatools-scar` |
| `create_mapping_fibres.py`, `compare_from_mapping.py` | `imatools-comparisons` |
| `mesh_report.py`, `vtk2png.py` | `imatools-report` |
| `convert_coords_to_index.py`, `create_test_data.py` | `imatools-image` |
| `calculate_volume.py` | `imatools-volume` |

---

## DICOM

`patient_table_from_dicom.py` and `stack_individual_dicoms.py` → **`imatools-dicom`**

```bash
# I used to:  python -m imatools.patient_table_from_dicom folder -in /data/dicom -out table.csv
imatools-dicom patient-table folder -in /data/dicom -out table.csv

# I used to:  python -m imatools.patient_table_from_dicom single -in /data/dicom/file.dcm
imatools-dicom patient-table single -in /data/dicom/file.dcm

# I used to:  python -m imatools.stack_individual_dicoms -d /data -i slices -o volume.nii.gz
imatools-dicom stack-slices -d /data -i slices -o volume.nii.gz
```

---

## Meshes and mapping

`create_mapping.py` and `compare_fibrosis_overlap.py` → **`imatools-mesh`**

```bash
# I used to:  python -m imatools.create_mapping map -in1 left.vtk -in2 right.vtk -map elem
imatools-mesh map -in1 left.vtk -in2 right.vtk -map elem

# I used to:  python -m imatools.create_mapping compare -m MAPPING/
imatools-mesh map-stats -m MAPPING/

# I used to:  python -m imatools.compare_fibrosis_overlap -d /data -imsh0 msh0 -imsh1 msh1 -t0 0.5 -t1 0.5
imatools-mesh fibrosis-overlap -d /data -imsh0 msh0 -imsh1 msh1 -t0 0.5 -t1 0.5
```

Note `create_mapping compare` became **`map-stats`** — the old name collided with the
unrelated mesh comparison below.

---

## Scar

`scarq_tools.py`, `vscar_projection.py`, `enhance_debug_scar.py`,
`pool_enhance_debug_scar.py`, `scr_check_scar.py` and `common/scarqtools.py` →
**`imatools-scar`**

Six scripts collapsed into one tool. The `vscar_projection.py` subcommands are now
prefixed `vscar-`.

```bash
# I used to:  python scarq_tools.py lge --base-dir /data --lge-method iir
imatools-scar lge --base-dir /data --lge-method iir

# I used to:  python scarq_tools.py surf --base-dir /data --input PVeinsCroppedImage.nii
imatools-scar surf --base-dir /data -i PVeinsCroppedImage.nii

# I used to:  python scarq_tools.py scar_opts --base-dir /data --input options.json
imatools-scar scar-opts --base-dir /data -i options.json

# I used to:  python scarq_tools.py scar --base-dir /data --input dcm-LGE.nii --scar-opts options.json
imatools-scar scar --base-dir /data -i dcm-LGE.nii --scar-opts options.json

# I used to:  python scarq_tools.py mask --base-dir /data --input lge.nii --mask DebugScar.nii --mask-threshold-file prodStats.txt
imatools-scar mask --base-dir /data -i lge.nii --mask DebugScar.nii --mask-threshold-file prodStats.txt
```

### Ventricular scar projection

```bash
# I used to:  python vscar_projection.py pipeline --input msh.vtk -mirtk /opt/mirtk -moving cine.nii -fixed lge.nii -ref seg.nii -label 3
imatools-scar vscar-pipeline --input msh.vtk --path-to-mirtk /opt/mirtk \
    --path-to-moving cine.nii --path-to-fixed lge.nii --reference-image seg.nii --label 3

# I used to:  python vscar_projection.py scale --input msh.vtk --scale 0.001
imatools-scar vscar-scale --input msh.vtk --scale 0.001

# I used to:  python vscar_projection.py deform --input msh_mm.vtk -mirtk /opt/mirtk -moving cine.nii -fixed lge.nii
imatools-scar vscar-deform --input msh_mm.vtk --path-to-mirtk /opt/mirtk \
    --path-to-moving cine.nii --path-to-fixed lge.nii

# I used to:  python vscar_projection.py cog --input msh_mm_on_LGE.vtk
imatools-scar vscar-cog --input msh_mm_on_LGE.vtk

# I used to:  python vscar_projection.py scar --input cog.pts -ref seg.nii -label 3
imatools-scar vscar-project --input cog.pts --reference-image seg.nii --label 3
```

Short flags became explicit long ones: `-mirtk` → `--path-to-mirtk`, `-moving` →
`--path-to-moving`, `-fixed` → `--path-to-fixed`, `-ref` → `--reference-image`.

### Enhance and check

`enhance_debug_scar.py` and its parallel twin `pool_enhance_debug_scar.py` are now one
command — parallelism is a flag, not a separate script.

```bash
# I used to:  python enhance_debug_scar.py --input lge.nii --scar-corridor-image scar.nii --image-info-file prodStats.txt -m iir --threshold 0.97 1.2 1.32
# I used to:  python pool_enhance_debug_scar.py   (same args, parallel)
imatools-scar enhance --input lge.nii --scar-corridor-image scar.nii \
    --image-info-file prodStats.txt -m iir --threshold 0.97 1.2 1.32 [--jobs N]

# I used to:  python scr_check_scar.py -d /data -i scar_corridor.csv
imatools-scar check -d /data -i scar_corridor.csv
```

---

## Cohort comparisons

`create_mapping_fibres.py` and `compare_from_mapping.py` → **`imatools-comparisons`**

!!! warning "This one changed behaviour deliberately"
    Both scripts hardcoded an `011_comparisons/C0..C49` directory tree. They now read a
    **manifest CSV** with columns `comparison_dir,case_left,case_right`, so the tool
    works on any cohort layout rather than one collaborator's folder structure.

```bash
# I used to:  python create_mapping_fibres.py -d /data -n in -map pts
imatools-comparisons map-fibres --manifest manifest.csv -n in -map pts

# I used to:  python compare_from_mapping.py -d /data -n lat -f 1
imatools-comparisons compare --manifest manifest.csv -n lat -f 1
```

---

## Reports and rendering

`mesh_report.py` and `vtk2png.py` → **`imatools-report`**

```bash
# I used to:  python mesh_report.py --sims_folder /data/case1 --report_name report.pdf --print_all
imatools-report report --sims-folder /data/case1 --report-name report.pdf --print-all

# I used to:  python vtk2png.py single --base-dir /data/vtks --output grid.png
imatools-report render-single --base-dir /data/vtks --output grid.png

# I used to:  python vtk2png.py multi --base-dir /data/vtks --output frame.png
imatools-report render-multi --base-dir /data/vtks --output frame.png
```

Underscored flags became hyphenated: `--sims_folder` → `--sims-folder`, `--report_name`
→ `--report-name`, `--print_all` → `--print-all`.

---

## Images

`convert_coords_to_index.py` and `create_test_data.py` → **`imatools-image`**

```bash
# I used to:  python convert_coords_to_index.py -im image.nii -xyz points.mps -o indices.txt
imatools-image coords-to-index -im image.nii -xyz points.mps -o indices.txt

# I used to:  python create_test_data.py circle -out circle.nii -r 80 -c 150 150 50
imatools-image gen-circle -out circle.nii -r 80 -c 150 150 50

# I used to:  python create_test_data.py cube -out cube.nii -s 80 -c 150 150 50
imatools-image gen-cube -out cube.nii -s 80 -c 150 150 50
```

---

## Volumes

```bash
# I used to:  python -m imatools.calculate_volume /path/to/mesh.vtk
imatools-volume mesh-props /path/to/mesh.vtk
```

---

## Importing the library

If you imported `imatools` rather than calling scripts, the package is now layered.
`imatools.common` is gone:

| I used to import | Now I import |
|---|---|
| `from imatools.common import vtktools` | `from imatools.io import mesh_io` / `from imatools.core import mesh` |
| `from imatools.common import ioutils` | `from imatools.io import carp_io` / `from imatools.io import paths` |
| `from imatools.common import itktools` | `from imatools.core import image` / `from imatools.core import label` |

See [Architecture](../api/overview.md) for the layer map and [API Reference](../api/index.md)
for where each function landed. The [Tutorials](../tutorials/01_mesh_io.ipynb) walk through
the new API from scratch.
