# imatools

Medical image analysis and mesh handling tools for NIfTI, DICOM, and VTK formats.

## ⚠️ Migration Notice

**Version 0.2.0** introduces breaking changes as we modernize the codebase:
- New `src/` layout structure
- Migration from Poetry to standard `pip install -e .`
- CLI tools now use proper entry points
- Active refactoring to improve architecture

If you need the stable version, use v0.1.x from the `main` branch.

## Quick Install (Development)

### Prerequisites
- Python 3.9+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/alonsoJASL/imatools
cd imatools

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Check CLI tools are available
imatools-volume --help
imatools-segmentation --help
imatools-mesh --help

# Run tests
pytest
```

## Migration Status

### ✅ Completed
- Modern `pyproject.toml` with setuptools
- `src/` layout structure
- CLI entry point framework
- Test infrastructure skeleton

### 🚧 In Progress
- Migrating existing scripts to new CLI structure
- Extracting pure logic from utilities
- Creating data contracts
- Building I/O layer

### 📋 Planned
- Full architecture refactor (see development branch)
- Comprehensive test coverage
- API documentation

## Usage (Legacy Scripts)

During migration, legacy scripts in the old structure remain functional:

```bash
# Example: Calculate volume of a mesh
python -m imatools.calculate_volume /path/to/mesh.vtk
```

(The segmentation tools have been migrated — use the `imatools-segmentation` CLI below.)

> **Migration notice:** `patient_table_from_dicom.py` and `stack_individual_dicoms.py`
> have been removed.  Use the `imatools-dicom` subcommands instead:
>
> ```bash
> # Was: python -m imatools.patient_table_from_dicom folder -in /data/dicom -out table.csv
> imatools-dicom patient-table folder -in /data/dicom -out table.csv
>
> # Was: python -m imatools.patient_table_from_dicom single -in /data/dicom/file.dcm
> imatools-dicom patient-table single -in /data/dicom/file.dcm
>
> # Was: python -m imatools.stack_individual_dicoms -d /data -i slices -o volume.nii.gz
> imatools-dicom stack-slices -d /data -i slices -o volume.nii.gz
> ```

> **Migration notice:** `create_mapping.py` and `compare_fibrosis_overlap.py` have been
> removed.  Use the `imatools-mesh` subcommands below instead:
>
> ```bash
> # Was: python -m imatools.create_mapping map -in1 left.vtk -in2 right.vtk -map elem
> imatools-mesh map -in1 left.vtk -in2 right.vtk -map elem
>
> # Was: python -m imatools.create_mapping compare -m MAPPING/
> imatools-mesh map-stats -m MAPPING/
>
> # Was: python -m imatools.compare_fibrosis_overlap -d /data -imsh0 msh0 -imsh1 msh1 -t0 0.5 -t1 0.5
> imatools-mesh fibrosis-overlap -d /data -imsh0 msh0 -imsh1 msh1 -t0 0.5 -t1 0.5
> ```

> **Migration notice:** `scarq_tools.py`, `vscar_projection.py`, `enhance_debug_scar.py`,
> `pool_enhance_debug_scar.py`, `scr_check_scar.py`, and `common/scarqtools.py` have been
> removed.  Use the `imatools-scar` subcommands instead:
>
> ```bash
> # Was: python scarq_tools.py lge --base-dir /data --lge-method iir
> imatools-scar lge --base-dir /data --lge-method iir
>
> # Was: python scarq_tools.py surf --base-dir /data --input PVeinsCroppedImage.nii
> imatools-scar surf --base-dir /data -i PVeinsCroppedImage.nii
>
> # Was: python scarq_tools.py scar_opts --base-dir /data --input options.json
> imatools-scar scar-opts --base-dir /data -i options.json
>
> # Was: python scarq_tools.py scar --base-dir /data --input dcm-LGE.nii --scar-opts options.json
> imatools-scar scar --base-dir /data -i dcm-LGE.nii --scar-opts options.json
>
> # Was: python scarq_tools.py mask --base-dir /data --input lge.nii --mask DebugScar.nii --mask-threshold-file prodStats.txt
> imatools-scar mask --base-dir /data -i lge.nii --mask DebugScar.nii --mask-threshold-file prodStats.txt
>
> # Was: python vscar_projection.py pipeline --input msh.vtk -mirtk /opt/mirtk -moving cine.nii -fixed lge.nii -ref seg.nii -label 3
> imatools-scar vscar-pipeline --input msh.vtk --path-to-mirtk /opt/mirtk --path-to-moving cine.nii --path-to-fixed lge.nii --reference-image seg.nii --label 3
>
> # Was: python vscar_projection.py scale --input msh.vtk --scale 0.001
> imatools-scar vscar-scale --input msh.vtk --scale 0.001
>
> # Was: python vscar_projection.py deform --input msh_mm.vtk -mirtk /opt/mirtk -moving cine.nii -fixed lge.nii
> imatools-scar vscar-deform --input msh_mm.vtk --path-to-mirtk /opt/mirtk --path-to-moving cine.nii --path-to-fixed lge.nii
>
> # Was: python vscar_projection.py cog --input msh_mm_on_LGE.vtk
> imatools-scar vscar-cog --input msh_mm_on_LGE.vtk
>
> # Was: python vscar_projection.py scar --input cog.pts -ref seg.nii -label 3
> imatools-scar vscar-project --input cog.pts --reference-image seg.nii --label 3
>
> # Was: python enhance_debug_scar.py --input lge.nii --scar-corridor-image scar.nii --image-info-file prodStats.txt -m iir --threshold 0.97 1.2 1.32
> # Was: python pool_enhance_debug_scar.py (same args, parallel)
> imatools-scar enhance --input lge.nii --scar-corridor-image scar.nii --image-info-file prodStats.txt -m iir --threshold 0.97 1.2 1.32 [--jobs N]
>
> # Was: python scr_check_scar.py -d /data -i scar_corridor.csv
> imatools-scar check -d /data -i scar_corridor.csv
> ```

> **Migration notice:** `create_mapping_fibres.py` and `compare_from_mapping.py` have been
> removed.  Use the `imatools-comparisons` subcommands instead (both now read a manifest
> CSV — columns `comparison_dir,case_left,case_right` — instead of the hardcoded
> `011_comparisons/C0..C49` tree):
>
> ```bash
> # Was: python create_mapping_fibres.py -d /data -n in -map pts
> imatools-comparisons map-fibres --manifest manifest.csv -n in -map pts
>
> # Was: python compare_from_mapping.py -d /data -n lat -f 1
> imatools-comparisons compare --manifest manifest.csv -n lat -f 1
> ```

> **Migration notice:** `mesh_report.py` and `vtk2png.py` have been removed.  Use the
> `imatools-report` subcommands instead:
>
> ```bash
> # Was: python mesh_report.py --sims_folder /data/case1 --report_name report.pdf --print_all
> imatools-report report --sims-folder /data/case1 --report-name report.pdf --print-all
>
> # Was: python vtk2png.py single --base-dir /data/vtks --output grid.png
> imatools-report render-single --base-dir /data/vtks --output grid.png
>
> # Was: python vtk2png.py multi --base-dir /data/vtks --output frame.png
> imatools-report render-multi --base-dir /data/vtks --output frame.png
> ```

> **Migration notice:** `convert_coords_to_index.py` and `create_test_data.py` have been
> removed.  Use the `imatools-image` subcommands instead:
>
> ```bash
> # Was: python convert_coords_to_index.py -im image.nii -xyz points.mps -o indices.txt
> imatools-image coords-to-index -im image.nii -xyz points.mps -o indices.txt
>
> # Was: python create_test_data.py circle -out circle.nii -r 80 -c 150 150 50
> imatools-image gen-circle -out circle.nii -r 80 -c 150 150 50
>
> # Was: python create_test_data.py cube -out cube.nii -s 80 -c 150 150 50
> imatools-image gen-cube -out cube.nii -s 80 -c 150 150 50
> ```

## New CLI (Post-Migration)

The new CLI structure provides:

```bash
# Volume calculations
imatools-volume /path/to/mesh.vtk

# Segmentation operations
imatools-segmentation extract-label -in image.nii -l 1 -out label1.nii
imatools-segmentation mask -in image.nii -mask mask.nii

# Mesh operations
imatools-mesh convert input.mesh -o output.vtk
imatools-mesh report mesh.vtk
```

## Development Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Dependencies

Core processing libraries:
- **VTK** (≥9.2.6): Mesh processing
- **SimpleITK** (≥2.2.1): Medical image I/O and processing
- **ITK** (≥5.3.0): Advanced image processing
- **nibabel** (≥5.1.0): NIfTI support
- **pydicom** (≥2.4.1): DICOM support

Visualization and analysis:
- **PyVista** (≥0.43.5): 3D visualization
- **matplotlib**, **seaborn**: Plotting
- **pandas**, **numpy**, **scipy**: Data processing

## Architecture Principles

This project follows the **Developer's Manifest** principles:
1. Radical separation of orchestration and logic
2. Contract-driven interfaces
3. Stateless engine design
4. No global state or singletons
5. Explicit dependency injection

See `developers_manifest.md` for details.

## License

MIT License - see `LICENSE` file for details.

## Contributing

Contributions welcome! This is an active refactoring effort. Please:
1. Check existing issues and PRs
2. Follow the architecture principles in the manifest
3. Add tests for new functionality
4. Run formatters and linters before committing

## Support

For questions or issues:
- GitHub Issues: https://github.com/alonsoJASL/imatools/issues
- Email: j.solis-lemus@imperial.ac.uk