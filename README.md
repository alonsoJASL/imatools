# imatools

Medical image analysis and mesh handling tools for NIfTI, DICOM, and VTK formats.

**[Documentation](https://alonsojasl.github.io/imatools/)** ·
[Getting started](https://alonsojasl.github.io/imatools/getting-started/) ·
[Tutorials](https://alonsojasl.github.io/imatools/tutorials/01_mesh_io/) ·
[CLI reference](https://alonsojasl.github.io/imatools/cli/)

## What's new in 0.2.0

**Version 0.2.0** is a full modernization of the codebase, now complete:
- New `src/` layout with a layered `core/ io/ contracts/ cli/` architecture
- Standard `pip install -e .` (no more Poetry)
- Eight `imatools-*` CLI entry points replacing the old flat scripts
- Behaviour verified against the 0.1.x code by a golden-master characterization test net

## Quick Install (Development)

### Prerequisites
- Python 3.10+
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
imatools-segmentation --help
imatools-mesh --help
imatools-volume --help
imatools-image --help
imatools-dicom --help
imatools-scar --help
imatools-comparisons --help
imatools-report --help

# Run tests
pytest
```

## Architecture

The package is organized in layers (see the documentation site for the full map):

- **`cli/`** — argument parsing, file I/O, and path construction for the `imatools-*` tools
- **`io/`** — load/save for VTK, CARP, DICOM, NIfTI, and parfiles (returns typed contracts)
- **`core/`** — pure, stateless workflows (mesh, scar, label, segmentation, spatial, ...)
- **`contracts/`** — the data contracts passed between layers

Every `core`/`io` function is characterized by golden-master tests captured from the 0.1.x
code, so behaviour is preserved across the refactor.

## Upgrading from 0.1.x

The old flat scripts (`python -m imatools.<script>`) were **removed** in 0.2.0 and
replaced by the `imatools-*` CLIs. Every old command and its replacement is listed in
the **[upgrade guide](https://alonsojasl.github.io/imatools/getting-started/upgrading/)**.

If you need the pre-refactor code, it lives on the `legacy` branch and is pinned by the
`v0.1.x-final` tag.

## CLI overview

Each tool is a subcommand group; run `imatools-<tool> --help` for the full list.

```bash
# Volume / area
imatools-volume mesh-props /path/to/mesh.vtk         # surface area + volume of a mesh
imatools-volume label-volumes -in seg.nii            # per-label volumes of a segmentation

# Segmentation (extract-label, mask, combine, morph-label, cc-extract, ... — see --help)
imatools-segmentation extract-label -in image.nii -l 1 -out label1.nii
imatools-segmentation mask -in image.nii -mask mask.nii

# Mesh (export, flip-xy, from-dotmesh, points-to-image, project-scalars, map, map-stats, fibrosis-overlap)
imatools-mesh export input.vtk -o output.vtp
imatools-mesh from-dotmesh input.mesh -o output          # -> CARP .pts/.elem

# Scar (lge, surf, scar-opts, scar, mask, vscar-*, enhance, check, score)
imatools-scar score --mesh mesh.vtk --mean-bp 120 --stdev-bp 15 --method iir --value 0.97 1.2 1.32

# Reports / rendering
imatools-report report --sims-folder /data/case1 --report-name report.pdf
```

## Development Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# The gate (mirrored by CI in .github/workflows/tests.yml)
pytest tests/ -q -p no:cacheprovider   # golden-master characterization suite
pytest --nbmake ipynb/                 # tutorial notebooks are executed, not just linted
black --check src/ tests/              # black is PINNED in the dev extra — do not upgrade casually
ruff check src/ tests/                 # advisory: a known baseline exists, do not add to it
```

Behaviour is pinned by golden masters captured from the 0.1.x code. **Never re-capture a
golden to make a test pass** — a golden-backed failure is a bug or a tolerance issue, not
a stale golden. Re-baselining is a deliberate, reviewed decision.

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

## License

MIT License - see `LICENSE` file for details.

## Contributing

Contributions welcome — see **[CONTRIBUTING.md](CONTRIBUTING.md)**.

The short version: branch from `development`, hold the gate before pushing, and **never
re-capture a golden master to make a test pass** — the `legacy` branch is the behaviour
oracle, and a failing golden means your change altered behaviour.

## Support

For questions or issues:
- GitHub Issues: https://github.com/alonsoJASL/imatools/issues
- Email: j.solis-lemus@imperial.ac.uk