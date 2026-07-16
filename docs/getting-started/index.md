# Getting Started

## Prerequisites

- Python 3.10+
- Git

## Install

```bash
git clone https://github.com/alonsoJASL/imatools
cd imatools

# Editable install with dev dependencies (pytest, black, ruff, mypy)
pip install -e ".[dev]"
```

## Verify

All eight command-line tools are installed as entry points:

```bash
imatools-segmentation --help
imatools-mesh --help
imatools-volume --help
imatools-image --help
imatools-dicom --help
imatools-scar --help
imatools-comparisons --help
imatools-report --help
```

Run the test suite:

```bash
pytest
```

## Your first command

Print the surface area and volume of a mesh:

```bash
imatools-volume mesh-props /path/to/mesh.vtk
```

Convert a `.mesh` (dotmesh) file to CARP `.pts`/`.elem`:

```bash
imatools-mesh from-dotmesh input.mesh -o output
```

See the **[CLI Reference](../cli/index.md)** for the full command surface, and
**[Architecture](../api/overview.md)** for how a command flows through the layers.
