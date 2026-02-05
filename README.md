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

# Example: Extract label from segmentation
python -m imatools.segmentation_tools extract -in image.nii -l 1
```

## New CLI (Post-Migration)

The new CLI structure will provide:

```bash
# Volume calculations
imatools-volume /path/to/mesh.vtk

# Segmentation operations
imatools-segmentation extract -in image.nii -l 1 -out label1.nii
imatools-segmentation mask -in image.nii -in2 mask.nii

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