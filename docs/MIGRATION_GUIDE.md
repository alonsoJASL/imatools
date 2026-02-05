# Migration Guide: v0.1.x → v0.2.0

## Overview

Version 0.2.0 modernizes imatools with a new structure and installation method. This guide helps existing users migrate.

## Breaking Changes

### 1. Installation Method

**OLD (v0.1.x):**
```bash
conda create -n imatools python=3.10
conda activate imatools
conda install -c conda-forge vtk simpleitk ...
# or
poetry install
```

**NEW (v0.2.0):**
```bash
pip install -e .
# or for development
pip install -e ".[dev]"
```

### 2. Package Structure

**OLD:**
```
imatools/
├── imatools/
│   ├── calculate_volume.py
│   ├── segmentation_tools.py
│   └── common/
│       ├── vtktools.py
│       └── itktools.py
```

**NEW:**
```
imatools/
├── src/
│   └── imatools/
│       ├── cli/
│       ├── core/          # Pure logic (coming soon)
│       ├── io/            # I/O layer (coming soon)
│       └── common/        # Legacy (temporary)
```

### 3. Import Paths

**During transition, old imports still work:**
```python
# These still work (deprecated)
from imatools.common import vtktools
from imatools.common import itktools
```

**Future imports (after full migration):**
```python
# New structure (coming soon)
from imatools.io import image_io
from imatools.core import label_ops
```

### 4. CLI Scripts

**OLD:**
```bash
python calculate_volume.py mesh.vtk
python segmentation_tools.py extract -in image.nii -l 1
```

**NEW (after migration):**
```bash
imatools-volume mesh.vtk
imatools-segmentation extract -in image.nii -l 1
```

**Current (transition):**
Both methods work, but old scripts will be deprecated.

## Step-by-Step Migration

### For Users (Just Running Scripts)

1. **Backup your current environment:**
   ```bash
   conda env export > old_imatools_env.yml
   ```

2. **Clone the development branch:**
   ```bash
   git fetch origin
   git checkout development
   ```

3. **Install the new version:**
   ```bash
   pip install -e .
   ```

4. **Test your workflows:**
   ```bash
   # Old scripts still work during transition
   python -m imatools.calculate_volume mesh.vtk
   ```

5. **Gradually adopt new CLI:**
   ```bash
   imatools-volume --help
   ```

### For Developers (Contributing Code)

1. **Update your development environment:**
   ```bash
   pip install -e ".[dev]"
   ```

2. **Run the test suite:**
   ```bash
   pytest
   ```

3. **Update imports in your code:**
   - Check for deprecation warnings
   - Gradually migrate to new import structure

4. **Follow new architecture:**
   - Read `developers_manifest.md`
   - Separate orchestration from logic
   - Use data contracts

## Compatibility Table

| Feature | v0.1.x | v0.2.0 (Current) | v0.2.0 (Future) |
|---------|--------|------------------|-----------------|
| Poetry install | ✅ | ❌ | ❌ |
| pip install | ❌ | ✅ | ✅ |
| Old script names | ✅ | ✅ (deprecated) | ❌ |
| New CLI entry points | ❌ | 🚧 (partial) | ✅ |
| `common/` imports | ✅ | ✅ (deprecated) | ❌ |
| New structure imports | ❌ | 🚧 (partial) | ✅ |

## Deprecation Timeline

- **v0.2.0 (current)**: Old structure works with deprecation warnings
- **v0.3.0 (planned)**: Old structure removed, migration complete
- **v0.4.0 (planned)**: Stable API with new architecture

## Getting Help

- **Issues**: https://github.com/alonsoJASL/imatools/issues
- **Tag with**: `migration`, `v0.2.0`
- **Email**: j.solis-lemus@imperial.ac.uk

## Rollback Instructions

If you need to return to v0.1.x:

```bash
git checkout main
# Restore old environment
conda env create -f old_imatools_env.yml
conda activate imatools
```
