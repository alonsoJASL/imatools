# imatools Restructuring Plan

## Current Status: Phase 0 Complete ✅

### Phase 0: Modernize Structure (DONE)
- ✅ Convert to setuptools-based `pyproject.toml`
- ✅ Implement `src/` layout
- ✅ Create CLI entry point framework
- ✅ Set up test infrastructure
- ✅ Update documentation

**Next Step**: Begin Phase 1 implementation

---

## Phase 1: Establish Boundaries (Foundation)

### 1.1 Extract Pure Logic from Utilities
**Goal**: Separate I/O from algorithms

**Actions**:
- [ ] Split `itktools.py` into:
  - [ ] `src/imatools/core/image.py` - Pure transforms (no I/O)
  - [ ] `src/imatools/core/label.py` - Label manipulation
  - [ ] `src/imatools/core/spatial.py` - Coordinate transforms
  
- [ ] Split `vtktools.py` into:
  - [ ] `src/imatools/core/mesh.py` - Pure mesh algorithms
  - [ ] `src/imatools/core/geometry.py` - Geometric calculations
  - [ ] `src/imatools/parsers/dotmesh.py` - Format parsing

**Success Criteria**:
- No `sitk.ReadImage()` or file paths in `core/` modules
- All functions take data structures, not file paths
- Tests can run without filesystem access

### 1.2 Create Data Contracts
**Goal**: Define explicit interfaces between layers

**Actions**:
- [ ] Create `src/imatools/contracts/`
  - [ ] `contracts.py` - `ImageContract` (path, metadata, optional array)
  - [ ] `contracts.py` - `MeshContract` (path, mesh_type, optional polydata)
  - [ ] `operations.py` - `LabelOperationContract`, `TransformContract`
  
**Success Criteria**:
- Contracts use `@dataclass` or `TypedDict`
- All mandatory fields are type-hinted
- Validation logic included in contract classes

### 1.3 Build Thin I/O Layer
**Goal**: Centralize all file operations

**Actions**:
- [ ] Create `src/imatools/io/`
  - [ ] `image_io.py` - All SimpleITK/NRRD/NIfTI operations
  - [ ] `mesh_io.py` - All VTK file operations
  - [ ] `dicom_io.py` - DICOM-specific operations

**Success Criteria**:
- I/O functions accept/return contracts
- No VTK/SimpleITK imports outside `io/` and `core/`
- Error handling centralized

---

## Phase 2: Decompose CLI Scripts (Orchestrators)

### 2.1 Extract CLI Orchestrators
**Actions**:
- [ ] Migrate `calculate_volume.py`:
  - [ ] CLI parser in `src/imatools/cli/volume.py`
  - [ ] Logic in `src/imatools/orchestrators/volume_orchestrator.py`
  
- [ ] Migrate `segmentation_tools.py`:
  - [ ] CLI parser in `src/imatools/cli/segmentation.py`
  - [ ] Logic in `src/imatools/orchestrators/segmentation_orchestrator.py`

### 2.2 Create Logic Engines
**Actions**:
- [ ] `src/imatools/engines/label_engine.py`
- [ ] `src/imatools/engines/mesh_engine.py`
- [ ] `src/imatools/engines/analysis_engine.py`

### 2.3 Separate Concerns
**Success Criteria**:
- Orchestrators: CLI → contracts → engines → results → files
- Engines: Stateless, receive contracts, return contracts
- No `argparse` or file I/O in engine code

---

## Phase 3: Dependency Injection & Testing

### 3.1 Inject Dependencies
**Actions**:
- [ ] Engines receive I/O handlers at initialization
- [ ] Use protocols/ABCs for interface contracts
- [ ] Remove global SimpleITK/VTK imports from engines

### 3.2 Build Test Infrastructure
**Actions**:
- [ ] Create `tests/fixtures/` with mock data
- [ ] Write integration tests for orchestrators
- [ ] Unit tests for pure functions in `core/`

### 3.3 Configuration Layer
**Actions**:
- [ ] `src/imatools/config/defaults.py`
- [ ] `src/imatools/config/logging_config.py`
- [ ] Remove `common/config.py` global patterns

---

## Phase 4: Specialty Domain Separation

### 4.1 Extract Domain Logic
**Actions**:
- [ ] `src/imatools/domains/scar_analysis/`
- [ ] `src/imatools/domains/fibrosis/`
- [ ] `src/imatools/domains/mesh_reports/`

### 4.2 Semantic Mapping Layer
**Actions**:
- [ ] `src/imatools/mappers/label_mapper.py`
- [ ] User labels → functional roles

---

## Phase 5: Clean Public API

### 5.1 Define User-Facing API
**Actions**:
- [ ] Create `src/imatools/api.py`
- [ ] Hide internal modules from direct import
- [ ] Expose only high-level functions

### 5.2 Deprecation Path
**Actions**:
- [ ] Add deprecation warnings to `common/`
- [ ] Maintain shim layer for 1-2 versions
- [ ] Document migration path

---

## Phase 6: Optimize for Users

### 6.1 Performance & Parallelization
**Actions**:
- [ ] Enable batch operations with stateless engines
- [ ] Add `parallel_engine.py` wrapper

### 6.2 Builder Pattern
**Actions**:
- [ ] `src/imatools/builders/workflow_builder.py`
- [ ] Encapsulate common operation chains

---

## Implementation Priority

**Critical Path (Do First)**:
1. ✅ Phase 0: Modernize structure
2. Phase 1.1: Extract pure operations
3. Phase 1.2: Define contracts
4. Phase 1.3: Build I/O layer
5. Phase 2.1: Migrate 1-2 CLI scripts as proof-of-concept

**Medium Priority**:
6. Phase 2: Complete CLI migration
7. Phase 3.1-3.2: Dependency injection + tests

**Lower Priority**:
8. Phase 4: Domain modules
9. Phase 5: Public API
10. Phase 6: Optimization

---

## Development Workflow

For each phase:
1. Create feature branch: `feature/phase-X-Y`
2. Implement changes incrementally
3. Add tests before merging
4. Update documentation
5. Merge to `development` branch
6. Test integration before promoting to `main`

---

## Success Metrics

- [ ] Zero file I/O in `core/` modules
- [ ] All functions use contracts, not primitives
- [ ] 70%+ test coverage
- [ ] All CLI tools migrated to new structure
- [ ] `common/` module deprecated and removed
- [ ] Documentation complete
- [ ] Users can `pip install -e .` and run immediately
