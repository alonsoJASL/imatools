# imatools

Medical image analysis and mesh handling tools for NIfTI, DICOM, and VTK formats.
Maintained by the [José Alonso Solís-Lemus](https://alonsojasl.github.io/), 
used by the [CEMRG](www.cemrg.com).

---

## What this library does

`imatools` is a **layered toolbox** for cardiac image and mesh processing. It pairs a
set of pure, stateless workflows with eight command-line tools. It handles:

- Segmentation label operations (extract, mask, morph, combine, connected components)
- Mesh conversion and analysis (VTK ⇄ CARP, surface area / volume, scalar projection, mapping)
- Scar and fibrosis quantification (LGE thresholding, scar projection, fibrosis overlap, scar scoring)
- DICOM handling (patient tables, slice stacking) and NIfTI test-image generation
- Cohort comparisons and mesh-quality PDF reports / VTK rendering

---

## Where to go next

- **[Getting Started](getting-started/index.md)** — install the library and run your first command.
- **[Architecture](api/overview.md)** — the layered design and how a command flows through it.
- **[API Reference](api/index.md)** — the toolbox map of `core`, `io`, and `contracts`.
- **[CLI Reference](cli/index.md)** — the eight `imatools-*` commands and their subcommands.

---

## Design principles

| Principle | What it means in practice |
|---|---|
| **Layered separation** | `cli/` owns argument parsing, file I/O, and paths; `io/` loads/saves and returns typed contracts; `core/` holds pure, stateless workflows; `contracts/` defines the data passed between them. |
| **Contract-driven interfaces** | Layers exchange explicit dataclass contracts (`contracts/`), not loose dicts or globals. |
| **Stateless core** | `core/` functions take inputs and return outputs with no hidden state or side effects, so they are trivially testable and reusable. |
| **Behaviour preservation** | Every migrated `core`/`io` function is pinned by golden-master characterization tests captured from the pre-refactor code; behaviour cannot drift silently. |
| **CLI owns I/O** | Path construction, overwrite handling, and reading/writing files live only in `cli/`; workflows never touch the filesystem. |

---

## Key domain terms

| Term | Meaning |
|---|---|
| **CARP** (`.pts`/`.elem`) | Cardiac mesh format (points + element connectivity) used by openCARP-family tools. |
| **VTK polydata** | Surface mesh representation (`.vtk`/`.vtp`) used throughout the mesh workflows. |
| **LGE** | Late Gadolinium Enhancement — the MRI contrast used to detect myocardial scar. |
| **Scar projection** | Mapping LGE image intensities onto a mesh surface to quantify scar. |
| **Fibrosis score** | Fraction of a mesh above an intensity threshold; generalised to accept a custom blood-pool mean/stdev by the `imatools-scar score` command. |
| **Threshold method** (`iir`, `msd`) | Rules for turning blood-pool statistics into a scar intensity cutoff (image-intensity-ratio vs. mean-plus-standard-deviations). |
| **Golden master** | A recorded output from the trusted pre-refactor code that a characterization test asserts against, so refactors preserve behaviour. |
| **Contract** | A typed dataclass (in `contracts/`) that carries data between the `io`, `core`, and `cli` layers. |
