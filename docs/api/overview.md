# Architecture

`imatools` is organized as four layers. Data flows **down** from the command line
into pure workflows and **back up** as typed contracts — no layer reaches around
its neighbours.

```
imatools-<tool>  (entry point)
      │
      ▼
   cli/          argument parsing · path construction · overwrite · file I/O
      │  loads via io/*, calls core/*
      ▼
   io/           load/save VTK · CARP · DICOM · NIfTI · parfiles  →  returns contracts
      │
      ▼
   core/         pure, stateless workflows (mesh, scar, label, segmentation, spatial, ...)
      │  exchanges
      ▼
   contracts/    typed dataclasses passed between layers
```

## The layers

| Layer | Responsibility | Rule |
|---|---|---|
| **`cli/`** | Parse arguments, build paths, handle overwrite, read/write files, print output. | The *only* layer allowed to touch the filesystem or `argv`. |
| **`io/`** | Read and write each on-disk format and return a typed contract (`return_contract=` controls raw-vs-contract). | No workflow logic; no path construction. |
| **`core/`** | Pure workflows: geometry, mesh, label, scar, segmentation, spatial, metrics. | No file I/O, no globals, no hidden state. |
| **`contracts/`** | The dataclasses (`image`, `mesh`, `operations`, `report`) exchanged between layers. | Data only — no behaviour. |

Two supporting packages sit alongside: **`parsers/`** (format parsing, e.g. dotmesh)
and **`render/`** (VTK-to-PNG and report views used by `imatools-report`).

## The canonical command flow

Every `imatools-*` command follows the same shape:

1. **`cli/`** parses arguments and resolves input/output paths.
2. It calls **`io/<x>_io.load_*(..., return_contract=False)`** to read inputs.
3. It hands the data to a **`core/<module>`** workflow — pure computation.
4. It writes the result back through **`io/`** / a shared `_write_output()` tail.

## Behaviour preservation

The layered package was migrated from a flat pre-refactor codebase **without
changing behaviour**. Every `core`/`io` function is characterized by golden-master
tests: outputs recorded from the trusted original code, asserted against on every
run. This is why the toolbox can be reorganized freely — the golden net catches
any drift. See [Design principles](../index.md#design-principles).
