# Contributing to imatools

Thanks for helping out. This file covers the things you could not guess from reading the
code. Everything else — be reasonable, write tests, keep it readable — goes without saying.

**If you read only one section, read [The golden-master rule](#the-golden-master-rule).**

## Setup

```bash
git clone https://github.com/alonsoJASL/imatools
cd imatools
pip install -e ".[dev]"
```

Python 3.9+ is supported; CI runs on 3.10. The dependencies are heavy (VTK, ITK, SimpleITK,
PyVista) but all install from wheels — no system packages needed beyond `libgl1` on headless
Linux.

## The gate

Run this before you push. CI runs the same thing on every push to `main`/`development` and
on PRs to `main`:

```bash
pytest tests/ -q -p no:cacheprovider   # the characterization suite
pytest --nbmake ipynb/                 # tutorial notebooks are executed, not just collected
black --check src/ tests/              # must be clean — blocking in CI
ruff check src/ tests/                 # advisory — see below
```

`-p no:cacheprovider` is deliberate: never trust a cached run when the whole point of the
suite is detecting drift.

## The golden-master rule

`imatools` was refactored from a flat package into its current layered form **without
changing behaviour**. That guarantee is enforced by *golden masters*: recorded outputs from
the pre-refactor code, stored in `tests/golden/`, asserted on every run.

> ### Never re-capture a golden to make a test pass.
>
> A failing golden-backed test means **your change altered behaviour**. That is the test
> doing its job. Fix the code.

The oracle is the **`legacy`** branch, pinned by the **`v0.1.x-final`** tag. It is the
original flat implementation and it is frozen. Goldens come from there and nowhere else —
never regenerate one from the current code, which would make the test assert that the code
equals itself.

Two legitimate exceptions, both requiring a deliberate, reviewed decision:

- **Floating-point noise.** A golden-backed test failing at ~1e-16 (e.g. `assert_allclose`
  defaults to `atol=0`) is a **test tolerance** problem, not a golden problem. Fix the
  tolerance, not the golden.
- **An intentional behaviour change.** If you mean to change behaviour, re-baseline the
  golden **in the same commit**, and explain the diff in the message. This is rare. Say so
  loudly in the PR.

If a change cannot preserve behaviour and you are not sure it should, open an issue first.

## Branch model

```
feature/x  ->  development  ->  (release)  ->  main + tag
```

| Branch | What it is |
|---|---|
| `main` | released code, tagged (`v0.2.0`, …). The GitHub default. |
| `development` | integration branch — where work lands |
| `legacy` | the frozen pre-0.2.0 flat implementation. **The behaviour oracle.** Never merge into it. |

- **Branch from `development`, PR into `development`.**
- `main` moves only on release: fast-forward from `development`, then tag.
- `legacy` is read-only history. If you need the old code, check it out or use `v0.1.x-final`.

## Architecture

The package is layered, and the layering is the point — it is what the whole refactor bought.

| Layer | Owns | Never does |
|---|---|---|
| `cli/` | argument parsing, path construction, overwrite handling, reading/writing files | workflow logic |
| `io/` | load/save per format; returns a typed contract | path construction, workflow logic |
| `core/` | pure, stateless workflows | file I/O, globals, hidden state |
| `contracts/` | the dataclasses passed between layers | behaviour |

Rules that follow from that:

- **`core/` must never touch the filesystem.** If a workflow needs a file, the CLI reads it
  and passes the data in. This is why `core/` is testable.
- **Load through `io/`**, typically `io/<x>_io.load_*(..., return_contract=False)` when the
  CLI wants the raw object.
- **No new top-level scripts.** Add a subcommand to an existing `imatools-*` CLI, or a new
  CLI module under `cli/` with an entry point in `pyproject.toml`. The flat-script era is
  over and is not coming back.
- Within a CLI module, follow the local conventions — e.g. `cli/segmentation.py` adds
  `-l/--label` per-subparser via `_add_label()` (never on the shared parent parser) and ends
  handlers with `_write_output()`.

## Renaming public functions

The library is used by external tools (notably CemrgApp), so **a rename is an API break**.
Do not rename in place. Add the new name and keep the old one as a delegating shim:

```python
def get_hausdorff_distance(mesh0, mesh1, label=0):
    ...   # the implementation

def getHausdorffDistance(mesh0, mesh1, label=0):  # noqa: N802
    logger.warning("getHausdorffDistance is deprecated; use get_hausdorff_distance instead")
    return get_hausdorff_distance(mesh0, mesh1, label)
```

The shim delegates — it never duplicates the body. Repoint all in-tree callers to the new
name in the same PR, otherwise the library warns about itself and users see deprecation
notices for code they did not write.

## Style

- **`black` is pinned to an exact version** (`black==26.5.1`) in the `dev` extra, and the
  lint job in `.github/workflows/tests.yml` pins the same one. **Keep the two in sync, and do
  not upgrade casually** — black's stable style changes between releases, and an unpinned
  black silently reformats the entire tree. This has already happened once.
- **`ruff` is advisory in CI** (`continue-on-error`) because the tree carries a known,
  documented baseline. **Do not add to it.** Run `ruff check src/ tests/` and make sure your
  change introduces nothing new. The rule is *0 new findings*, not 0 findings.
- `import SimpleITK as sitk` is the repo-wide idiom (`N813` is ignored project-wide).
- Gotcha: when black wraps a long signature, a trailing `# noqa` ends up on the closing
  paren and **silently stops suppressing**. `noqa` must sit on the line ruff reports.

## Notebooks

`ipynb/` holds the tutorial library. It is documentation *and* a test suite — CI executes
every notebook via `nbmake`, and the site renders them as the Tutorials section.

- **Self-contained, always.** A notebook generates its own data (e.g. `vtkSphereSource`, or
  the synthetic image builders) and writes to a temp directory. **Never reference an external
  data path.** The pre-0.2.0 notebooks all died this way: they hardcoded paths on machines
  that no longer exist, and nothing ran them, so nobody noticed for months.
- **Strip outputs before committing.** CI regenerates them.
- `ipynb/` is excluded from ruff — notebooks are teaching material, not library code.
- If it cannot run in CI, it does not belong in `ipynb/`.

## Documentation

The site is MkDocs Material, deployed to https://alonsojasl.github.io/imatools/ automatically
on pushes to `main` touching `docs/**` or `mkdocs.yml`.

```bash
pip install mkdocs-material mkdocs-jupyter
mkdocs build --strict   # this is the CI gate — it fails on broken internal links
mkdocs serve            # preview locally
```

`--strict` catches broken links, including the easy mistake of writing a bare-domain link
(`[CEMRG](www.cemrg.com)`), which MkDocs reads as an internal page and fails on. Use the
protocol. `docs/tutorials` is a symlink to `../ipynb`; edit the notebooks, not the symlink.

## Parked on purpose

`src/imatools/m3d_parfile.py` and `src/imatools/qulati_downsample_pair.py` raise
`ImportError` at import time **by design**. They depend on things not packaged here
(pycemrg-meshing, quLATi). Their guards explain why. Do not "fix" their imports or un-park
them without an optional-dependency story — open an issue first.

## Pull requests

- Keep unrelated changes in separate commits. Formatting churn especially — a `black` run
  mixed into a behaviour change makes the diff unreviewable.
- Say what you verified. "Gate green (N passed)" is worth more than "should be fine".
- If you changed behaviour deliberately, say so in the PR title and explain the golden diff.

## Questions

Open an issue: https://github.com/alonsoJASL/imatools/issues
