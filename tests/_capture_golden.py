#!/usr/bin/env python3
"""
Golden-master capture harness — records master's behaviour as the test contract.

This is the **orchestrator**: it discovers capture cases, runs each function from the
master worktree against the shared synthetic fixtures, and serializes the output into
``tests/golden/``. It owns I/O, paths, and serialization; the cases own the data
(which function, which inputs, how to reduce the output to something serializable).

Run it in the MASTER conda env with BOTH master path entries (master's submodules use
``imatools.common.*`` while its top-level scripts use bare ``common.*``)::

    ~/opt/anaconda3/bin/conda run -n imatools env \\
        PYTHONPATH=~/dev/python/imatools.worktrees/master:~/dev/python/imatools.worktrees/master/imatools \\
        python tests/_capture_golden.py --out tests/golden

Cases live in ``tests/_golden_cases/<group>.py``, each exposing ``CASES: list[CaptureCase]``.
That split lets the parallel Wave-1 characterization tasks add cases in disjoint files
without touching this harness. ``--module <group>`` captures a single group; ``--list``
prints what would be captured without running anything.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

# Make this directory importable so ``import _golden_cases`` and ``import _fixtures``
# resolve whether invoked as ``python tests/_capture_golden.py`` or ``-m``.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


def _identity(value: Any) -> Any:
    return value


@dataclass(frozen=True)
class CaptureCase:
    """One unit of behaviour to record.

    Attributes:
        name:   path stem under ``golden/`` (e.g. ``"metrics/performance_balanced"``);
                forward slashes create subdirectories.
        func:   the master function under test (resolved by the case module).
        args:   positional arguments.
        kwargs: keyword arguments.
        reduce: maps ``func(*args, **kwargs)`` to a serializable value (arrays, dicts,
                scalars). Defaults to identity for already-serializable returns.
        fmt:    ``"json"`` (dicts/scalars/lists) or ``"npy"`` (ndarray).
    """

    name: str
    func: Callable[..., Any]
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    reduce: Callable[[Any], Any] = _identity
    fmt: str = "json"


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that understands the numpy scalar/array zoo."""

    def default(self, obj):  # noqa: N802 (json API name)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def discover_cases(only_module: str | None = None) -> List[CaptureCase]:
    """Import every ``_golden_cases`` submodule and collect its ``CASES`` list."""
    import _golden_cases  # noqa: WPS433 (local import; only valid on sys.path)

    cases: List[CaptureCase] = []
    for info in pkgutil.iter_modules(_golden_cases.__path__):
        if only_module and info.name != only_module:
            continue
        module = importlib.import_module(f"_golden_cases.{info.name}")
        module_cases = getattr(module, "CASES", None)
        if not module_cases:
            continue
        cases.extend(module_cases)
    return cases


def _serialize(value: Any, out_dir: Path, name: str, fmt: str) -> Path:
    target = out_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        path = target.with_suffix(".npy")
        np.save(path, np.asarray(value))
    elif fmt == "json":
        path = target.with_suffix(".json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(value, fh, cls=_NumpyEncoder, indent=2, sort_keys=True)
            fh.write("\n")
    else:
        raise ValueError(f"unknown serialization format: {fmt!r}")
    return path


def capture(cases: List[CaptureCase], out_dir: Path) -> List[Path]:
    """Run each case and write its golden file; return the written paths."""
    written: List[Path] = []
    for case in cases:
        result = case.func(*case.args, **case.kwargs)
        value = case.reduce(result)
        path = _serialize(value, out_dir, case.name, case.fmt)
        written.append(path)
        print(f"  captured {case.name} -> {path.relative_to(out_dir.parent)}")
    return written


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture golden-master values from master.")
    parser.add_argument(
        "--out",
        default=str(_THIS_DIR / "golden"),
        help="Output directory for golden files (default: tests/golden).",
    )
    parser.add_argument(
        "--module",
        default=None,
        help="Capture only this case group (a file stem under tests/_golden_cases/).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the cases that would be captured, then exit.",
    )
    args = parser.parse_args(argv)

    cases = discover_cases(only_module=args.module)
    if not cases:
        print("No capture cases found.", file=sys.stderr)
        return 1

    if args.list:
        for case in cases:
            print(f"{case.name}  [{case.fmt}]  <- {case.func.__module__}.{case.func.__name__}")
        return 0

    out_dir = Path(args.out).resolve()
    print(f"Capturing {len(cases)} case(s) to {out_dir}")
    written = capture(cases, out_dir)
    print(f"Done: {len(written)} golden file(s) written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
