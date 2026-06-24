# Golden-master values

This directory holds the **behaviour contract** for the master→development refactor:
serialized outputs of master's functions, run against the shared synthetic fixtures in
[`tests/_fixtures.py`](../_fixtures.py). Characterization tests on `development` compare the
migrated `core/`/`io/` code against these committed values.

**These files are generated, but committed on purpose** — they are the oracle. Do not
hand-edit them. Regenerate only by deliberately re-running the capture harness against
master (e.g. when adding new cases).

## Layout

```
golden/
  <group>/<case>.npy     # numpy arrays
  <group>/<case>.json    # dicts / scalars / lists  (sorted keys, 2-space indent)
```

The `<group>/<case>` stem matches the `CaptureCase.name` in
[`tests/_golden_cases/<group>.py`](../_golden_cases). Read a value in a test with the
`golden` fixture: `golden("metrics/performance_balanced")`.

## Regenerating

Run in the **master** conda env with BOTH master path entries (master's submodules import
`imatools.common.*`, its top-level scripts use bare `common.*`):

```bash
~/opt/anaconda3/bin/conda run -n imatools env \
  PYTHONPATH=~/dev/python/imatools.worktrees/master:~/dev/python/imatools.worktrees/master/imatools \
  python tests/_capture_golden.py --out tests/golden
```

`--module <group>` captures one group; `--list` shows the cases without running them.

## Adding cases (Wave-1 characterization tasks)

Add a `CaptureCase` to the relevant `tests/_golden_cases/<group>.py` (create the file if
your concern has none yet), re-run the harness for that `--module`, commit the new golden
file alongside your `@pytest.mark.xfail` characterization test. The migration task later
removes the `xfail` and the test must go green.
