"""M1.6b — command-assembly tests for the pycemrg CommandRunner swap.

These assert that external commands are assembled into the correct no-shell
token Sequence (the critical arg-flattening), WITHOUT executing any binary —
MIRTK/CEMRG/meshtool are absent in CI, so there is no golden here.
"""

import os

import pytest

from imatools import vscar_projection as vscar
from imatools.common.scarqtools import ScarQuantificationTools


class _FakeRunner:
    """Records the command Sequences passed to .run instead of executing."""

    calls = []

    def __init__(self, logger=None):
        pass

    def run(self, cmd, expected_outputs=None, cwd=None, ignore_errors=None, env=None):
        _FakeRunner.calls.append(list(cmd))
        return ""


@pytest.fixture(autouse=True)
def _reset_calls():
    _FakeRunner.calls = []
    yield
    _FakeRunner.calls = []


def test_scale_mesh_assembles_meshtool_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(vscar, "CommandRunner", _FakeRunner)
    info = {"dirname": str(tmp_path), "name": "msh", "ext": "vtk", "base": "msh.vtk"}

    vscar.execute_scale_mesh(info, scale=0.001, convert_format=False)

    assert len(_FakeRunner.calls) == 1
    cmd = _FakeRunner.calls[0]
    base = os.path.join(str(tmp_path), "msh")
    assert cmd == [
        "meshtool",
        "convert",
        f"-imsh={base}",
        "-ifmt=vtk",
        "-scale=0.001",
        f"-omsh={base}_mm",
        "-ofmt=vtk",
    ]


def test_deform_mesh_flattens_register_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(vscar, "CommandRunner", _FakeRunner)
    # Dummy MIRTK executables so the existence check passes (never run).
    mirtk = tmp_path / "mirtk"
    mirtk.mkdir()
    for exe in ("register", "transform-points"):
        (mirtk / exe).write_text("")
    info = {"dirname": str(tmp_path), "name": "msh_mm", "ext": "vtk", "base": "msh_mm.vtk"}

    vscar.execute_deform_mesh(info, str(mirtk), "moving.nii", "fixed.nii")

    register_cmd = _FakeRunner.calls[0]
    dof = os.path.join(str(tmp_path), "rigid.dof")
    assert register_cmd == [
        os.path.join(str(mirtk), "register"),
        "moving.nii",
        "fixed.nii",
        "-dofout",
        dof,
        "-model",
        "Rigid",
        "-verbose",
        "3",
    ]
    # No space-joined tokens leaked through.
    assert all(" " not in tok for tok in register_cmd)


def test_run_cmd_debug_assembles_without_executing():
    scarq = ScarQuantificationTools()
    status, cmd_str = scarq.run_cmd(
        "/opt/mirtk", "close-image", ["a.nii", "b.nii", "-iterations", "1"], debug=True
    )
    assert status == 0
    assert cmd_str == "/opt/mirtk/close-image a.nii b.nii -iterations 1"
