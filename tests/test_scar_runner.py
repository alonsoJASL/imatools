"""M1.6b/c — command-assembly tests for the pycemrg CommandRunner swap and
scar CLI orchestration.

These assert that external commands are assembled into the correct no-shell
token Sequence (the critical arg-flattening), WITHOUT executing any binary —
MIRTK/CEMRG/meshtool are absent in CI, so there is no golden here.

M1.6b tests (previously importing vscar_projection and scarqtools) are now
retargeted to cli.scar where those functions live after the M1.6c consolidation.
"""

import os

import pytest

from imatools.cli import scar as cli_scar


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
    """cli.scar.execute_scale_mesh builds the correct meshtool token Sequence."""
    monkeypatch.setattr(cli_scar, "CommandRunner", _FakeRunner)
    info = {"dirname": str(tmp_path), "name": "msh", "ext": "vtk", "base": "msh.vtk"}

    cli_scar.execute_scale_mesh(info, scale=0.001, convert_format=False)

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
    """cli.scar.execute_deform_mesh flattens all MIRTK flags into separate tokens."""
    monkeypatch.setattr(cli_scar, "CommandRunner", _FakeRunner)
    # Dummy MIRTK executables so the existence check passes (never run).
    mirtk = tmp_path / "mirtk"
    mirtk.mkdir()
    for exe in ("register", "transform-points"):
        (mirtk / exe).write_text("")
    info = {"dirname": str(tmp_path), "name": "msh_mm", "ext": "vtk", "base": "msh_mm.vtk"}

    cli_scar.execute_deform_mesh(info, str(mirtk), "moving.nii", "fixed.nii")

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
    """cli.scar._run_cmd in debug mode assembles correct command string."""
    status, cmd_str = cli_scar._run_cmd(
        "/opt/mirtk", "close-image", ["a.nii", "b.nii", "-iterations", "1"], debug=True
    )
    assert status == 0
    assert cmd_str == "/opt/mirtk/close-image a.nii b.nii -iterations 1"


# ---------------------------------------------------------------------------
# M1.6c — scar CLI command-assembly tests
# ---------------------------------------------------------------------------


def test_scar_cli_run_cmd_debug_assembles_tokens(tmp_path):
    """_run_cmd in debug mode assembles correct no-shell token sequence."""
    status, cmd_str = cli_scar._run_cmd(
        "/opt/cemrg",
        "MitkCemrgScarProjectionOptions",
        ["-i", "lge.nii", "-seg", "PVeinsCroppedImage.nii", "-opts", "options.json"],
        debug=True,
    )
    assert status == 0
    assert "MitkCemrgScarProjectionOptions" in cmd_str
    assert "-i lge.nii" in cmd_str
    assert "-opts options.json" in cmd_str


def test_scar_cli_create_segmentation_mesh_assembles_mirtk_tokens(tmp_path, monkeypatch):
    """_create_segmentation_mesh debug=True logs all three MIRTK commands."""
    calls = []

    def _fake_run_cmd(script_dir, cmd_name, arguments, debug=False):
        calls.append((cmd_name, list(arguments)))
        return 0, " ".join([script_dir, cmd_name] + arguments)

    monkeypatch.setattr(cli_scar, "_run_cmd", _fake_run_cmd)

    cli_scar._create_segmentation_mesh(
        mirtk_dir="/opt/mirtk",
        work_dir=str(tmp_path),
        pveins_file="PVeinsCroppedImage.nii",
        iterations=1,
        isovalue=0.5,
        blur=0.0,
        debug=True,
    )

    assert len(calls) == 3
    cmd_names = [c[0] for c in calls]
    assert cmd_names == ["close-image", "extract-surface", "smooth-surface"]
    # close-image should reference the pveins file
    assert any("PVeinsCroppedImage.nii" in str(a) for a in calls[0][1])
    # extract-surface should produce segmentation.vtk
    assert any("segmentation.vtk" in str(a) for a in calls[1][1])


def test_scar_config_round_trip(tmp_path):
    """ScarConfig saves to and loads from a state JSON correctly."""
    cfg = cli_scar.ScarConfig(
        cemrg_dir="/opt/cemrg",
        mirtk_dir="/opt/mirtk",
        scar_cmd_name="MitkCemrgScarProjectionOptions",
        clip_cmd_name="MitkCemrgApplyExternalClippers",
    )
    state_path = str(tmp_path / "state.json")
    cli_scar.scar_io.save_scar_state(state_path, cfg.to_state_dict())

    loaded = cli_scar.ScarConfig.load(state_path)
    assert loaded.cemrg_dir == "/opt/cemrg"
    assert loaded.mirtk_dir == "/opt/mirtk"
    assert loaded.scar_cmd_name == "MitkCemrgScarProjectionOptions"
    assert loaded.clip_cmd_name == "MitkCemrgApplyExternalClippers"
