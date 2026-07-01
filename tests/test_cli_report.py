"""
Tests for the imatools-report CLI (M1.8).

Covers argparse wiring, the `print_any` boolean-bug fix, and that each
`render_<anatomy>_views` composition function no-ops when its required
`MeshReportInputs` field(s) are None. No golden fixtures needed — nothing in
this module was ever characterized (rendering produces images/PDFs).
"""

from unittest.mock import MagicMock, patch

import pytest

from imatools.cli import report
from imatools.contracts.report import MeshReportInputs, RenderParams
from imatools.render import report_views


def test_report_cli_stub():
    """CLI module loads and exposes main()."""
    assert hasattr(report, "main")


def test_build_parser_has_all_subcommands():
    parser = report._build_parser()
    sub_actions = [action for action in parser._actions if action.dest == "command"]
    assert sub_actions
    choices = sub_actions[0].choices
    assert set(choices) == {"report", "render-single", "render-multi"}


def test_main_no_command_prints_help_and_returns_1(capsys):
    ret = report.main([])
    assert ret == 1


def test_report_parser_defaults():
    parser = report._build_parser()
    args = parser.parse_args(["report", "--sims-folder", "/data/case1"])
    assert args.report_name == "output.pdf"
    assert args.fig_w == 2480
    assert args.fig_h == 3508
    assert args.colormap == "RdBu"
    assert args.zoom == 1
    assert args.dpi == 100
    assert args.title_fontsize == 44
    assert args.title_position == 0.9
    assert args.print_whole_mesh is False
    assert args.print_all is False


def test_handle_report_no_print_flags_returns_error(tmp_path):
    """print_any must be False (and handled) when no --print-* flag is set —
    exercises the fixed boolean (the original repeated print_epicardium and
    never checked print_endocardia/print_veins/print_EAS)."""
    parser = report._build_parser()
    args = parser.parse_args(
        ["report", "--sims-folder", str(tmp_path), "--report-name", str(tmp_path / "out.pdf")]
    )
    ret = report.handle_report(args)
    assert ret == 1
    assert not (tmp_path / "out.pdf").exists()


def test_handle_report_endocardia_only_is_recognised_by_print_any(tmp_path, monkeypatch):
    """Regression test for the print_any bug: --print-endocardia alone (with
    no whole-mesh/fibres/etc.) must be treated as "something to print", i.e.
    must NOT early-return with an error. We stub out the mesh-building/PDF
    machinery so this only exercises the print_any gate."""
    parser = report._build_parser()
    args = parser.parse_args(
        [
            "report",
            "--sims-folder",
            str(tmp_path),
            "--report-name",
            str(tmp_path / "out.pdf"),
            "--print-endocardia",
        ]
    )

    # Stub the mesh-building + PDF writing so we don't need real CARP files.
    fake_mesh = MagicMock()
    monkeypatch.setattr(report.carp_io, "read_pts", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr(report.carp_io, "read_elem", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr(report.render, "pts_elem_to_pyvista", lambda **_k: fake_mesh)
    monkeypatch.setattr(report.core_mesh, "rotate_mesh", lambda m, **_k: m)
    monkeypatch.setattr(
        report, "_load_report_inputs", lambda *a, **k: MeshReportInputs(mesh=fake_mesh)
    )
    monkeypatch.setattr(report.render, "render_mesh_views", lambda *a, **k: None)
    monkeypatch.setattr(report.render, "render_fibres_views", lambda *a, **k: None)
    monkeypatch.setattr(report.render, "render_pericardium_views", lambda *a, **k: None)
    monkeypatch.setattr(report.render, "render_epicardium_views", lambda *a, **k: None)
    monkeypatch.setattr(report.render, "render_endocardia_views", lambda *a, **k: None)
    monkeypatch.setattr(report.render, "render_veins_views", lambda *a, **k: None)
    monkeypatch.setattr(report.render, "render_eas_views", lambda *a, **k: None)

    with patch("matplotlib.backends.backend_pdf.PdfPages") as mock_pdf_pages:
        mock_pdf_pages.return_value.__enter__.return_value = MagicMock()
        ret = report.handle_report(args)

    assert ret == 0


def test_load_report_inputs_skips_absent_files(tmp_path):
    """Files that don't exist are skipped (logged), not a crash."""
    fake_mesh = MagicMock()
    inputs = report._load_report_inputs(
        str(tmp_path),
        fake_mesh,
        None,
        print_pericardium=True,
        print_epicardium=True,
        print_endocardia=True,
        print_veins=True,
        print_eas=True,
    )
    assert inputs.mesh is fake_mesh
    assert inputs.pericardium_scale is None
    assert inputs.epicardium_surf is None
    assert inputs.lv_endo_surf is None
    assert inputs.rv_endo_surf is None
    assert inputs.la_endo_surf is None
    assert inputs.ra_endo_surf is None
    assert inputs.rpvs_surf is None
    assert inputs.svc_surf is None
    assert inputs.san_vtx is None
    assert inputs.fascicles_lv_vtx is None
    assert inputs.fascicles_rv_vtx is None


# ---------------------------------------------------------------------------
# render/report_views.py — no-op on absent inputs
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_inputs():
    return MeshReportInputs(mesh=MagicMock())


@pytest.fixture
def params():
    return RenderParams()


def test_render_fibres_views_noop_when_lon_none(empty_inputs, params):
    with patch.object(report_views, "visualise_fibres") as mock_fn:
        report_views.render_fibres_views(empty_inputs, params, pdf=MagicMock())
    mock_fn.assert_not_called()


def test_render_pericardium_views_noop_when_absent(empty_inputs, params):
    with patch.object(report_views, "visualise_pericardium") as mock_fn:
        report_views.render_pericardium_views(empty_inputs, params, pdf=MagicMock())
    mock_fn.assert_not_called()


def test_render_epicardium_views_noop_when_absent(empty_inputs, params):
    with patch.object(report_views, "visualise_mesh") as mock_fn:
        report_views.render_epicardium_views(empty_inputs, params, pdf=MagicMock())
    mock_fn.assert_not_called()


def test_render_endocardia_views_noop_when_all_absent(empty_inputs, params):
    with patch.object(report_views, "visualise_mesh") as mock_fn:
        report_views.render_endocardia_views(empty_inputs, params, pdf=MagicMock())
    mock_fn.assert_not_called()


def test_render_endocardia_views_only_lv_present(params):
    inputs = MeshReportInputs(mesh=MagicMock(points=MagicMock()), lv_endo_surf=MagicMock())
    with (
        patch.object(report_views, "pts_elem_to_pyvista", return_value=MagicMock()),
        patch.object(report_views, "visualise_mesh") as mock_fn,
    ):
        report_views.render_endocardia_views(inputs, params, pdf=MagicMock())
    # 3 views for LV only
    assert mock_fn.call_count == 3


def test_render_veins_views_noop_when_all_absent(empty_inputs, params):
    with (
        patch.object(report_views, "visualise_two_meshes") as mock_two,
        patch.object(report_views, "visualise_mesh") as mock_mesh,
    ):
        report_views.render_veins_views(empty_inputs, params, pdf=MagicMock())
    mock_two.assert_not_called()
    mock_mesh.assert_not_called()


def test_render_eas_views_noop_when_all_absent(empty_inputs, params):
    with patch.object(report_views, "visualise_vtx") as mock_fn:
        report_views.render_eas_views(empty_inputs, params, pdf=MagicMock())
    mock_fn.assert_not_called()


def test_render_eas_views_only_san_present(params):
    inputs = MeshReportInputs(mesh=MagicMock(), san_vtx=MagicMock())
    with patch.object(report_views, "visualise_vtx") as mock_fn:
        report_views.render_eas_views(inputs, params, pdf=MagicMock())
    assert mock_fn.call_count == 1


def test_render_mesh_views_always_runs(empty_inputs, params):
    """render_mesh_views has no optional gate — mesh is required and always present."""
    with patch.object(report_views, "visualise_mesh") as mock_fn:
        report_views.render_mesh_views(empty_inputs, params, pdf=MagicMock())
    assert mock_fn.call_count == 4
