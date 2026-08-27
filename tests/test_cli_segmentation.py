"""
Tests for the imatools-segmentation CLI.

Covers the multi-label `extract-label` wiring added alongside the
`extract_single_label` dtype fix, and the `count` subcommand. No golden
fixtures: these exercise argparse wiring and output-path derivation over
already-tested core/label and core/image functions.

`main()` reads sys.argv, so these drive `_build_parser()` and call the resolved
handler directly — the same path `main()` takes after parsing.
"""

import numpy as np
import SimpleITK as sitk  # noqa: N813

from imatools.cli import segmentation


def test_segmentation_cli_stub():
    """CLI module loads and exposes main()."""
    assert hasattr(segmentation, "main")


def _write_label_image(path, dtype=np.uint8):
    """Label map with three disjoint labels (1, 2, 3) on a small grid."""
    arr = np.zeros((6, 6, 6), dtype=dtype)
    arr[0:2] = 1
    arr[2:4] = 2
    arr[4:6] = 3
    sitk.WriteImage(sitk.GetImageFromArray(arr), str(path))
    return path


def _run(argv):
    args = segmentation._build_parser().parse_args(argv)
    return args.func(args)


def _read(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def test_extract_label_single_is_binarised_by_default(tmp_path):
    src = _write_label_image(tmp_path / "labels.nrrd")
    out = tmp_path / "one.nrrd"

    assert _run(["extract-label", "-in", str(src), "-l", "2", "-out", str(out)]) == 0
    assert set(np.unique(_read(out)).tolist()) == {0, 1}


def test_extract_label_single_default_output_name_unchanged(tmp_path):
    """A single label still derives '<stem>_extracted_<zero-padded>.nrrd'."""
    src = _write_label_image(tmp_path / "labels.nrrd")

    assert _run(["extract-label", "-in", str(src), "-l", "2"]) == 0
    assert (tmp_path / "labels_extracted_002.nrrd").exists()


def test_extract_label_many_keep_values(tmp_path):
    """Several labels, original values preserved, everything else zeroed."""
    src = _write_label_image(tmp_path / "labels.nrrd")
    out = tmp_path / "many.nrrd"

    assert (
        _run(["extract-label", "-in", str(src), "-l", "1", "3", "--keep-values", "-out", str(out)])
        == 0
    )
    result = _read(out)
    assert set(np.unique(result).tolist()) == {0, 1, 3}
    # label 2 was not requested, so its voxels are now background
    assert (result[2:4] == 0).all()


def test_extract_label_many_binarised_by_default(tmp_path):
    """Without --keep-values, several labels collapse to 1 (as merge-labels does)."""
    src = _write_label_image(tmp_path / "labels.nrrd")
    out = tmp_path / "many_bin.nrrd"

    assert _run(["extract-label", "-in", str(src), "-l", "1", "3", "-out", str(out)]) == 0
    assert set(np.unique(_read(out)).tolist()) == {0, 1}


def test_extract_label_many_default_output_name(tmp_path):
    """Several labels join with '_', matching merge-labels / delete-labels."""
    src = _write_label_image(tmp_path / "labels.nrrd")

    assert _run(["extract-label", "-in", str(src), "-l", "1", "3", "--keep-values"]) == 0
    assert (tmp_path / "labels_extracted_1_3.nrrd").exists()


def test_extract_label_keep_values_survives_wide_dtype(tmp_path):
    """End-to-end guard on the uint8 cast that wrapped 300 -> 44."""
    arr = np.zeros((4, 4, 4), dtype=np.uint16)
    arr[0] = 300
    arr[1] = 4000
    src = tmp_path / "wide.nrrd"
    sitk.WriteImage(sitk.GetImageFromArray(arr), str(src))
    out = tmp_path / "wide_out.nrrd"

    assert (
        _run(
            [
                "extract-label",
                "-in",
                str(src),
                "-l",
                "300",
                "4000",
                "--keep-values",
                "-out",
                str(out),
            ]
        )
        == 0
    )
    assert set(np.unique(_read(out)).tolist()) == {0, 300, 4000}


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


def _write_grey_and_roi(tmp_path, spacing=(1.0, 1.0, 1.0)):
    """Greyscale image with zeros inside the ROI, plus a stray 7 outside it."""
    grey = np.zeros((6, 6, 6), dtype=np.uint8)
    grey[1:5, 1:5, 1:5] = 7
    grey[2, 2, 2] = 0
    grey[3, 3, 3] = 0
    grey[0, 0, 0] = 7

    roi = np.zeros((6, 6, 6), dtype=np.uint8)
    roi[1:5, 1:5, 1:5] = 1

    gpath, rpath = tmp_path / "grey.nrrd", tmp_path / "roi.nrrd"
    for arr, path in ((grey, gpath), (roi, rpath)):
        im = sitk.GetImageFromArray(arr)
        im.SetSpacing(spacing)
        sitk.WriteImage(im, str(path))
    return gpath, rpath


def test_count_zero_inside_mask(tmp_path, caplog):
    """The motivating case: count value 0 inside an ROI of a greyscale image."""
    grey, roi = _write_grey_and_roi(tmp_path)

    with caplog.at_level("INFO"):
        assert _run(["count", "-in", str(grey), "-l", "0", "-mask", str(roi)]) == 0

    assert "Value 0 (inside mask): 2 voxels" in caplog.text


def test_count_without_mask_covers_whole_image(tmp_path, caplog):
    grey, _ = _write_grey_and_roi(tmp_path)

    with caplog.at_level("INFO"):
        assert _run(["count", "-in", str(grey), "-l", "7"]) == 0

    # 4^3 block minus the two zeroed voxels, plus the stray 7 outside the ROI
    assert "Value 7: 63 voxels" in caplog.text
    assert "inside mask" not in caplog.text


def test_count_mask_excludes_outside_voxels(tmp_path, caplog):
    grey, roi = _write_grey_and_roi(tmp_path)

    with caplog.at_level("INFO"):
        assert _run(["count", "-in", str(grey), "-l", "7", "-mask", str(roi)]) == 0

    assert "Value 7 (inside mask): 62 voxels" in caplog.text


def test_count_reports_volume_from_spacing(tmp_path, caplog):
    """Volume is the count times the voxel volume taken from the image spacing."""
    grey, roi = _write_grey_and_roi(tmp_path, spacing=(2.0, 2.0, 3.0))

    with caplog.at_level("INFO"):
        assert _run(["count", "-in", str(grey), "-l", "0", "-mask", str(roi)]) == 0

    assert "2 voxels (24.000 mm³)" in caplog.text  # 2 * 2*2*3


def test_count_reports_volume_in_ml(tmp_path, caplog):
    grey, roi = _write_grey_and_roi(tmp_path, spacing=(2.0, 2.0, 3.0))

    with caplog.at_level("INFO"):
        assert _run(["count", "-in", str(grey), "-l", "0", "-mask", str(roi), "--units", "mL"]) == 0

    assert "2 voxels (0.024 mL)" in caplog.text
