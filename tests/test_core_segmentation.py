import SimpleITK as sitk

from imatools.core import segmentation as core_seg


def test_segmentation_parser_builds():
    from imatools.cli import segmentation

    segmentation._build_parser()  # raises on missing handler / arg conflict


def test_morph_label_returns_image_same_geometry(label_image):  # conftest fixture
    out = core_seg.morph_label(label_image, label=1, operation="dilate", radius=1)
    assert isinstance(out, sitk.Image)
    assert out.GetSize() == label_image.GetSize()
