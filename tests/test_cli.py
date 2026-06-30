"""
Tests for CLI entry points.

These will be populated as we migrate functionality.
"""

from imatools.cli import mesh, scar, segmentation, volume


def test_volume_cli_stub():
    """Test that volume CLI loads."""
    assert hasattr(volume, "main")


def test_segmentation_cli_stub():
    """Test that segmentation CLI loads."""
    assert hasattr(segmentation, "main")


def test_mesh_cli_stub():
    """Test that mesh CLI loads."""
    assert hasattr(mesh, "main")


def test_scar_cli_stub():
    """Test that scar CLI loads."""
    assert hasattr(scar, "main")
