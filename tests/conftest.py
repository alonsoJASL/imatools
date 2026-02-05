"""
Test configuration and fixtures for imatools.
"""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_data_dir():
    """Path to sample test data (to be populated)."""
    data_dir = Path(__file__).parent / "fixtures"
    return data_dir
