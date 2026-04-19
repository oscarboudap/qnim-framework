"""Pytest configuration and fixtures for QNIM tests"""

import sys
from pathlib import Path

# Setup path for imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

qnim_root = Path(__file__).resolve().parent.parent
if str(qnim_root) not in sys.path:
    sys.path.insert(0, str(qnim_root))

import pytest
import numpy as np


@pytest.fixture
def sample_strain():
    """Create a sample strain signal for testing"""
    n_samples = 1024
    fs = 16384.0
    duration = n_samples / fs
    
    h_plus = np.random.randn(n_samples) * 1e-20
    h_cross = np.random.randn(n_samples) * 1e-20
    
    return {
        "h_plus": h_plus,
        "h_cross": h_cross,
        "fs": fs,
        "n_samples": n_samples,
        "duration": duration,
    }


@pytest.fixture
def mass_parameters():
    """Standard mass parameters for BH signals"""
    return {
        "total_mass_msun": 65.0,  # ~GW150914
        "mass_ratio": 1.0,
        "distance_mpc": 410.0,
        "fs": 16384.0,
    }
