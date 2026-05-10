#!/usr/bin/env python3
"""Verify HDF5 file from test generation"""

import h5py
import os
from pathlib import Path

file_path = Path("data/synthetic/massive_dataset/synthetic_gw_20260510_012726.h5")

if not file_path.exists():
    print(f"❌ File not found: {file_path}")
    exit(1)

with h5py.File(file_path, 'r') as f:
    print("HDF5 Validation:")
    print(f"  Datasets: {list(f.keys())}")
    print(f"  strain_plus shape: {f['strain_plus'].shape}")
    print(f"  strain_cross shape: {f['strain_cross'].shape}")
    print(f"  theory_labels: {[t.decode() if isinstance(t, bytes) else t for t in f['theory_labels'][:]]}")
    print(f"  source_types: {[t.decode() if isinstance(t, bytes) else t for t in f['source_types'][:]]}")

file_size_mb = os.path.getsize(file_path) / 1024 / 1024
print(f"  File size: {file_size_mb:.2f} MB")
print("✓ HDF5 structure valid - Ready for 500-event generation")
