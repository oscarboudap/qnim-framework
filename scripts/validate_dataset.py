#!/usr/bin/env python3
"""Validate HDF5 dataset for VQC training"""

import h5py

f = h5py.File('data/synthetic/massive_dataset/synthetic_gw_20260510_012853.h5', 'r')
print('HDF5 VALIDATION FOR VQC TRAINING:')
print(f'  Datasets: {list(f.keys())}')
print(f'  strain_plus: {f["strain_plus"].shape}')
print(f'  strain_cross: {f["strain_cross"].shape}')

labels = [t.decode() if isinstance(t, bytes) else t for t in f["theory_labels"][:]]
unique_labels = set(labels)
print(f'  Theory labels: {unique_labels}')
print(f'  Label distribution:')
for label in sorted(unique_labels):
    count = sum(1 for l in labels if l == label)
    print(f'    - {label}: {count}')

f.close()
print('✓ Ready for VQC training')
