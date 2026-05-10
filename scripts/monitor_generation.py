#!/usr/bin/env python3
"""Monitor generation progress"""

import h5py
from pathlib import Path
import os
import time

output_dir = Path("data/synthetic/massive_dataset")

# Find the latest generation file
files = sorted(output_dir.glob("synthetic_gw_*.h5"))
if not files:
    print("No generation files found yet")
    exit(0)

latest_file = files[-1]
print(f"Monitoring: {latest_file.name}")

try:
    with h5py.File(latest_file, 'r') as f:
        num_events = f['strain_plus'].shape[0]
        file_size_mb = os.path.getsize(latest_file) / 1024 / 1024
        
        print(f"Current progress: {num_events} events")
        print(f"File size: {file_size_mb:.2f} MB")
        
        if num_events > 0:
            # Estimate remaining time
            per_event_mb = file_size_mb / num_events
            estimated_final_mb = per_event_mb * 500
            print(f"Estimated total size: {estimated_final_mb:.0f} MB")
            print(f"ETA: ~{(estimated_final_mb - file_size_mb) / 60:.0f} more minutes")
except Exception as e:
    print(f"Could not read file: {e}")
