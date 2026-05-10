#!/usr/bin/env python3
"""
PHASE 1: Diagnostic test for synthetic GW generator
- Test 1 event generation
- Verify imports and dependencies
- Check HDF5 output
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import tempfile
import h5py
import numpy as np

print("=" * 80)
print("PHASE 1: DIAGNOSTIC TEST")
print("=" * 80)

# Test 1: Import SimpleSyntheticGWGenerator
print("\n[1/5] Testing imports...")
try:
    from src.domain.astrophysics.sstg.simple_generator import SimpleSyntheticGWGenerator
    print("✅ SimpleSyntheticGWGenerator imported successfully")
except Exception as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

# Test 2: Instantiate generator
print("\n[2/5] Instantiating generator...")
try:
    generator = SimpleSyntheticGWGenerator(sample_rate=4096)
    print(f"✅ Generator instantiated with sample_rate=4096")
except Exception as e:
    print(f"❌ Failed to instantiate: {e}")
    sys.exit(1)

# Test 3: Generate single event
print("\n[3/5] Generating single event (CBC)...")
try:
    result = generator.generate_event(source_type="CBC")
    h_plus = result["h_plus"]
    h_cross = result["h_cross"]
    params = result.get("parameters", {})
    
    print(f"✅ Event generated successfully")
    print(f"   - h_plus shape: {h_plus.shape}, dtype: {h_plus.dtype}, range: [{h_plus.min():.2e}, {h_plus.max():.2e}]")
    print(f"   - h_cross shape: {h_cross.shape}, dtype: {h_cross.dtype}, range: [{h_cross.min():.2e}, {h_cross.max():.2e}]")
    print(f"   - parameters keys: {list(params.keys())}")
except Exception as e:
    print(f"❌ Failed to generate event: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Write and read HDF5
print("\n[4/5] Testing HDF5 write/read...")
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_event.h5"
        
        # Write
        with h5py.File(test_file, 'w') as h5f:
            h5f.create_dataset('strain_plus', data=h_plus.astype(np.float32), compression='gzip')
            h5f.create_dataset('strain_cross', data=h_cross.astype(np.float32), compression='gzip')
            h5f.create_dataset('theory_label', data="RG", dtype=h5py.string_dtype(encoding='utf-8'))
            h5f.create_dataset('source_type', data="CBC", dtype=h5py.string_dtype(encoding='utf-8'))
        
        # Read back
        with h5py.File(test_file, 'r') as h5f:
            sp = h5f['strain_plus'][:]
            sc = h5f['strain_cross'][:]
            theory = h5f['theory_label'][()]
            source = h5f['source_type'][()]
        
        print(f"✅ HDF5 write/read successful")
        print(f"   - Strain_plus recovered: shape {sp.shape}")
        print(f"   - Strain_cross recovered: shape {sc.shape}")
        print(f"   - Theory: {theory}, Source: {source}")
except Exception as e:
    print(f"❌ HDF5 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Verify config loading
print("\n[5/5] Testing configuration loading...")
try:
    from src.cli import ScriptConfig
    config = ScriptConfig.from_env()
    print(f"✅ Config loaded successfully")
    print(f"   - target_samples: {config.training.preprocessing.target_samples}")
    print(f"   - sample_rate_hz: {config.training.preprocessing.sample_rate_hz}")
    print(f"   - data_synthetic_path: {config.training.paths.get_data_synthetic_path()}")
except Exception as e:
    print(f"❌ Config failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL DIAGNOSTIC TESTS PASSED")
print("=" * 80)
print("\nReady to proceed to 10-event batch test")
print("Next command: python scripts/pipelines/01_generate_synthetic_gw.py")
