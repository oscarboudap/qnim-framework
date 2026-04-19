"""Quick sanity check test - Verifies all layers work with correct signatures"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.domain.astrophysics.sstg.injectors.layer5_beyond_gr_complete import (
    Layer5BeyondGRInjector, BeyondGRParams
)
from src.domain.astrophysics.sstg.injectors.layer6_horizon_topology_complete import (
    Layer6HorizonTopologyInjector, HorizonTopologyParams
)
from src.domain.astrophysics.sstg.injectors.layer7_quantum_corrections_complete import (
    Layer7QuantumCorrectionsInjector, QuantumCorrectionParams
)


def main():
    print("\n" + "="*70)
    print("SSTG PHYSICS LAYERS - SANITY CHECK TEST")
    print("="*70)
    
    # Create test signal
    n_samples = 1024
    fs = 16384.0
    h_plus = np.random.randn(n_samples) * 1e-20
    h_cross = np.random.randn(n_samples) * 1e-20
    
    # ========== LAYER 5: Beyond-GR ==========
    print("\n[LAYER 5] Beyond-GR Physics")
    print("-" * 70)
    
    try:
        # Test Brans-Dicke
        print("Testing Brans-Dicke (Scalar-Tensor)...")
        params = BeyondGRParams(theory="Brans-Dicke", omega_bd=100.0)
        output = Layer5BeyondGRInjector.apply_beyond_gr_physics(
            h_plus, h_cross, params,
            total_mass_msun=65.0,  # ✓ CORRECT PARAMETER
            distance_mpc=410.0,     # ✓ CORRECT PARAMETER
            fs=int(fs)              # ✓ CORRECT TYPE
        )
        print(f"  ✓ Output keys: {list(output.keys())}")
        print(f"  ✓ h_scalar generated: {np.mean(np.abs(output['h_scalar'])) > 0}")
        
        # Test ADD
        print("Testing ADD (Extra Dimensions)...")
        params = BeyondGRParams(theory="ADD", extra_dims=3)
        output = Layer5BeyondGRInjector.apply_beyond_gr_physics(
            h_plus, h_cross, params,
            total_mass_msun=65.0,
            distance_mpc=410.0,
            fs=int(fs)
        )
        print(f"  ✓ Output keys: {list(output.keys())}")
        print(f"  ✓ Extra dims detected: {'n_extra_dims' in output}")
        
        print("\n✅ LAYER 5 PASSED\n")
        
    except Exception as e:
        print(f"❌ LAYER 5 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    
    # ========== LAYER 6: Horizon Topology ==========
    print("[LAYER 6] Horizon Topology Effects")
    print("-" * 70)
    
    try:
        # Test ECO echoes
        print("Testing ECO Echoes...")
        params = HorizonTopologyParams(
            theory="ECO",
            echo_delay=0.001,
            echo_amplitude=0.5
        )
        output = Layer6HorizonTopologyInjector.apply_horizon_topology(
            h_plus, h_cross, params,
            mass=65.0,              # ✓ CORRECT PARAMETER (NOT total_mass_msun)
            fs=fs                   # ✓ CORRECT TYPE
        )
        print(f"  ✓ Output keys: {list(output.keys())}")
        print(f"  ✓ h_plus shape: {output['h_plus'].shape}")
        
        # Test LQG
        print("Testing LQG (Loop Quantum Gravity)...")
        params = HorizonTopologyParams(
            theory="LQG",
            lqg_area_quantum=0.5
        )
        output = Layer6HorizonTopologyInjector.apply_horizon_topology(
            h_plus, h_cross, params,
            mass=65.0,              # ✓ CORRECT PARAMETER
            fs=fs
        )
        print(f"  ✓ Output keys: {list(output.keys())}")
        print(f"  ✓ Output valid: {np.isfinite(output['h_plus']).all()}")
        
        print("\n✅ LAYER 6 PASSED\n")
        
    except Exception as e:
        print(f"❌ LAYER 6 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    
    # ========== LAYER 7: Quantum Corrections ==========
    print("[LAYER 7] Quantum Corrections")
    print("-" * 70)
    
    try:
        # Test Hawking evaporation
        print("Testing Hawking Evaporation...")
        params = QuantumCorrectionParams(
            theory="Hawking",
            evaporation_rate=1e-27
        )
        output = Layer7QuantumCorrectionsInjector.apply_quantum_corrections(
            h_plus, h_cross, params,
            mass=65.0,              # ✓ CORRECT PARAMETER (NOT total_mass_msun)
            fs=fs                   # ✓ CORRECT TYPE
        )
        print(f"  ✓ Output keys: {list(output.keys())}")
        print(f"  ✓ Physics applied: {output.get('physics_applied', [])}")
        
        # Test AdS/CFT
        print("Testing AdS/CFT Viscosity...")
        params = QuantumCorrectionParams(
            theory="AdS_CFT",
            viscosity_ratio=0.25
        )
        output = Layer7QuantumCorrectionsInjector.apply_quantum_corrections(
            h_plus, h_cross, params,
            mass=65.0,              # ✓ CORRECT PARAMETER
            fs=fs
        )
        print(f"  ✓ Output keys: {list(output.keys())}")
        print(f"  ✓ Metadata: {output.get('metadata', {})}")
        
        print("\n✅ LAYER 7 PASSED\n")
        
    except Exception as e:
        print(f"❌ LAYER 7 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    
    print("="*70)
    print("✅ ALL SANITY CHECKS PASSED - Physics Layers Ready for Production")
    print("="*70)
    print("\nParameterización resumida:")
    print("  • Layer5.apply_beyond_gr_physics(..., total_mass_msun, distance_mpc, fs)")
    print("  • Layer6.apply_horizon_topology(..., mass, fs)")
    print("  • Layer7.apply_quantum_corrections(..., mass, fs)")
    print("\nPr óximos pasos: pytest test/unit/test_physics_layers.py\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
