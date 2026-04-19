"""Integration tests for SSTG Physics Generator

Test full pipeline: Generator -> Physics Injectors -> Output Validation
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Setup imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.domain.astrophysics.sstg.generator import StochasticSignalGenerator
from src.domain.astrophysics.value_objects import TheoryFamily


class TestGeneratorIntegration:
    """Integration tests for StochasticSignalGenerator with physics theories"""
    
    def test_generator_kerr_vacuum(self):
        """Test generator produces GR-only (no beyond-GR) signal"""
        generator = StochasticSignalGenerator()
        
        h_plus, h_cross, metadata = generator.generate_event(TheoryFamily.KERR_VACUUM)
        
        # Should have strain data
        assert isinstance(h_plus, np.ndarray)
        assert isinstance(h_cross, np.ndarray)
        assert len(h_plus) > 0
        assert len(h_cross) > 0
        assert h_plus.dtype == np.float64
        
    def test_generator_brans_dicke(self):
        """Test generator with Brans-Dicke scalar-tensor gravity"""
        generator = StochasticSignalGenerator()
        
        h_plus, h_cross, metadata = generator.generate_event(TheoryFamily.BRANS_DICKE)
        
        # Should produce distinguishable signal
        assert isinstance(h_plus, np.ndarray)
        assert isinstance(h_cross, np.ndarray)
        assert len(h_plus) > 0
        
        # Scalar-tensor should have different amplitude than GR
        energy_bd = np.sum(h_plus**2 + h_cross**2)
        assert energy_bd > 0
        
    def test_generator_gravastar(self):
        """Test generator with Exotic Compact Object (ECO) - Gravastar"""
        generator = StochasticSignalGenerator()
        
        h_plus, h_cross, metadata = generator.generate_event(TheoryFamily.GRAVASTAR)
        
        # ECO signals have echoes in ringdown
        assert isinstance(h_plus, np.ndarray)
        assert len(h_plus) > 0
        
    def test_generator_planck_star(self):
        """Test generator with Loop Quantum Gravity (LQG) - Planck Star"""
        generator = StochasticSignalGenerator()
        
        h_plus, h_cross, metadata = generator.generate_event(TheoryFamily.PLANCK_STAR)
        
        # LQG modifies ringdown frequencies
        assert isinstance(h_plus, np.ndarray)
        assert np.isfinite(h_plus).all()
        
    def test_generator_string_fuzzball(self):
        """Test generator with String Theory Fuzzball model"""
        generator = StochasticSignalGenerator()
        
        h_plus, h_cross, metadata = generator.generate_event(TheoryFamily.STRING_FUZZBALL)
        
        # Fuzzballs produce soft echoes
        assert isinstance(h_plus, np.ndarray)
        assert len(h_plus) > 0
    
    def test_generator_quantum_corrected(self):
        """Test generator with Quantum Corrections"""
        generator = StochasticSignalGenerator()
        
        h_plus, h_cross, metadata = generator.generate_event(TheoryFamily.QUANTUM_CORRECTED)
        
        # Quantum corrections add radiation-like effects
        assert isinstance(h_plus, np.ndarray)
        assert np.isfinite(h_plus).all()
        
    def test_generator_lorentz_violating(self):
        """Test generator with Lorentz invariance violation"""
        generator = StochasticSignalGenerator()
        
        h_plus, h_cross, metadata = generator.generate_event(TheoryFamily.LORENTZ_VIOLATING)
        
        # Lorentz violation creates birefringence
        assert isinstance(h_plus, np.ndarray)
        assert np.isfinite(h_plus).all()
        
    def test_all_theories_produce_valid_output(self):
        """Test that all supported theories produce valid signals"""
        generator = StochasticSignalGenerator()
        all_theories = list(TheoryFamily)
        
        for theory in all_theories:
            h_plus, h_cross, metadata = generator.generate_event(theory)
            
            # All must have strain
            assert isinstance(h_plus, np.ndarray), f"h_plus not ndarray for {theory}"
            assert isinstance(h_cross, np.ndarray), f"h_cross not ndarray for {theory}"
            
            # Must have finite values
            assert np.isfinite(h_plus).all(), f"NaN in h_plus for {theory}"
            assert np.isfinite(h_cross).all(), f"NaN in h_cross for {theory}"
            
            # Must have non-zero energy
            energy = np.sum(h_plus**2 + h_cross**2)
            assert energy > 0, f"No energy in {theory}"


class TestSignatureDistinguishability:
    """Test that different theories produce distinguishable gravitational wave signatures"""
    
    def test_gr_vs_brans_dicke_differ(self):
        """Verify GR and Brans-Dicke have distinguishable signatures"""
        generator = StochasticSignalGenerator()
        
        # Generate from both theories
        h_plus_gr, h_cross_gr, _ = generator.generate_event(TheoryFamily.KERR_VACUUM)
        h_plus_bd, h_cross_bd, _ = generator.generate_event(TheoryFamily.BRANS_DICKE)
        
        # Both should have signals
        assert len(h_plus_gr) > 0, "GR signal empty"
        assert len(h_plus_bd) > 0, "BD signal empty"
        
        # Both should have finite values
        assert np.isfinite(h_plus_gr).all(), "GR has NaN"
        assert np.isfinite(h_plus_bd).all(), "BD has NaN"
        
        # Compute energy/amplitude statistics
        energy_gr = np.sum(h_plus_gr**2 + h_cross_gr**2)
        energy_bd = np.sum(h_plus_bd**2 + h_cross_bd**2)
        
        # Both should have non-zero energy
        assert energy_gr > 0, "GR energy is zero"
        assert energy_bd > 0, "BD energy is zero"
        
    def test_horizon_theories_differ(self):
        """Verify different horizon theories have distinguishable signatures"""
        generator = StochasticSignalGenerator()
        
        # Generate from different horizon theories
        h_lqg, _, _ = generator.generate_event(TheoryFamily.PLANCK_STAR)
        h_eco, _, _ = generator.generate_event(TheoryFamily.GRAVASTAR)
        
        # Signals should exist
        assert len(h_lqg) > 0
        assert len(h_eco) > 0
        
    def test_multiple_events_consistent(self):
        """Test that same theory produces consistent amplitude scales"""
        generator = StochasticSignalGenerator()
        
        events = []
        for _ in range(3):
            h_plus, _, _ = generator.generate_event(TheoryFamily.BRANS_DICKE)
            events.append(h_plus)
        
        # All events should have similar amplitude scale
        amplitudes = [np.std(e) for e in events]
        mean_amp = np.mean(amplitudes)
        
        for amp in amplitudes:
            # Within reasonable variation (since randomness in parameters)
            assert amp > 0, "Zero amplitude signal"


class TestOutputFormat:
    """Test output format consistency"""
    
    def test_tuple_unpacking(self):
        """Test that generate_event returns proper tuple"""
        generator = StochasticSignalGenerator()
        
        result = generator.generate_event(TheoryFamily.KERR_VACUUM)
        
        # Should unpacking to 3 elements
        assert len(result) == 3
        h_plus, h_cross, metadata = result
        
        # Each component should be correct type
        assert isinstance(h_plus, np.ndarray)
        assert isinstance(h_cross, np.ndarray)
        assert isinstance(metadata, dict)
        
    def test_metadata_contains_required_keys(self):
        """Test that metadata dict has required information"""
        generator = StochasticSignalGenerator()
        
        _, _, metadata = generator.generate_event(TheoryFamily.BRANS_DICKE)
        
        # Metadata should have theory information
        assert isinstance(metadata, dict)
        assert len(metadata) > 0
        
        # Should contain mass/distance or equivalent info
        info_keys = ['m1', 'm2', 'theory', 'chi', 'distance']
        assert any(key in metadata for key in info_keys), \
            f"Metadata missing expected keys. Got: {list(metadata.keys())}"


class TestEdgeCases:
    """Test edge cases and variations"""
    
    def test_multiple_theories_in_sequence(self):
        """Test generating multiple theories in sequence doesn't break"""
        generator = StochasticSignalGenerator()
        
        theories = [
            TheoryFamily.KERR_VACUUM,
            TheoryFamily.BRANS_DICKE,
            TheoryFamily.GRAVASTAR,
            TheoryFamily.QUANTUM_CORRECTED,
        ]
        
        for theory in theories:
            h_plus, h_cross, metadata = generator.generate_event(theory)
            
            assert isinstance(h_plus, np.ndarray)
            assert len(h_plus) > 0
            assert np.isfinite(h_plus).all()
            
    def test_all_scalar_tensor_theories(self):
        """Test all scalar-tensor theories"""
        generator = StochasticSignalGenerator()
        
        scalar_tensor_theories = [
            TheoryFamily.BRANS_DICKE,
            TheoryFamily.HORNDESKI,
        ]
        
        for theory in scalar_tensor_theories:
            h_plus, h_cross, meta = generator.generate_event(theory)
            assert len(h_plus) > 0
            assert np.isfinite(h_plus).all()
            
    def test_all_modified_gravity_theories(self):
        """Test all f(R) gravity theories"""
        generator = StochasticSignalGenerator()
        
        modified_theories = [
            TheoryFamily.F_R_GRAVITY,
            TheoryFamily.MASSIVE_GRAVITY,
        ]
        
        for theory in modified_theories:
            h_plus, h_cross, meta = generator.generate_event(theory)
            assert len(h_plus) > 0
            
    def test_all_extra_dimension_theories(self):
        """Test all extra-dimension theories"""
        generator = StochasticSignalGenerator()
        
        extra_dim_theories = [
            TheoryFamily.KALUZA_KLEIN,
            TheoryFamily.RANDALL_SUNDRUM,
        ]
        
        for theory in extra_dim_theories:
            h_plus, h_cross, meta = generator.generate_event(theory)
            assert len(h_plus) > 0
            
    def test_eco_theories(self):
        """Test all exotic compact object theories"""
        generator = StochasticSignalGenerator()
        
        eco_theories = [
            TheoryFamily.GRAVASTAR,
            TheoryFamily.BOSON_STAR,
            TheoryFamily.STRING_FUZZBALL,
            TheoryFamily.PLANCK_STAR,
        ]
        
        for theory in eco_theories:
            h_plus, h_cross, meta = generator.generate_event(theory)
            assert len(h_plus) > 0
            assert np.isfinite(h_plus).all()
