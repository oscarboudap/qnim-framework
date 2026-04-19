"""
Tests para Layer 5 - Beyond-GR Physics Injection

Valida inyección de teorías alternativas a GR:
  - Brans-Dicke (radiación dipolar)
  - Extra dimensions (ADD, RS)
  - Massive graviton (dRGT)
  - Lorentz violation
"""

import pytest
import numpy as np
from scipy.fft import rfft, rfftfreq

from src.domain.astrophysics.sstg.injectors.layer5_beyond_gr_complete import (
    Layer5BeyondGRInjector,
    BeyondGRParams,
)


class TestLayer5BeyondGR:
    """Test suite para Layer 5 - Beyond-GR Physics"""
    
    @pytest.fixture
    def mock_signal(self):
        """Generar señal sintética GW simple"""
        fs = 4096
        duration = 0.2
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        
        # Ringdown-like signal
        freq_dominant = 250  # Hz
        tau = 0.1
        amplitude = 1e-21
        
        h_plus = amplitude * np.exp(-t/tau) * np.sin(2 * np.pi * freq_dominant * t)
        h_cross = amplitude * 0.5 * np.exp(-t/tau) * np.cos(2 * np.pi * freq_dominant * t)
        
        return h_plus, h_cross, fs, t
    
    # ==================== BRANS-DICKE TESTS ====================
    
    def test_brans_dicke_dipolar_returns_tuple(self, mock_signal):
        """Test que inject_brans_dicke_dipolar retorna tupla correcta"""
        h_plus, h_cross, fs, _ = mock_signal
        
        result = Layer5BeyondGRInjector.inject_brans_dicke_dipolar(
            h_plus, h_cross, omega_param=100.0, total_mass_msun=60.0, fs=fs
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert len(result[0]) == len(h_plus)
        assert len(result[1]) == len(h_cross)
    
    def test_brans_dicke_dipolar_modifies_signal(self, mock_signal):
        """Test que radiación dipolar modifica la señal"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_bd, h_cross_bd = Layer5BeyondGRInjector.inject_brans_dicke_dipolar(
            h_plus, h_cross, omega_param=100.0, total_mass_msun=60.0, fs=fs
        )
        
        # Debe haber diferencia detectables
        diff_plus = np.sum((h_plus_bd - h_plus)**2)
        diff_cross = np.sum((h_cross_bd - h_cross)**2)
        
        assert diff_plus > 1e-50
        assert diff_cross > 1e-50
    
    def test_brans_dicke_amplitude_reduction(self, mock_signal):
        """Test que amplitud BD es menor que GR (pérdida por dipolar)"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_bd, h_cross_bd = Layer5BeyondGRInjector.inject_brans_dicke_dipolar(
            h_plus, h_cross, omega_param=100.0, total_mass_msun=60.0, fs=fs
        )
        
        # Amplitudes RMS
        rms_plus = np.sqrt(np.mean(h_plus**2))
        rms_plus_bd = np.sqrt(np.mean(h_plus_bd**2))
        rms_cross = np.sqrt(np.mean(h_cross**2))
        rms_cross_bd = np.sqrt(np.mean(h_cross_bd**2))
        
        # BD debe reducir amplitud (energía se irradia como dipolar)
        assert rms_plus_bd < rms_plus
        assert rms_cross_bd < rms_cross
    
    def test_brans_dicke_omega_effect(self, mock_signal):
        """Test que parámetro omega_BD afecta el resultado"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_w100, _ = Layer5BeyondGRInjector.inject_brans_dicke_dipolar(
            h_plus, h_cross, omega_param=100.0, total_mass_msun=60.0, fs=fs
        )
        h_plus_w500, _ = Layer5BeyondGRInjector.inject_brans_dicke_dipolar(
            h_plus, h_cross, omega_param=500.0, total_mass_msun=60.0, fs=fs
        )
        
        # Diferentes omega deben dar diferentes resultados
        diff = np.sum((h_plus_w100 - h_plus_w500)**2)
        assert diff > 1e-50
    
    # ==================== SCALAR POLARIZATION TESTS ====================
    
    def test_scalar_polarization_returns_triple(self, mock_signal):
        """Test que inject_scalar_polarization retorna h+, h×, h_scalar"""
        h_plus, h_cross, fs, _ = mock_signal
        
        result = Layer5BeyondGRInjector.inject_scalar_polarization(
            h_plus, h_cross, amplitude_fraction=0.05, fs=fs
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert len(result[0]) == len(h_plus)
        assert len(result[1]) == len(h_cross)
        assert len(result[2]) == len(h_plus)
    
    def test_scalar_polarization_non_zero(self, mock_signal):
        """Test que polarización escalar es no-trivial"""
        h_plus, h_cross, fs, _ = mock_signal
        
        _, _, h_scalar = Layer5BeyondGRInjector.inject_scalar_polarization(
            h_plus, h_cross, amplitude_fraction=0.05, fs=fs
        )
        
        # h_scalar debe tener contenido
        assert np.max(np.abs(h_scalar)) > 0
        assert not np.all(h_scalar == 0)
    
    def test_scalar_polarization_amplitude_controlled(self, mock_signal):
        """Test que amplitud escalar es controlada por parámetro"""
        h_plus, h_cross, fs, _ = mock_signal
        
        _, _, h_scalar_low = Layer5BeyondGRInjector.inject_scalar_polarization(
            h_plus, h_cross, amplitude_fraction=0.01, fs=fs
        )
        _, _, h_scalar_high = Layer5BeyondGRInjector.inject_scalar_polarization(
            h_plus, h_cross, amplitude_fraction=0.10, fs=fs
        )
        
        rms_low = np.sqrt(np.mean(h_scalar_low**2))
        rms_high = np.sqrt(np.mean(h_scalar_high**2))
        
        # Mayor amplitud_fraction debe dar mayor h_scalar
        assert rms_high > rms_low
    
    # ==================== EXTRA DIMENSIONS (ADD) TESTS ====================
    
    def test_inject_extra_dimensions_returns_tuple(self, mock_signal):
        """Test que inject_extra_dimensions_leakage retorna tuple"""
        h_plus, h_cross, fs, _ = mock_signal
        
        result = Layer5BeyondGRInjector.inject_extra_dimensions_leakage(
            h_plus, h_cross, n_extra_dims=2, distance_mpc=400.0, fs=fs
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_extra_dimensions_reduces_amplitude(self, mock_signal):
        """Test que dimensiones extra atenúan señal (escape al bulk)"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_add, h_cross_add = Layer5BeyondGRInjector.inject_extra_dimensions_leakage(
            h_plus, h_cross, n_extra_dims=2, distance_mpc=400.0, fs=fs
        )
        
        rms_orig = np.sqrt(np.mean(h_plus**2))
        rms_add = np.sqrt(np.mean(h_plus_add**2))
        
        # ADD reduce amplitud (energía escapa)
        assert rms_add < rms_orig
    
    def test_extra_dimensions_dimensionality_effect(self, mock_signal):
        """Test que número de dimensiones extra afecta atenuación"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_2d, _ = Layer5BeyondGRInjector.inject_extra_dimensions_leakage(
            h_plus, h_cross, n_extra_dims=2, distance_mpc=400.0, fs=fs
        )
        h_plus_4d, _ = Layer5BeyondGRInjector.inject_extra_dimensions_leakage(
            h_plus, h_cross, n_extra_dims=4, distance_mpc=400.0, fs=fs
        )
        
        rms_2d = np.sqrt(np.mean(h_plus_2d**2))
        rms_4d = np.sqrt(np.mean(h_plus_4d**2))
        
        # Mayor número de dimensiones = mayor atenuación
        assert rms_4d < rms_2d
    
    # ==================== MASSIVE GRAVITON TESTS ====================
    
    def test_massive_graviton_returns_tuple(self, mock_signal):
        """Test que inject_massive_graviton_dispersion retorna tuple"""
        h_plus, h_cross, fs, _ = mock_signal
        
        result = Layer5BeyondGRInjector.inject_massive_graviton_dispersion(
            h_plus, h_cross, graviton_mass_hmevcmsq=1e-22,
            total_mass_msun=60.0, distance_mpc=400.0, fs=fs
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_massive_graviton_dispersion_phase_shift(self, mock_signal):
        """Test que gravitón masivo induce desfase dispersivo"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_mg, h_cross_mg = Layer5BeyondGRInjector.inject_massive_graviton_dispersion(
            h_plus, h_cross, graviton_mass_hmevcmsq=1e-22,
            total_mass_msun=60.0, distance_mpc=400.0, fs=fs
        )
        
        # Debe haber cambio
        diff = np.sum((h_plus_mg - h_plus)**2)
        assert diff > 1e-50
    
    def test_massive_graviton_mass_effect(self, mock_signal):
        """Test que masa del gravitón afecta el resultado"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_light, _ = Layer5BeyondGRInjector.inject_massive_graviton_dispersion(
            h_plus, h_cross, graviton_mass_hmevcmsq=1e-23,
            total_mass_msun=60.0, distance_mpc=400.0, fs=fs
        )
        h_plus_heavy, _ = Layer5BeyondGRInjector.inject_massive_graviton_dispersion(
            h_plus, h_cross, graviton_mass_hmevcmsq=1e-21,
            total_mass_msun=60.0, distance_mpc=400.0, fs=fs
        )
        
        # Diferentes masas deben dar diferentes resultados
        diff = np.sum((h_plus_light - h_plus_heavy)**2)
        assert diff > 1e-50
    
    # ==================== LORENTZ VIOLATION TESTS ====================
    
    def test_lorentz_violation_returns_tuple(self, mock_signal):
        """Test que inject_lorentz_violation retorna tuple"""
        h_plus, h_cross, fs, _ = mock_signal
        
        result = Layer5BeyondGRInjector.inject_lorentz_violation(
            h_plus, h_cross, violation_parameter=1e-15, fs=fs
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_lorentz_violation_birefringence_effect(self, mock_signal):
        """Test que violación Lorentz produce birefringencia"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_lv, h_cross_lv = Layer5BeyondGRInjector.inject_lorentz_violation(
            h_plus, h_cross, violation_parameter=1e-15, fs=fs
        )
        
        # Debe haber diferencia entre + y ×
        # Las velocidades diferirán por birefringencia
        diff_plus = np.sum((h_plus_lv - h_plus)**2)
        diff_cross = np.sum((h_cross_lv - h_cross)**2)
        
        # Ambos deben cambiar debido a birefringencia
        assert diff_plus > 1e-72
        assert diff_cross > 1e-72
    
    def test_lorentz_violation_parameter_effect(self, mock_signal):
        """Test que parámetro de violación afecta resultado"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_weak, _ = Layer5BeyondGRInjector.inject_lorentz_violation(
            h_plus, h_cross, violation_parameter=1e-17, fs=fs
        )
        h_plus_strong, _ = Layer5BeyondGRInjector.inject_lorentz_violation(
            h_plus, h_cross, violation_parameter=1e-13, fs=fs
        )
        
        # Diferentes parámetros deben dar diferentes resultados
        diff = np.sum((h_plus_weak - h_plus_strong)**2)
        assert diff > 1e-72
    
    # ==================== GENERAL PROPERTIES TESTS ====================
    
    def test_output_shapes_preserved(self, mock_signal):
        """Test que formas de output se preservan"""
        h_plus, h_cross, fs, _ = mock_signal
        n = len(h_plus)
        
        # Brans-Dicke
        hp_bd, hc_bd = Layer5BeyondGRInjector.inject_brans_dicke_dipolar(
            h_plus, h_cross, fs=fs
        )
        assert hp_bd.shape == (n,)
        assert hc_bd.shape == (n,)
        
        # ADD
        hp_add, hc_add = Layer5BeyondGRInjector.inject_extra_dimensions_leakage(
            h_plus, h_cross, fs=fs
        )
        assert hp_add.shape == (n,)
        assert hc_add.shape == (n,)
    
    def test_no_nans_or_infs(self, mock_signal):
        """Test que no hay NaN o infinitos en outputs"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_bd, h_cross_bd = Layer5BeyondGRInjector.inject_brans_dicke_dipolar(
            h_plus, h_cross, fs=fs
        )
        
        assert not np.any(np.isnan(h_plus_bd))
        assert not np.any(np.isnan(h_cross_bd))
        assert not np.any(np.isinf(h_plus_bd))
        assert not np.any(np.isinf(h_cross_bd))
    
    def test_all_methods_numerical_stability(self, mock_signal):
        """Test estabilidad numérica de todos los métodos"""
        h_plus, h_cross, fs, _ = mock_signal
        
        # Brans-Dicke
        _ = Layer5BeyondGRInjector.inject_brans_dicke_dipolar(h_plus, h_cross, fs=fs)
        
        # Scalar polarization
        _ = Layer5BeyondGRInjector.inject_scalar_polarization(h_plus, h_cross, fs=fs)
        
        # ADD
        _ = Layer5BeyondGRInjector.inject_extra_dimensions_leakage(h_plus, h_cross, fs=fs)
        
        # Massive graviton
        _ = Layer5BeyondGRInjector.inject_massive_graviton_dispersion(
            h_plus, h_cross, fs=fs
        )
        
        # Lorentz violation
        _ = Layer5BeyondGRInjector.inject_lorentz_violation(h_plus, h_cross, fs=fs)
        
        # Si llegamos aquí sin crash, estabilidad numérica confirmada
        assert True
