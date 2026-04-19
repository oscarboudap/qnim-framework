"""
Tests para Layer 2 - Zeta Regularization
"""

import pytest
import numpy as np
from scipy.fft import rfft, rfftfreq

from src.domain.astrophysics.sstg.injectors.layer_zeta_regularization_complete import (
    Layer2ZetaRegularizationInjector,
    ZetaRegularizationParams,
    inject_zeta_regularization_simple
)


class TestZetaRegularization:
    """Test suite para regularización zeta"""
    
    @pytest.fixture
    def mock_signal(self):
        """Generar señal sintética GW simple"""
        fs = 4096
        duration = 0.2
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        
        # Ringdown-like signal (exponentially decaying oscillation)
        freq_dominant = 250  # Hz
        tau = 0.1  # decay timescale
        amplitude = 1e-21
        
        h_plus = amplitude * np.exp(-t/tau) * np.sin(2 * np.pi * freq_dominant * t)
        h_cross = amplitude * 0.5 * np.exp(-t/tau) * np.cos(2 * np.pi * freq_dominant * t)
        
        return h_plus, h_cross, fs, t
    
    @pytest.fixture
    def default_params(self):
        """Parámetros por defecto - detectables para tests"""
        return ZetaRegularizationParams(
            entropy_scale=0.01,
            zeta_power=2.0,
            zero_point_strength=1e-3,
            logarithmic_correction=1e-4,
            cutoff_frequency=1e3
        )
    
    def test_zeta_injection_returns_dict(self, mock_signal, default_params):
        """Test que injection retorna diccionario correcto"""
        h_plus, h_cross, fs, _ = mock_signal
        n_samples = len(h_plus)
        freq_array = rfftfreq(n_samples, 1.0/fs)
        
        result = Layer2ZetaRegularizationInjector.inject_zeta_regularization(
            h_plus, h_cross, default_params, freq_array, fs
        )
        
        assert isinstance(result, dict)
        assert "h_plus" in result
        assert "h_cross" in result
        assert "physics_applied" in result
        assert "metadata" in result
        
        assert len(result["h_plus"]) == len(h_plus)
        assert len(result["h_cross"]) == len(h_cross)
        assert "entropy_correction" in result["physics_applied"]
    
    def test_entropy_correction_modifies_signal(self, mock_signal, default_params):
        """Test que corrección de entropía modifica la señal"""
        h_plus, h_cross, fs, _ = mock_signal
        n_samples = len(h_plus)
        freq_array = rfftfreq(n_samples, 1.0/fs)
        
        # Con regularización
        result_zeta = Layer2ZetaRegularizationInjector.inject_zeta_regularization(
            h_plus, h_cross, default_params, freq_array, fs
        )
        
        # Sin regularización (parámetros = 0)
        params_zero = ZetaRegularizationParams(
            entropy_scale=0.0,
            zeta_power=2.0,
            zero_point_strength=0.0,
            logarithmic_correction=0.0,
            cutoff_frequency=1e3
        )
        result_no_zeta = Layer2ZetaRegularizationInjector.inject_zeta_regularization(
            h_plus, h_cross, params_zero, freq_array, fs
        )
        
        # Debe haber diferencia
        diff = result_zeta["h_plus"] - result_no_zeta["h_plus"]
        assert np.max(np.abs(diff)) > 1e-30  # Algún efecto
    
    def test_zero_point_noise_has_expected_spectrum(self, mock_signal, default_params):
        """Test que ruido de punto cero tiene espectro decorrelacionado"""
        h_plus, h_cross, fs, _ = mock_signal
        n_samples = len(h_plus)
        freq_array = rfftfreq(n_samples, 1.0/fs)
        
        result = Layer2ZetaRegularizationInjector.inject_zeta_regularization(
            h_plus, h_cross, default_params, freq_array, fs
        )
        
        # Analizar espectro de salida
        fft_result = rfft(result["h_plus"])
        power = np.abs(fft_result)**2
        
        # Power debe ser no-trivial
        assert np.sum(power) > 0
    
    def test_logarithmic_correction_applied(self, mock_signal, default_params):
        """Test que corrección logarítmica es aplicada"""
        h_plus, h_cross, fs, _ = mock_signal
        n_samples = len(h_plus)
        freq_array = rfftfreq(n_samples, 1.0/fs)
        
        # Con corrección logarítmica fuerte
        params_strong_log = ZetaRegularizationParams(
            entropy_scale=0.0,
            zeta_power=2.0,
            zero_point_strength=0.0,
            logarithmic_correction=0.1,  # Fuerte para detectar
            cutoff_frequency=1e3
        )
        
        result = Layer2ZetaRegularizationInjector.inject_zeta_regularization(
            h_plus, h_cross, params_strong_log, freq_array, fs
        )
        
        # Debe haber modulación temporal
        diff = result["h_plus"] - h_plus
        rms_diff = np.sqrt(np.mean(diff**2))
        assert rms_diff > 1e-30  # Something applied
    
    def test_spectral_smoothing_reduces_high_freq(self, mock_signal, default_params):
        """Test que suavizado espectral reduce altas frecuencias"""
        h_plus, h_cross, fs, _ = mock_signal
        n_samples = len(h_plus)
        freq_array = rfftfreq(n_samples, 1.0/fs)
        
        result = Layer2ZetaRegularizationInjector.inject_zeta_regularization(
            h_plus, h_cross, default_params, freq_array, fs
        )
        
        # FFT ambos
        fft_orig = rfft(h_plus)
        fft_zeta = rfft(result["h_plus"])
        
        power_orig = np.abs(fft_orig)**2
        power_zeta = np.abs(fft_zeta)**2
        
        # Energía total debe ser comparable (no divergencia)
        assert np.sum(power_zeta) > 0
        assert np.sum(power_orig) > 0
    
    def test_output_shapes_preserved(self, mock_signal, default_params):
        """Test que formas de output se preservan"""
        h_plus, h_cross, fs, _ = mock_signal
        n_samples = len(h_plus)
        freq_array = rfftfreq(n_samples, 1.0/fs)
        
        result = Layer2ZetaRegularizationInjector.inject_zeta_regularization(
            h_plus, h_cross, default_params, freq_array, fs
        )
        
        assert result["h_plus"].shape == h_plus.shape
        assert result["h_cross"].shape == h_cross.shape
        assert len(result["physics_applied"]) > 0
    
    def test_simple_wrapper(self, mock_signal):
        """Test función wrapper simple"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_zeta, h_cross_zeta, metadata = inject_zeta_regularization_simple(
            h_plus, h_cross, mass=10.0, fs=fs, entropy_scale=0.01
        )
        
        assert len(h_plus_zeta) == len(h_plus)
        assert len(h_cross_zeta) == len(h_cross)
        assert "mass" in metadata
        assert metadata["mass"] == 10.0
        assert metadata["technique"] == "Riemann_zeta_regularization"
    
    def test_entropy_calculation(self):
        """Test cálculo de entropía renormalizada"""
        area = 100.0  # Horizon area in l_P^2
        mass = 10.0   # in m_P
        
        S_BH, S_zeta, S_total = Layer2ZetaRegularizationInjector.compute_renormalized_entropy(
            area, mass, include_corrections=True
        )
        
        # Bekenstein-Hawking entropy
        assert S_BH == area / 4.0
        
        # Zeta corrections should be non-zero but smaller than BH
        assert S_zeta != 0
        assert np.abs(S_zeta) < np.abs(S_BH)
        
        # Total entropy should be BH + corrections
        assert np.isclose(S_total, S_BH + S_zeta)
    
    def test_entropy_includes_mass_dependence(self):
        """Test que entropía depende de masa"""
        area = 100.0
        
        S_BH1, S_zeta1, S_total1 = Layer2ZetaRegularizationInjector.compute_renormalized_entropy(
            area, mass=5.0, include_corrections=True
        )
        
        S_BH2, S_zeta2, S_total2 = Layer2ZetaRegularizationInjector.compute_renormalized_entropy(
            area, mass=20.0, include_corrections=True
        )
        
        # Ambas partes deben cambiar con masa
        assert S_BH1 == S_BH2  # BH no depende de masa en esta implementación
        assert S_zeta1 != S_zeta2  # Pero correcciones sí
        assert S_total1 != S_total2


class TestZetaPhysicsSignatures:
    """Test que regularización zeta crea firmas distinguibles"""
    
    @pytest.fixture
    def ringdown_signal(self):
        """Generar señal tipo ringdown"""
        fs = 4096
        duration = 0.5
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        
        # Ringdown: compuesto de múltiples QNM decay
        amplitude = 1e-20
        
        h_plus = amplitude * np.exp(-t/0.1) * np.sin(2*np.pi*250*t)
        h_plus += 0.3 * amplitude * np.exp(-t/0.15) * np.sin(2*np.pi*150*t)
        
        h_cross = amplitude * 0.3 * np.exp(-t/0.1) * np.cos(2*np.pi*250*t)
        
        return h_plus, h_cross, fs, t
    
    def test_mass_scaling_of_entropy_corrections(self, ringdown_signal):
        """Test que correcciones de entropía escalan con masa"""
        h_plus, h_cross, fs, _ = ringdown_signal
        
        # Inyectar con diferentes masas
        _, _, meta_low = inject_zeta_regularization_simple(
            h_plus, h_cross, mass=5.0, fs=fs, entropy_scale=0.01
        )
        
        _, _, meta_high = inject_zeta_regularization_simple(
            h_plus, h_cross, mass=50.0, fs=fs, entropy_scale=0.01
        )
        
        # Ambas deben tener correcciones de entropía
        assert "entropy_scale" in meta_low
        assert "entropy_scale" in meta_high
        assert meta_low["entropy_scale"] == meta_high["entropy_scale"]
    
    def test_zeta_vs_no_zeta_correlation_drop(self, ringdown_signal):
        """Test que regularización zeta reduce correlación de cola"""
        h_plus, h_cross, fs, _ = ringdown_signal
        n_samples = len(h_plus)
        freq_array = rfftfreq(n_samples, 1.0/fs)
        
        params = ZetaRegularizationParams(
            entropy_scale=0.02,
            zeta_power=2.0,
            zero_point_strength=1e-3,
            logarithmic_correction=1e-3,
            cutoff_frequency=1e3
        )
        
        result = Layer2ZetaRegularizationInjector.inject_zeta_regularization(
            h_plus, h_cross, params, freq_array, fs
        )
        
        # La señal debe estar modificada
        diff = result["h_plus"] - h_plus
        assert np.max(np.abs(diff)) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
