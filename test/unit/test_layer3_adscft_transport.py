"""
Tests para Layer 3 - AdS/CFT Holographic Transport
"""

import pytest
import numpy as np
from scipy.fft import rfft, rfftfreq

from src.domain.astrophysics.sstg.injectors.layer3_adscft_transport_complete import (
    Layer3AdSCFTTransportInjector,
    AdSCFTTransportParams,
    inject_adscft_transport_simple
)


class TestAdSCFTTransport:
    """Test suite para transporte holográfico AdS/CFT"""
    
    @pytest.fixture
    def mock_signal(self):
        """Generar señal sintética GW tipo ringdown"""
        fs = 4096
        duration = 0.2
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        
        # Ringdown-like signal
        freq_dominant = 250  # Hz
        tau = 0.1  # decay timescale
        amplitude = 1e-21
        
        h_plus = amplitude * np.exp(-t/tau) * np.sin(2 * np.pi * freq_dominant * t)
        h_cross = amplitude * 0.5 * np.exp(-t/tau) * np.cos(2 * np.pi * freq_dominant * t)
        
        return h_plus, h_cross, fs, t
    
    @pytest.fixture
    def default_params(self):
        """Parámetros por defecto - compatibles con KSS bound"""
        return AdSCFTTransportParams(
            viscosity_ratio=1.5,  # 1.5× KSS bound
            thermalization_time=0.05,
            conductivity_scale=1e-3,
            qnm_damping=1.0,
            n_qnm_modes=3
        )
    
    def test_adscft_injection_returns_dict(self, mock_signal, default_params):
        """Test que injection retorna diccionario correcto"""
        h_plus, h_cross, fs, _ = mock_signal
        n_samples = len(h_plus)
        freq_array = rfftfreq(n_samples, 1.0/fs)
        
        result = Layer3AdSCFTTransportInjector.inject_adscft_transport(
            h_plus, h_cross, default_params, freq_array, fs
        )
        
        assert isinstance(result, dict)
        assert "h_plus" in result
        assert "h_cross" in result
        assert "physics_applied" in result
        assert "metadata" in result
        
        assert len(result["h_plus"]) == len(h_plus)
        assert len(result["h_cross"]) == len(h_cross)
        assert "viscous_dissipation" in result["physics_applied"]
    
    def test_viscous_dissipation_reduces_amplitude(self, mock_signal, default_params):
        """Test que disipación viscosa reduce amplitud de señal"""
        h_plus, h_cross, fs, _ = mock_signal
        n_samples = len(h_plus)
        freq_array = rfftfreq(n_samples, 1.0/fs)
        
        # Con disipación viscosa
        result = Layer3AdSCFTTransportInjector.inject_adscft_transport(
            h_plus, h_cross, default_params, freq_array, fs
        )
        
        # Sin disipación (params zero)
        params_no_diss = AdSCFTTransportParams(
            viscosity_ratio=0.0,
            thermalization_time=0.05,
            conductivity_scale=0.0,
            qnm_damping=0.0,
            n_qnm_modes=0
        )
        result_no_diss = Layer3AdSCFTTransportInjector.inject_adscft_transport(
            h_plus, h_cross, params_no_diss, freq_array, fs
        )
        
        # Con disipación, amplitud media debe ser menor
        amp_with = np.mean(np.abs(result["h_plus"]))
        amp_without = np.mean(np.abs(result_no_diss["h_plus"]))
        
        assert amp_with < amp_without
    
    def test_thermalization_envelope_smooth(self, mock_signal, default_params):
        """Test que envolvente de termalizacion es suave"""
        h_plus, h_cross, fs, t = mock_signal
        n_samples = len(h_plus)
        freq_array = rfftfreq(n_samples, 1.0/fs)
        
        result = Layer3AdSCFTTransportInjector.inject_adscft_transport(
            h_plus, h_cross, default_params, freq_array, fs
        )
        
        # La derivada de la envelope debe ser suave (sin discontinuidades)
        d_h = np.abs(np.gradient(result["h_plus"]))
        
        # Máxima pendiente
        max_slope = np.max(d_h)
        assert max_slope < 1.0  # Reasonable maximum derivative
    
    def test_holographic_conductivity_phase_shift(self, mock_signal, default_params):
        """Test que conductividad holográfica introduce cambios de fase"""
        h_plus, h_cross, fs, _ = mock_signal
        n_samples = len(h_plus)
        freq_array = rfftfreq(n_samples, 1.0/fs)
        
        result = Layer3AdSCFTTransportInjector.inject_adscft_transport(
            h_plus, h_cross, default_params, freq_array, fs
        )
        
        # Comparar espectros original vs modificado
        fft_orig = rfft(h_plus)
        fft_result = rfft(result["h_plus"])
        
        # Debe haber diferencia en fase
        phase_orig = np.angle(fft_orig)
        phase_result = np.angle(fft_result)
        phase_diff = np.abs(phase_result - phase_orig)
        
        # Algún cambio de fase debe estar presente
        assert np.mean(phase_diff) > 0.01  # More than numerical noise
    
    def test_qnm_spectrum_visible_at_late_times(self, mock_signal, default_params):
        """Test que espectro QNM visible en tiempos tardios"""
        h_plus, h_cross, fs, t = mock_signal
        n_samples = len(h_plus)
        freq_array = rfftfreq(n_samples, 1.0/fs)
        
        result = Layer3AdSCFTTransportInjector.inject_adscft_transport(
            h_plus, h_cross, default_params, freq_array, fs
        )
        
        # Late times: last 30% of signal
        t_late_start = int(0.7 * n_samples)
        h_late = result["h_plus"][t_late_start:]
        
        # Debe haber oscilaciones en tiempos tardios (QNM ringdown)
        fft_late = rfft(h_late)
        power_late = np.abs(fft_late)**2
        
        # Peak de potencia debe estar en rango esperado
        peak_freq = freq_array[np.argmax(power_late)]
        # QNM ~ 1/τ_therm ~ 1/0.05 ~ 20 Hz base (más modos pueden ser más altos)
        assert 5 < peak_freq < 1000  # Flexible range for QNM (depends on τ_therm)
    
    def test_output_shapes_preserved(self, mock_signal, default_params):
        """Test que formas de output se preservan"""
        h_plus, h_cross, fs, _ = mock_signal
        n_samples = len(h_plus)
        freq_array = rfftfreq(n_samples, 1.0/fs)
        
        result = Layer3AdSCFTTransportInjector.inject_adscft_transport(
            h_plus, h_cross, default_params, freq_array, fs
        )
        
        assert result["h_plus"].shape == h_plus.shape
        assert result["h_cross"].shape == h_cross.shape
        assert len(result["physics_applied"]) > 0
    
    def test_simple_wrapper(self, mock_signal):
        """Test función wrapper simple"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_adscft, h_cross_adscft, metadata = inject_adscft_transport_simple(
            h_plus, h_cross, mass=10.0, fs=fs, viscosity_ratio=1.5
        )
        
        assert len(h_plus_adscft) == len(h_plus)
        assert len(h_cross_adscft) == len(h_cross)
        assert "mass" in metadata
        assert metadata["mass"] == 10.0
        assert metadata["technique"] == "AdS_CFT_holographic_transport"
    
    def test_higher_viscosity_more_dissipation(self, mock_signal):
        """Test que viscosidad más alta causa más disipación"""
        h_plus, h_cross, fs, _ = mock_signal
        n_samples = len(h_plus)
        freq_array = rfftfreq(n_samples, 1.0/fs)
        
        # Baja viscosidad
        params_low = AdSCFTTransportParams(viscosity_ratio=0.5)
        result_low = Layer3AdSCFTTransportInjector.inject_adscft_transport(
            h_plus, h_cross, params_low, freq_array, fs
        )
        
        # Alta viscosidad
        params_high = AdSCFTTransportParams(viscosity_ratio=5.0)
        result_high = Layer3AdSCFTTransportInjector.inject_adscft_transport(
            h_plus, h_cross, params_high, freq_array, fs
        )
        
        # Mayor viscosidad → mayor disipación → señal más pequeña
        amp_low = np.mean(np.abs(result_low["h_plus"]))
        amp_high = np.mean(np.abs(result_high["h_plus"]))
        
        assert amp_high < amp_low
    
    def test_thermalization_timescale_affects_decay(self, mock_signal):
        """Test que escala de termalizacion afecta decaimiento"""
        h_plus, h_cross, fs, t = mock_signal
        n_samples = len(h_plus)
        freq_array = rfftfreq(n_samples, 1.0/fs)
        
        # Termalizacion rápida
        params_fast = AdSCFTTransportParams(thermalization_time=0.01)
        result_fast = Layer3AdSCFTTransportInjector.inject_adscft_transport(
            h_plus, h_cross, params_fast, freq_array, fs
        )
        
        # Termalizacion lenta
        params_slow = AdSCFTTransportParams(thermalization_time=0.1)
        result_slow = Layer3AdSCFTTransportInjector.inject_adscft_transport(
            h_plus, h_cross, params_slow, freq_array, fs
        )
        
        # Ambas deben tener efectos distintos
        diff_fast = result_fast["h_plus"] - h_plus
        diff_slow = result_slow["h_plus"] - h_plus
        
        rms_fast = np.sqrt(np.mean(diff_fast**2))
        rms_slow = np.sqrt(np.mean(diff_slow**2))
        
        # Ambos deben ser diferentes
        assert np.abs(rms_fast - rms_slow) > 1e-30


class TestAdSCFTPhysicsSignatures:
    """Test que AdS/CFT holográfico crea firmas físicas esperadas"""
    
    @pytest.fixture
    def ringdown_signal(self):
        """Generar ringdown típico de BH"""
        fs = 4096
        duration = 0.5
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        
        amplitude = 1e-20
        h_plus = amplitude * np.exp(-t/0.1) * np.sin(2*np.pi*250*t)
        h_plus += 0.3 * amplitude * np.exp(-t/0.15) * np.sin(2*np.pi*150*t)
        
        h_cross = amplitude * 0.3 * np.exp(-t/0.1) * np.cos(2*np.pi*250*t)
        
        return h_plus, h_cross, fs, t
    
    def test_kss_bound_implementation(self):
        """Test que se implementa correctamente el bound KSS"""
        kss_bound = Layer3AdSCFTTransportInjector.KSS_BOUND
        
        # KSS bound: η/s ≥ 1/(4π)
        expected = 1.0 / (4.0 * np.pi)
        assert np.isclose(kss_bound, expected, rtol=1e-6)
    
    def test_thermalization_and_entropy(self):
        """Test cálculo de termalizacion y entropía"""
        energy = 10.0  # in m_P
        temperature = 1.0  # in m_P
        viscosity_ratio = 1.5
        
        tau_therm, S_init, S_final = Layer3AdSCFTTransportInjector.compute_holographic_thermalization(
            energy, temperature, viscosity_ratio
        )
        
        # Thermalization timescale must be positive
        assert tau_therm > 0
        
        # Entropies must be positive
        assert S_init > 0
        assert S_final > 0
        
        # Final entropy should be related to viscosity
        assert S_final != S_init  # They differ due to viscosity correction
    
    def test_mass_scaling_in_wrapper(self, ringdown_signal):
        """Test que wrapper escala correctamente con masa"""
        h_plus, h_cross, fs, _ = ringdown_signal
        
        # Low mass
        _, _, meta_low = inject_adscft_transport_simple(
            h_plus, h_cross, mass=5.0, fs=fs
        )
        
        # High mass
        _, _, meta_high = inject_adscft_transport_simple(
            h_plus, h_cross, mass=50.0, fs=fs
        )
        
        assert meta_low["mass"] == 5.0
        assert meta_high["mass"] == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
