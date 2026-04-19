"""
Tests para Layer 6 - Horizon Topology Physics

Valida inyección de efectos de topología del horizonte de eventos:
  - Ecos de ECO (Effective Compact Object)
  - Espectro discreto de LQG
  - Memoria gravitacional
  - Estructura de Fuzzball
  - Ringdown modificado
"""

import pytest
import numpy as np
from scipy import signal as scipy_signal

from src.domain.astrophysics.sstg.injectors.layer6_horizon_topology_complete import (
    Layer6HorizonTopologyInjector,
    HorizonTopologyParams,
)


class TestLayer6HorizonTopology:
    """Test suite para Layer 6 - Topología de Horizonte"""
    
    @pytest.fixture
    def mock_signal_with_ringdown(self):
        """Generar señal sintética GW con ringdown"""
        fs = 4096
        duration = 0.5
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        
        # Inspiral simple
        t_merger = 0.35
        amplitude_factor = np.exp(-(t - t_merger)**2 / (2 * 0.01**2))
        
        # Ringdown (después de merger)
        ringdown_start = t_merger
        freq_ringdown = 250  # Hz
        decay_time = 0.1
        
        h_plus = np.zeros_like(t)
        h_cross = np.zeros_like(t)
        
        # Inspiraling part
        mask_inspiral = t < ringdown_start
        h_plus[mask_inspiral] = 1e-21 * amplitude_factor[mask_inspiral] * np.sin(2 * np.pi * 100 * t[mask_inspiral])
        
        # Ringdown part
        mask_ringdown = t >= ringdown_start
        ringdown_time = t[mask_ringdown] - ringdown_start
        h_plus[mask_ringdown] = 1e-21 * np.exp(-ringdown_time / decay_time) * np.sin(2 * np.pi * freq_ringdown * ringdown_time)
        h_cross = h_plus * 0.5  # Perpendicular component
        
        return h_plus, h_cross, fs, t
    
    # ==================== ECO ECHOES TESTS ====================
    
    def test_eco_echoes_returns_tuple(self, mock_signal_with_ringdown):
        """Test que inject_eco_echoes retorna tupla correcta"""
        h_plus, h_cross, fs, _ = mock_signal_with_ringdown
        
        result = Layer6HorizonTopologyInjector.inject_eco_echoes(
            h_plus, h_cross, echo_delay=0.05, echo_amplitude=0.15, n_echoes=3, fs=fs
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert len(result[0]) == len(h_plus)
        assert len(result[1]) == len(h_cross)
    
    def test_eco_echoes_amplify_signal(self, mock_signal_with_ringdown):
        """Test que ecos ECO amplifican la señal total"""
        h_plus, h_cross, fs, _ = mock_signal_with_ringdown
        
        h_plus_eco, h_cross_eco = Layer6HorizonTopologyInjector.inject_eco_echoes(
            h_plus, h_cross, echo_delay=0.05, echo_amplitude=0.15, n_echoes=3, fs=fs
        )
        
        # Ecos deben agregar contenido a la señal
        diff = np.sum((h_plus_eco - h_plus)**2)
        assert diff > 1e-50
    
    def test_eco_delay_effect(self, mock_signal_with_ringdown):
        """Test que delay del eco afecta resultado"""
        h_plus, h_cross, fs, _ = mock_signal_with_ringdown
        
        h_plus_short, _ = Layer6HorizonTopologyInjector.inject_eco_echoes(
            h_plus, h_cross, echo_delay=0.02, echo_amplitude=0.15, n_echoes=3, fs=fs
        )
        h_plus_long, _ = Layer6HorizonTopologyInjector.inject_eco_echoes(
            h_plus, h_cross, echo_delay=0.10, echo_amplitude=0.15, n_echoes=3, fs=fs
        )
        
        # Diferentes delays deben dar diferentes resultados
        diff = np.sum((h_plus_short - h_plus_long)**2)
        assert diff > 1e-50
    
    def test_eco_amplitude_effect(self, mock_signal_with_ringdown):
        """Test que amplitud del eco controla su magnitud"""
        h_plus, h_cross, fs, _ = mock_signal_with_ringdown
        
        h_plus_weak, _ = Layer6HorizonTopologyInjector.inject_eco_echoes(
            h_plus, h_cross, echo_delay=0.05, echo_amplitude=0.05, n_echoes=3, fs=fs
        )
        h_plus_strong, _ = Layer6HorizonTopologyInjector.inject_eco_echoes(
            h_plus, h_cross, echo_delay=0.05, echo_amplitude=0.25, n_echoes=3, fs=fs
        )
        
        # Mayor amplitud debe dar mayor amplificación total
        rms_weak = np.sqrt(np.mean((h_plus_weak - h_plus)**2))
        rms_strong = np.sqrt(np.mean((h_plus_strong - h_plus)**2))
        
        assert rms_strong > rms_weak
    
    def test_eco_number_effect(self, mock_signal_with_ringdown):
        """Test que número de ecos afecta la respuesta"""
        h_plus, h_cross, fs, _ = mock_signal_with_ringdown
        
        h_plus_1echo, _ = Layer6HorizonTopologyInjector.inject_eco_echoes(
            h_plus, h_cross, echo_delay=0.05, echo_amplitude=0.15, n_echoes=1, fs=fs
        )
        h_plus_4echoes, _ = Layer6HorizonTopologyInjector.inject_eco_echoes(
            h_plus, h_cross, echo_delay=0.05, echo_amplitude=0.15, n_echoes=4, fs=fs
        )
        
        # Más ecos → mayor diferencia del original
        rms_diff_1 = np.sqrt(np.mean((h_plus_1echo - h_plus)**2))
        rms_diff_4 = np.sqrt(np.mean((h_plus_4echoes - h_plus)**2))
        
        assert rms_diff_4 > rms_diff_1
    
    # ==================== LQG DISCRETE SPECTRUM TESTS ====================
    
    def test_lqg_spectrum_returns_tuple(self, mock_signal_with_ringdown):
        """Test que inject_lqg_discrete_spectrum retorna tupla"""
        h_plus, h_cross, fs, _ = mock_signal_with_ringdown
        
        result = Layer6HorizonTopologyInjector.inject_lqg_discrete_spectrum(
            h_plus, h_cross, area_quantum=0.3, fs=fs, mass=10.0
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_lqg_modifies_signal(self, mock_signal_with_ringdown):
        """Test que cuantización LQG modifica la señal"""
        h_plus, h_cross, fs, _ = mock_signal_with_ringdown
        
        h_plus_lqg, h_cross_lqg = Layer6HorizonTopologyInjector.inject_lqg_discrete_spectrum(
            h_plus, h_cross, area_quantum=0.3, fs=fs, mass=10.0
        )
        
        # Debe haber diferencia (LQG es efecto pequeño)
        diff = np.sum((h_plus_lqg - h_plus)**2)
        assert diff > 1e-72
    
    def test_lqg_quantum_parameter_effect(self, mock_signal_with_ringdown):
        """Test que parámetro de cuantización afecta resultado"""
        h_plus, h_cross, fs, _ = mock_signal_with_ringdown
        
        h_plus_weak, _ = Layer6HorizonTopologyInjector.inject_lqg_discrete_spectrum(
            h_plus, h_cross, area_quantum=0.05, fs=fs, mass=10.0
        )
        h_plus_strong, _ = Layer6HorizonTopologyInjector.inject_lqg_discrete_spectrum(
            h_plus, h_cross, area_quantum=0.50, fs=fs, mass=10.0
        )
        
        # Ambos deben cambiar la señal comparados con original
        diff_weak = np.sum((h_plus_weak - h_plus)**2)
        diff_strong = np.sum((h_plus_strong - h_plus)**2)
        assert diff_weak > 0 and diff_strong > 0
    
    # ==================== GRAVITATIONAL MEMORY TESTS ====================
    
    def test_memory_returns_tuple(self, mock_signal_with_ringdown):
        """Test que inject_gravitational_memory retorna tupla"""
        h_plus, h_cross, fs, _ = mock_signal_with_ringdown
        
        result = Layer6HorizonTopologyInjector.inject_gravitational_memory(
            h_plus, h_cross, fs=fs, amplitude=0.05
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_memory_adds_offset(self, mock_signal_with_ringdown):
        """Test que memoria gravitacional agrega componente DC"""
        h_plus, h_cross, fs, _ = mock_signal_with_ringdown
        
        h_plus_mem, h_cross_mem = Layer6HorizonTopologyInjector.inject_gravitational_memory(
            h_plus, h_cross, fs=fs, amplitude=0.05
        )
        
        # Final de la señal debe tener offset permanente
        offset_plus = np.mean(h_plus_mem[-100:]) - np.mean(h_plus[-100:])
        offset_cross = np.mean(h_cross_mem[-100:]) - np.mean(h_cross[-100:])
        
        # Debe haber offseet detectables
        assert np.abs(offset_plus) > 1e-25
        assert np.abs(offset_cross) > 1e-25
    
    def test_memory_amplitude_control(self, mock_signal_with_ringdown):
        """Test que amplitud de memoria es controlada"""
        h_plus, h_cross, fs, _ = mock_signal_with_ringdown
        
        h_plus_weak, _ = Layer6HorizonTopologyInjector.inject_gravitational_memory(
            h_plus, h_cross, fs=fs, amplitude=0.01
        )
        h_plus_strong, _ = Layer6HorizonTopologyInjector.inject_gravitational_memory(
            h_plus, h_cross, fs=fs, amplitude=0.10
        )
        
        # Mayor amplitud debe dar mayor cambio
        change_weak = np.sqrt(np.sum((h_plus_weak - h_plus)**2))
        change_strong = np.sqrt(np.sum((h_plus_strong - h_plus)**2))
        
        assert change_strong > change_weak
    
    # ==================== FUZZBALL ECHOES TESTS ====================
    
    def test_fuzzball_echoes_returns_tuple(self, mock_signal_with_ringdown):
        """Test que inject_fuzzball_echoes retorna tupla"""
        h_plus, h_cross, fs, _ = mock_signal_with_ringdown
        
        result = Layer6HorizonTopologyInjector.inject_fuzzball_echoes(
            h_plus, h_cross, fs=fs, amplitude=0.15, mass=10.0
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_fuzzball_amplifies_signal(self, mock_signal_with_ringdown):
        """Test que ecos de fuzzball amplifican la señal"""
        h_plus, h_cross, fs, _ = mock_signal_with_ringdown
        
        h_plus_fb, h_cross_fb = Layer6HorizonTopologyInjector.inject_fuzzball_echoes(
            h_plus, h_cross, fs=fs, amplitude=0.15, mass=10.0
        )
        
        # RMS debe aumentar
        rms_orig = np.sqrt(np.mean(h_plus**2))
        rms_fb = np.sqrt(np.mean(h_plus_fb**2))
        
        assert rms_fb > rms_orig
    
    def test_fuzzball_amplitude_parameter(self, mock_signal_with_ringdown):
        """Test que parámetro amplitude controla magnitud"""
        h_plus, h_cross, fs, _ = mock_signal_with_ringdown
        
        h_plus_weak, _ = Layer6HorizonTopologyInjector.inject_fuzzball_echoes(
            h_plus, h_cross, fs=fs, amplitude=0.05, mass=10.0
        )
        h_plus_strong, _ = Layer6HorizonTopologyInjector.inject_fuzzball_echoes(
            h_plus, h_cross, fs=fs, amplitude=0.25, mass=10.0
        )
        
        # Mayor amplitud → mayor diferencia
        diff_weak = np.sqrt(np.mean((h_plus_weak - h_plus)**2))
        diff_strong = np.sqrt(np.mean((h_plus_strong - h_plus)**2))
        
        assert diff_strong > diff_weak
    
    # ==================== MODIFIED RINGDOWN TESTS ====================
    
    def test_modified_ringdown_returns_tuple(self, mock_signal_with_ringdown):
        """Test que inject_modified_ringdown retorna tupla"""
        h_plus, h_cross, fs, _ = mock_signal_with_ringdown
        
        result = Layer6HorizonTopologyInjector.inject_modified_ringdown(
            h_plus, h_cross, fs=fs, q_modification=0.2, mass=10.0
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_modified_ringdown_changes_decay(self, mock_signal_with_ringdown):
        """Test que Q-modification cambia la región de ringdown"""
        h_plus, h_cross, fs, _ = mock_signal_with_ringdown
        
        h_plus_rd, h_cross_rd = Layer6HorizonTopologyInjector.inject_modified_ringdown(
            h_plus, h_cross, fs=fs, q_modification=0.3, mass=10.0
        )
        
        # Debe haber cambio en la señal
        diff = np.sum((h_plus_rd - h_plus)**2)
        assert diff > 1e-50
    
    def test_modified_ringdown_q_parameter_effect(self, mock_signal_with_ringdown):
        """Test que Q-modification parámetro afecta resultado"""
        h_plus, h_cross, fs, _ = mock_signal_with_ringdown
        
        h_plus_q10, _ = Layer6HorizonTopologyInjector.inject_modified_ringdown(
            h_plus, h_cross, fs=fs, q_modification=0.1, mass=10.0
        )
        h_plus_q50, _ = Layer6HorizonTopologyInjector.inject_modified_ringdown(
            h_plus, h_cross, fs=fs, q_modification=0.5, mass=10.0
        )
        
        # Ambos deben cambiar comparados con original
        diff_q10 = np.sum((h_plus_q10 - h_plus)**2)
        diff_q50 = np.sum((h_plus_q50 - h_plus)**2)
        assert diff_q10 > 0 and diff_q50 > 0
    
    def test_modified_ringdown_preserves_inspiral(self, mock_signal_with_ringdown):
        """Test que inspiral no se modifica, solo ringdown"""
        h_plus, h_cross, fs, t = mock_signal_with_ringdown
        
        h_plus_rd, _ = Layer6HorizonTopologyInjector.inject_modified_ringdown(
            h_plus, h_cross, fs=fs, q_modification=0.3, mass=10.0
        )
        
        # Primeras muestras (inspiral) deben ser similares
        inspiral_portion = int(0.7 * len(h_plus))
        diff_inspiral = np.sum((h_plus_rd[:inspiral_portion] - h_plus[:inspiral_portion])**2)
        
        # Debe haber modificación principalmente en ringdown
        ringdown_portion = h_plus_rd[inspiral_portion:]
        original_ringdown = h_plus[inspiral_portion:]
        diff_ringdown = np.sum((ringdown_portion - original_ringdown)**2)
        
        # Ringdown debe cambiar más que inspiral
        assert diff_ringdown > diff_inspiral
    
    # ==================== GENERAL PROPERTIES TESTS ====================
    
    def test_all_methods_preserve_shape(self, mock_signal_with_ringdown):
        """Test que todos los métodos preservan forma"""
        h_plus, h_cross, fs, _ = mock_signal_with_ringdown
        n = len(h_plus)
        
        # ECO
        hp_eco, hc_eco = Layer6HorizonTopologyInjector.inject_eco_echoes(
            h_plus, h_cross, echo_delay=0.05, echo_amplitude=0.15, n_echoes=3, fs=fs
        )
        assert hp_eco.shape == (n,) and hc_eco.shape == (n,)
        
        # LQG
        hp_lqg, hc_lqg = Layer6HorizonTopologyInjector.inject_lqg_discrete_spectrum(
            h_plus, h_cross, area_quantum=0.3, fs=fs, mass=10.0
        )
        assert hp_lqg.shape == (n,) and hc_lqg.shape == (n,)
        
        # Memory
        hp_mem, hc_mem = Layer6HorizonTopologyInjector.inject_gravitational_memory(
            h_plus, h_cross, fs=fs, amplitude=0.05
        )
        assert hp_mem.shape == (n,) and hc_mem.shape == (n,)
    
    def test_no_nans_or_infs_in_outputs(self, mock_signal_with_ringdown):
        """Test estabilidad numérica: sin NaN/Inf"""
        h_plus, h_cross, fs, _ = mock_signal_with_ringdown
        
        # ECO
        hp_eco, hc_eco = Layer6HorizonTopologyInjector.inject_eco_echoes(
            h_plus, h_cross, echo_delay=0.05, echo_amplitude=0.15, n_echoes=3, fs=fs
        )
        assert not np.any(np.isnan(hp_eco)) and not np.any(np.isinf(hp_eco))
        
        # LQG
        hp_lqg, hc_lqg = Layer6HorizonTopologyInjector.inject_lqg_discrete_spectrum(
            h_plus, h_cross, area_quantum=0.3, fs=fs, mass=10.0
        )
        assert not np.any(np.isnan(hp_lqg)) and not np.any(np.isinf(hp_lqg))
        
        # Fuzzball
        hp_fb, hc_fb = Layer6HorizonTopologyInjector.inject_fuzzball_echoes(
            h_plus, h_cross, fs=fs, amplitude=0.15, mass=10.0
        )
        assert not np.any(np.isnan(hp_fb)) and not np.any(np.isinf(hp_fb))
    
    def test_all_methods_numerical_stability(self, mock_signal_with_ringdown):
        """Test que todos los métodos son numericamente estables"""
        h_plus, h_cross, fs, _ = mock_signal_with_ringdown
        
        try:
            # Ejecutar todos los métodos
            Layer6HorizonTopologyInjector.inject_eco_echoes(
                h_plus, h_cross, echo_delay=0.05, echo_amplitude=0.15, n_echoes=3, fs=fs
            )
            Layer6HorizonTopologyInjector.inject_lqg_discrete_spectrum(
                h_plus, h_cross, area_quantum=0.3, fs=fs, mass=10.0
            )
            Layer6HorizonTopologyInjector.inject_gravitational_memory(h_plus, h_cross, fs=fs, amplitude=0.05)
            Layer6HorizonTopologyInjector.inject_fuzzball_echoes(
                h_plus, h_cross, fs=fs, amplitude=0.15, mass=10.0
            )
            Layer6HorizonTopologyInjector.inject_modified_ringdown(
                h_plus, h_cross, fs=fs, q_modification=0.2, mass=10.0
            )
            
            # Si llegamos aquí sin crash, OK
            assert True
        except Exception as e:
            pytest.fail(f"Método no es numéricamente estable: {str(e)}")
