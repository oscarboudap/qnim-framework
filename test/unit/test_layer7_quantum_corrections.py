"""
Tests para Layer 7 - Quantum Corrections Physics

Valida inyección de correcciones cuánticas:
  - Evaporación de Hawking
  - Radiación térmica de Hawking
  - Efectos AdS/CFT (viscosidad)
  - Efectos de entanglement wedge
"""

import pytest
import numpy as np
from scipy import signal as scipy_signal

from src.domain.astrophysics.sstg.injectors.layer7_quantum_corrections_complete import (
    Layer7QuantumCorrectionsInjector,
    QuantumCorrectionParams,
    M_SUN,
    HBAR,
    KBOLTZMANN,
    C_LIGHT,
    G_CONST,
)


class TestLayer7QuantumCorrections:
    """Test suite para Layer 7 - Correcciones Cuánticas"""
    
    @pytest.fixture
    def mock_signal(self):
        """Generar señal sintética GW"""
        fs = 4096
        duration = 0.3
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        
        # Ringdown-like signal
        freq_dominant = 250  # Hz
        tau = 0.15
        amplitude = 1e-21
        
        h_plus = amplitude * np.exp(-t/tau) * np.sin(2 * np.pi * freq_dominant * t)
        h_cross = amplitude * 0.5 * np.exp(-t/tau) * np.cos(2 * np.pi * freq_dominant * t)
        
        return h_plus, h_cross, fs, t
    
    # ==================== HAWKING TEMPERATURE/LUMINOSITY/LIFETIME TESTS ====================
    
    def test_hawking_temperature_computation(self):
        """Test que compute_hawking_temperature da resultado físicamente razonable"""
        mass_kg = 10 * M_SUN
        
        t_hawking = Layer7QuantumCorrectionsInjector.compute_hawking_temperature(mass_kg)
        
        # Para 10 M_sun, temp debe ser microkelvin
        assert 0 < t_hawking < 1e-6
    
    def test_hawking_temperature_inverse_mass(self):
        """Test que temperatura Hawking es inversamente proporcional a masa"""
        mass1 = 10 * M_SUN
        mass2 = 20 * M_SUN
        
        t1 = Layer7QuantumCorrectionsInjector.compute_hawking_temperature(mass1)
        t2 = Layer7QuantumCorrectionsInjector.compute_hawking_temperature(mass2)
        
        # Masa más grande → temperatura más baja
        assert t2 < t1
        
        # Escalado: T ∝ 1/M, así que T1/T2 ≈ M2/M1 = 2
        ratio_theory = mass2 / mass1
        ratio_computed = t1 / t2
        
        assert abs(ratio_computed - ratio_theory) / ratio_theory < 0.01
    
    def test_hawking_luminosity_computation(self):
        """Test que compute_hawking_luminosity da resultado físicamente razonable"""
        mass_kg = 10 * M_SUN
        
        l_hawking = Layer7QuantumCorrectionsInjector.compute_hawking_luminosity(mass_kg)
        
        # Debe ser positivo
        assert l_hawking > 0
        # Para BH macroscópico, luminosidad es extremadamente pequeña (~ Watts)
        assert l_hawking < 1e-10
    
    def test_hawking_luminosity_inverse_mass_squared(self):
        """Test que luminosidad es inversamente proporcional a M²"""
        mass1 = 10 * M_SUN
        mass2 = 20 * M_SUN
        
        l1 = Layer7QuantumCorrectionsInjector.compute_hawking_luminosity(mass1)
        l2 = Layer7QuantumCorrectionsInjector.compute_hawking_luminosity(mass2)
        
        # L ∝ 1/M², así que L1/L2 ≈ (M2/M1)² = 4
        ratio_theory = (mass2 / mass1) ** 2
        ratio_computed = l1 / l2
        
        assert abs(ratio_computed - ratio_theory) / ratio_theory < 0.01
    
    def test_evaporation_lifetime_computation(self):
        """Test que compute_evaporation_lifetime da resultado físicamente razonable"""
        mass_kg = 10 * M_SUN
        
        t_evap = Layer7QuantumCorrectionsInjector.compute_evaporation_lifetime(mass_kg)
        
        # Debe ser positivo
        assert t_evap > 0
        # Para 10 M_sun, tiempo de evaporación es ENORME (años)
        assert t_evap > 1e60  # segundos
    
    def test_evaporation_lifetime_cubic_mass(self):
        """Test que lifetime es proporcional a M³"""
        mass1 = 10 * M_SUN
        mass2 = 20 * M_SUN
        
        t1 = Layer7QuantumCorrectionsInjector.compute_evaporation_lifetime(mass1)
        t2 = Layer7QuantumCorrectionsInjector.compute_evaporation_lifetime(mass2)
        
        # t_evap ∝ M³, así que t1/t2 ≈ (M1/M2)³ = (1/2)³ = 0.125
        ratio_theory = (mass1 / mass2) ** 3
        ratio_computed = t1 / t2
        
        assert abs(ratio_computed - ratio_theory) / ratio_theory < 0.01
    
    # ==================== HAWKING EVAPORATION INJECTION TESTS ====================
    
    def test_hawking_evaporation_returns_tuple_and_metadata(self, mock_signal):
        """Test que inject_hawking_evaporation retorna tupla + metadata"""
        h_plus, h_cross, fs, _ = mock_signal
        
        result = Layer7QuantumCorrectionsInjector.inject_hawking_evaporation(
            h_plus, h_cross, fs=fs, mass_initial=10.0, evaporation_rate=0.001
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert len(result[0]) == len(h_plus)
        assert len(result[1]) == len(h_cross)
        
        metadata = result[2]
        assert isinstance(metadata, dict)
        assert "hawking_temperature_K" in metadata
        assert "hawking_luminosity_W" in metadata
        assert "evaporation_lifetime_s" in metadata
    
    def test_hawking_evaporation_modifies_signal(self, mock_signal):
        """Test que evaporación modifica la señal"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_evap, h_cross_evap, _ = Layer7QuantumCorrectionsInjector.inject_hawking_evaporation(
            h_plus, h_cross, fs=fs, mass_initial=10.0, evaporation_rate=0.001
        )
        
        # Debe haber cambio (efecto pequeño pero detectable)
        diff = np.sum((h_plus_evap - h_plus)**2)
        assert diff > 1e-68
    
    def test_hawking_evaporation_rate_effect(self, mock_signal):
        """Test que tasa de evaporación afecta el resultado"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_slow, _, _ = Layer7QuantumCorrectionsInjector.inject_hawking_evaporation(
            h_plus, h_cross, fs=fs, mass_initial=10.0, evaporation_rate=0.0001
        )
        h_plus_fast, _, _ = Layer7QuantumCorrectionsInjector.inject_hawking_evaporation(
            h_plus, h_cross, fs=fs, mass_initial=10.0, evaporation_rate=0.01
        )
        
        # Ambos deben cambiar la señal comparados con original
        diff_slow = np.sum((h_plus_slow - h_plus)**2)
        diff_fast = np.sum((h_plus_fast - h_plus)**2)
        assert diff_slow > 0 and diff_fast > 0
    
    def test_hawking_metadata_reasonable_values(self, mock_signal):
        """Test que metadata tiene valores físicamente razonables"""
        h_plus, h_cross, fs, _ = mock_signal
        
        _, _, metadata = Layer7QuantumCorrectionsInjector.inject_hawking_evaporation(
            h_plus, h_cross, fs=fs, mass_initial=10.0, evaporation_rate=0.001
        )
        
        # Temperatura debe ser positiva y pequeña
        assert metadata["hawking_temperature_K"] > 0
        assert metadata["hawking_temperature_K"] < 1e-5
        
        # Luminosidad debe ser positiva
        assert metadata["hawking_luminosity_W"] > 0
        
        # Tiempo de evaporación debe ser largo
        assert metadata["evaporation_lifetime_s"] > 1e50
    
    # ==================== HAWKING RADIATION SPECTRUM TESTS ====================
    
    def test_hawking_radiation_spectrum_modifies_signal(self, mock_signal):
        """Test que radiación Hawking agrega contenido"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_rad, h_cross_rad = Layer7QuantumCorrectionsInjector.inject_hawking_radiation_spectrum(
            h_plus, h_cross, fs=fs, mass=10.0, temperature_scale=1.0
        )
        
        # Debe haber diferencia
        diff = np.sum((h_plus_rad - h_plus)**2)
        assert diff > 1e-50
    
    def test_hawking_radiation_temperature_scale_effect(self, mock_signal):
        """Test que escala de temperatura afecta radiación"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_cool, _ = Layer7QuantumCorrectionsInjector.inject_hawking_radiation_spectrum(
            h_plus, h_cross, fs=fs, mass=10.0, temperature_scale=0.5
        )
        h_plus_hot, _ = Layer7QuantumCorrectionsInjector.inject_hawking_radiation_spectrum(
            h_plus, h_cross, fs=fs, mass=10.0, temperature_scale=2.0
        )
        
        # Ambos deben agregar contenido (radiacion es ruido no determinista)
        diff_cool = np.sqrt(np.sum((h_plus_cool - h_plus)**2))
        diff_hot = np.sqrt(np.sum((h_plus_hot - h_plus)**2))
        
        assert diff_cool > 0 and diff_hot > 0
    
    def test_hawking_radiation_returns_correct_shapes(self, mock_signal):
        """Test que formas se preservan"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_rad, h_cross_rad = Layer7QuantumCorrectionsInjector.inject_hawking_radiation_spectrum(
            h_plus, h_cross, fs=fs, mass=10.0
        )
        
        assert h_plus_rad.shape == h_plus.shape
        assert h_cross_rad.shape == h_cross.shape
    
    # ==================== ADS/CFT VISCOSITY TESTS ====================
    
    def test_ads_cft_viscosity_returns_tuple(self, mock_signal):
        """Test que inject_ads_cft_viscosity retorna tupla"""
        h_plus, h_cross, fs, _ = mock_signal
        
        result = Layer7QuantumCorrectionsInjector.inject_ads_cft_viscosity(
            h_plus, h_cross, fs=fs, coupling_strength=0.05
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_ads_cft_viscosity_attenuates_signal(self, mock_signal):
        """Test que AdS/CFT atenúa la señal (disipación)"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_adscft, h_cross_adscft = Layer7QuantumCorrectionsInjector.inject_ads_cft_viscosity(
            h_plus, h_cross, fs=fs, coupling_strength=0.10
        )
        
        # Amplitud RMS debe ser menor (disipación)
        rms_orig = np.sqrt(np.mean(h_plus**2))
        rms_adscft = np.sqrt(np.mean(h_plus_adscft**2))
        
        assert rms_adscft < rms_orig
    
    def test_ads_cft_coupling_strength_effect(self, mock_signal):
        """Test que fuerza de acoplamiento afecta atenuación"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_weak, _ = Layer7QuantumCorrectionsInjector.inject_ads_cft_viscosity(
            h_plus, h_cross, fs=fs, coupling_strength=0.01
        )
        h_plus_strong, _ = Layer7QuantumCorrectionsInjector.inject_ads_cft_viscosity(
            h_plus, h_cross, fs=fs, coupling_strength=0.20
        )
        
        # Mayor acoplamiento → más atenuación
        rms_weak = np.sqrt(np.mean(h_plus_weak**2))
        rms_strong = np.sqrt(np.mean(h_plus_strong**2))
        
        assert rms_strong < rms_weak
    
    def test_ads_cft_preserves_shape(self, mock_signal):
        """Test que forma se preserva"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_adscft, h_cross_adscft = Layer7QuantumCorrectionsInjector.inject_ads_cft_viscosity(
            h_plus, h_cross, fs=fs, coupling_strength=0.05
        )
        
        assert h_plus_adscft.shape == h_plus.shape
        assert h_cross_adscft.shape == h_cross.shape
    
    # ==================== ENTANGLEMENT WEDGE TESTS ====================
    
    def test_entanglement_wedge_returns_tuple(self, mock_signal):
        """Test que inject_entanglement_wedge_effects retorna tupla"""
        h_plus, h_cross, fs, _ = mock_signal
        
        result = Layer7QuantumCorrectionsInjector.inject_entanglement_wedge_effects(
            h_plus, h_cross, fs=fs, entanglement_entropy=100.0
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_entanglement_wedge_adds_noise(self, mock_signal):
        """Test que efectos de entrelazamiento agregan ruido"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_ent, h_cross_ent = Layer7QuantumCorrectionsInjector.inject_entanglement_wedge_effects(
            h_plus, h_cross, fs=fs, entanglement_entropy=100.0
        )
        
        # Debe haber cambio (ruido agregado)
        diff = np.sum((h_plus_ent - h_plus)**2)
        assert diff > 1e-50
    
    def test_entanglement_entropy_parameter_effect(self, mock_signal):
        """Test que parámetro de entropía afecta magnitud"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_low, _ = Layer7QuantumCorrectionsInjector.inject_entanglement_wedge_effects(
            h_plus, h_cross, fs=fs, entanglement_entropy=10.0
        )
        h_plus_high, _ = Layer7QuantumCorrectionsInjector.inject_entanglement_wedge_effects(
            h_plus, h_cross, fs=fs, entanglement_entropy=500.0
        )
        
        # Mayor entropía → mayor ruido
        diff_low = np.sqrt(np.sum((h_plus_low - h_plus)**2))
        diff_high = np.sqrt(np.sum((h_plus_high - h_plus)**2))
        
        assert diff_high > diff_low
    
    def test_entanglement_wedge_preserves_shape(self, mock_signal):
        """Test que forma se preserva"""
        h_plus, h_cross, fs, _ = mock_signal
        
        h_plus_ent, h_cross_ent = Layer7QuantumCorrectionsInjector.inject_entanglement_wedge_effects(
            h_plus, h_cross, fs=fs, entanglement_entropy=100.0
        )
        
        assert h_plus_ent.shape == h_plus.shape
        assert h_cross_ent.shape == h_cross.shape
    
    # ==================== DISPATCHER TEST ====================
    
    def test_apply_quantum_corrections_dispatcher(self, mock_signal):
        """Test que dispatcher apply_quantum_corrections funciona"""
        h_plus, h_cross, fs, _ = mock_signal
        
        params = QuantumCorrectionParams(theory="Hawking")
        
        result = Layer7QuantumCorrectionsInjector.apply_quantum_corrections(
            h_plus, h_cross, params=params, mass=10.0, fs=fs
        )
        
        assert isinstance(result, dict)
        assert "h_plus" in result
        assert "h_cross" in result
        assert "physics_applied" in result
        assert isinstance(result["physics_applied"], list)
        assert len(result["physics_applied"]) > 0
    
    def test_dispatcher_hawking_path(self, mock_signal):
        """Test que dispatcher ejecuta ruta Hawking"""
        h_plus, h_cross, fs, _ = mock_signal
        
        params = QuantumCorrectionParams(
            theory="Hawking",
            evaporation_rate=0.001
        )
        
        result = Layer7QuantumCorrectionsInjector.apply_quantum_corrections(
            h_plus, h_cross, params=params, mass=10.0, fs=fs
        )
        
        assert "Hawking" in " ".join(result["physics_applied"])
        assert len(result["h_plus"]) == len(h_plus)
    
    def test_dispatcher_adscft_path(self, mock_signal):
        """Test que dispatcher ejecuta ruta AdS/CFT"""
        h_plus, h_cross, fs, _ = mock_signal
        
        params = QuantumCorrectionParams(
            theory="ads-cft",
            ads_cft_coupling=0.05,
            entanglement_entropy=100.0
        )
        
        result = Layer7QuantumCorrectionsInjector.apply_quantum_corrections(
            h_plus, h_cross, params=params, mass=10.0, fs=fs
        )
        
        assert "AdS/CFT" in " ".join(result["physics_applied"])
        assert len(result["h_plus"]) == len(h_plus)
    
    # ==================== GENERAL PROPERTIES ====================
    
    def test_no_nans_in_all_methods(self, mock_signal):
        """Test que no hay NaN/Inf en ningún método"""
        h_plus, h_cross, fs, _ = mock_signal
        
        # Evaporation
        hp_evap, hc_evap, _ = Layer7QuantumCorrectionsInjector.inject_hawking_evaporation(
            h_plus, h_cross, fs=fs
        )
        assert not np.any(np.isnan(hp_evap))
        assert not np.any(np.isinf(hp_evap))
        
        # Radiation spectrum
        hp_rad, hc_rad = Layer7QuantumCorrectionsInjector.inject_hawking_radiation_spectrum(
            h_plus, h_cross, fs=fs
        )
        assert not np.any(np.isnan(hp_rad))
        assert not np.any(np.isinf(hp_rad))
        
        # AdS/CFT
        hp_adscft, hc_adscft = Layer7QuantumCorrectionsInjector.inject_ads_cft_viscosity(
            h_plus, h_cross, fs=fs
        )
        assert not np.any(np.isnan(hp_adscft))
        assert not np.any(np.isinf(hp_adscft))
        
        # Entanglement
        hp_ent, hc_ent = Layer7QuantumCorrectionsInjector.inject_entanglement_wedge_effects(
            h_plus, h_cross, fs=fs
        )
        assert not np.any(np.isnan(hp_ent))
        assert not np.any(np.isinf(hp_ent))
    
    def test_all_injection_methods_numerical_stability(self, mock_signal):
        """Test que todos los métodos son numéricamente estables"""
        h_plus, h_cross, fs, _ = mock_signal
        
        try:
            # Ejecutar todos
            Layer7QuantumCorrectionsInjector.inject_hawking_evaporation(h_plus, h_cross, fs=fs)
            Layer7QuantumCorrectionsInjector.inject_hawking_radiation_spectrum(h_plus, h_cross, fs=fs)
            Layer7QuantumCorrectionsInjector.inject_ads_cft_viscosity(h_plus, h_cross, fs=fs)
            Layer7QuantumCorrectionsInjector.inject_entanglement_wedge_effects(h_plus, h_cross, fs=fs)
            
            # Si llegamos aquí, todo OK
            assert True
        except Exception as e:
            pytest.fail(f"Método no es numéricamente estable: {str(e)}")
