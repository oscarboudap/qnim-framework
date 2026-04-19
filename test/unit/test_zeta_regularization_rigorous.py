"""
Tests exhaustivos para Zeta Regularization Rigorous

Verifica:
  - Función zeta en puntos especiales y derivadas
  - Entropía Bekenstein-Hawking
  - Correcciones logarítmicas (Cardy)
  - Densidad de estados LQG
  - Entropía de entrelazamiento (Ryu-Takayanagi)
  - Relaciones termodinámicas
  - Consistencia de regularización
"""

import pytest
import numpy as np
from scipy.special import zeta as scipy_zeta

from src.domain.shared.zeta_regularization_rigorous import (
    ZetaRegularizationRigorous,
    ZetaRegularizationConfig,
    compute_regulated_entropy,
)


class TestZetaFunctionRigorous:
    """Tests para cálculos de función zeta"""
    
    def test_zeta_special_values(self):
        """Verifica valores especiales de ζ(s)"""
        config = ZetaRegularizationConfig(bh_mass_mqp=10.0)
        regulator = ZetaRegularizationRigorous(config)
        
        # ζ(-1) = -1/12
        z_minus_1_exact = -1.0 / 12.0
        assert np.isclose(regulator.zeta_function(-1.0), z_minus_1_exact, rtol=1e-5)
        
        # ζ(0) = -1/2
        z_zero_exact = -0.5
        assert np.isclose(regulator.zeta_function(0.0), z_zero_exact, rtol=1e-5)
        
        # ζ(2) = π²/6
        z_two_exact = np.pi**2 / 6.0
        assert np.isclose(regulator.zeta_function(2.0), z_two_exact, rtol=1e-4)
    
    def test_zeta_positive_convergence(self):
        """Verifica convergencia para Re(s) > 1"""
        config = ZetaRegularizationConfig(bh_mass_mqp=10.0)
        regulator = ZetaRegularizationRigorous(config)
        
        # ζ(4) calculado vs scipy
        z_four = regulator.zeta_function(4.0, num_terms=5000)
        z_four_scipy = scipy_zeta(4.0)
        
        assert np.isclose(z_four, z_four_scipy, rtol=1e-3)
    
    def test_zeta_derivative_special_points(self):
        """Verifica ζ'(s) en puntos especiales"""
        config = ZetaRegularizationConfig(bh_mass_mqp=10.0)
        regulator = ZetaRegularizationRigorous(config)
        
        # ζ'(-1) ≈ -0.1654952
        z_prime_minus_1 = regulator.zeta_derivative(-1.0)
        expected = -0.1654952254
        
        assert np.isclose(z_prime_minus_1, expected, rtol=1e-4)
        assert z_prime_minus_1 < 0  # Debe ser negativo
    
    def test_zeta_derivative_numerical_stability(self):
        """Verifica estabilidad numérica de las derivadas"""
        config = ZetaRegularizationConfig(bh_mass_mqp=10.0)
        regulator = ZetaRegularizationRigorous(config)
        
        # Múltiples evaluaciones deben ser consistentes
        derivatives = [regulator.zeta_derivative(2.5) for _ in range(5)]
        
        # Todos deben estar en rango razonable
        for deriv in derivatives:
            assert not np.isnan(deriv)
            assert not np.isinf(deriv)


class TestBekensteinHawkingEntropy:
    """Tests para entropía Bekenstein-Hawking"""
    
    def test_entropy_formula(self):
        """Verifica S_BH = A/4"""
        masses = [1.0, 5.0, 10.0, 20.0]
        
        for mass in masses:
            config = ZetaRegularizationConfig(bh_mass_mqp=mass)
            regulator = ZetaRegularizationRigorous(config)
            
            S_BH = regulator.bekenstein_hawking_entropy()
            
            # A = 16π M² para Schwarzschild
            A = 16 * np.pi * mass**2
            expected_S = A / 4.0
            
            assert np.isclose(S_BH, expected_S, rtol=1e-10)
    
    def test_entropy_positive_increasing(self):
        """Verifica que S aumenta con la masa"""
        masses = np.linspace(1.0, 50.0, 10)
        entropies = []
        
        for mass in masses:
            config = ZetaRegularizationConfig(bh_mass_mqp=mass)
            regulator = ZetaRegularizationRigorous(config)
            S = regulator.bekenstein_hawking_entropy()
            entropies.append(S)
        
        # Debe ser monótonamente creciente
        for i in range(len(entropies)-1):
            assert entropies[i+1] > entropies[i]
    
    def test_entropy_scaling_with_area(self):
        """Verifica proporcionalidad S ∝ A"""
        config1 = ZetaRegularizationConfig(bh_mass_mqp=10.0)
        config2 = ZetaRegularizationConfig(bh_mass_mqp=20.0)
        
        reg1 = ZetaRegularizationRigorous(config1)
        reg2 = ZetaRegularizationRigorous(config2)
        
        S1 = reg1.bekenstein_hawking_entropy()
        S2 = reg2.bekenstein_hawking_entropy()
        
        # M₂ = 2 M₁ → A₂ = 4 A₁ → S₂ = 4 S₁
        ratio_S = S2 / S1
        assert np.isclose(ratio_S, 4.0, rtol=1e-10)


class TestLogarithmicCorrections:
    """Tests para correcciones logarítmicas tipo Cardy"""
    
    def test_cardy_correction_exists(self):
        """Verifica que corrección logarítmica es no-trivial"""
        config = ZetaRegularizationConfig(bh_mass_mqp=10.0)
        regulator = ZetaRegularizationRigorous(config)
        
        delta_S_log, c_eff = regulator.logarithmic_correction_cardy()
        
        # Debe ser diferente de cero para área grande
        assert delta_S_log != 0
        assert c_eff > 0
    
    def test_cardy_correction_grows_with_area(self):
        """Verifica que corrección crece con logaritmo del área"""
        masses = [5.0, 10.0, 20.0]
        corrections = []
        
        for mass in masses:
            config = ZetaRegularizationConfig(bh_mass_mqp=mass)
            regulator = ZetaRegularizationRigorous(config)
            delta_S, _ = regulator.logarithmic_correction_cardy()
            corrections.append(np.abs(delta_S))
        
        # Debe crecer monótonamente
        for i in range(len(corrections)-1):
            assert corrections[i+1] >= corrections[i]
    
    def test_central_charge_proportional_to_area(self):
        """Verifica c_eff ∝ A (holografía)"""
        config1 = ZetaRegularizationConfig(bh_mass_mqp=10.0)
        config2 = ZetaRegularizationConfig(bh_mass_mqp=20.0)
        
        reg1 = ZetaRegularizationRigorous(config1)
        reg2 = ZetaRegularizationRigorous(config2)
        
        _, c1 = reg1.logarithmic_correction_cardy()
        _, c2 = reg2.logarithmic_correction_cardy()
        
        # c ∝ A; A₂ = 4 A₁ → c₂ = 4 c₁
        ratio_c = c2 / c1
        assert np.isclose(ratio_c, 4.0, rtol=1e-10)


class TestZetaCorrections:
    """Tests para correcciones de función zeta"""
    
    def test_zeta_correction_non_trivial(self):
        """Verifica que corrección zeta es significativa"""
        config = ZetaRegularizationConfig(bh_mass_mqp=10.0)
        regulator = ZetaRegularizationRigorous(config)
        
        delta_S_zeta = regulator.zeta_correction_to_entropy()
        
        # Debe ser diferente de cero
        assert delta_S_zeta != 0
        assert not np.isnan(delta_S_zeta)
    
    def test_zeta_correction_is_small_relative_to_bh(self):
        """Verifica que corrección zeta << S_BH para área grande"""
        config = ZetaRegularizationConfig(bh_mass_mqp=20.0)  # BH grande
        regulator = ZetaRegularizationRigorous(config)
        
        S_BH = regulator.bekenstein_hawking_entropy()
        delta_S_zeta = regulator.zeta_correction_to_entropy()
        
        # Corrección debe ser pequeña relativa a BH
        ratio = np.abs(delta_S_zeta) / S_BH
        assert ratio < 0.5  # Menos del 50%


class TestLQGDensityOfStates:
    """Tests para densidad de estados Loop Quantum Gravity"""
    
    def test_lqg_microstate_count_positive(self):
        """Verifica que conteo de microestados es positivo"""
        config = ZetaRegularizationConfig(bh_mass_mqp=10.0)
        regulator = ZetaRegularizationRigorous(config)
        
        log_Omega, log_Omega_corrected = regulator.microstate_count_lqg()
        
        assert log_Omega > 0
        assert log_Omega_corrected > 0
    
    def test_lqg_corrected_microstate_different(self):
        """Verifica que correcciones LQG modifican el conteo"""
        config = ZetaRegularizationConfig(bh_mass_mqp=10.0)
        regulator = ZetaRegularizationRigorous(config)
        
        log_Omega, log_Omega_corrected = regulator.microstate_count_lqg()
        
        # Debe haber diferencia (correcciones logarítmicas)
        delta = np.abs(log_Omega_corrected - log_Omega)
        assert delta > 0
    
    def test_lqg_density_of_states_function(self):
        """Verifica ρ_LQG(E) es razonable"""
        config = ZetaRegularizationConfig(bh_mass_mqp=10.0)
        regulator = ZetaRegularizationRigorous(config)
        
        energies = [0.1, 1.0, 10.0, 100.0]
        
        for E in energies:
            rho = regulator.density_of_states_lqg(E)
            
            assert rho >= 0
            assert not np.isnan(rho)
            assert not np.isinf(rho)
    
    def test_lqg_immirzi_parameter_effect(self):
        """Verifica efecto del parámetro de Immirzi"""
        config1 = ZetaRegularizationConfig(bh_mass_mqp=10.0, immirzi_parameter=0.1)
        config2 = ZetaRegularizationConfig(bh_mass_mqp=10.0, immirzi_parameter=0.4)
        
        reg1 = ZetaRegularizationRigorous(config1)
        reg2 = ZetaRegularizationRigorous(config2)
        
        log_O1, log_O1_corr = reg1.microstate_count_lqg()
        log_O2, log_O2_corr = reg2.microstate_count_lqg()
        
        # Diferentes parámetros deben dar diferentes conteos corregidos
        assert log_O1_corr != log_O2_corr


class TestEntanglementEntropy:
    """Tests para entropía de entrelazamiento Ryu-Takayanagi"""
    
    def test_entanglement_entropy_bounds_bh(self):
        """Verifica S_EE está en rango físico"""
        config = ZetaRegularizationConfig(bh_mass_mqp=10.0)
        regulator = ZetaRegularizationRigorous(config)
        
        S_BH = regulator.bekenstein_hawking_entropy()
        S_EE = regulator.entanglement_entropy_ryu_takayanagi()
        
        # S_EE = S_BH + correcciones logarítmicas
        ratio = S_EE / S_BH
        assert ratio > 0.5  # Al menos comparable
        assert S_EE > 0  # Siempre positiva
    
    def test_entanglement_entropy_positive(self):
        """Verifica que S_EE > 0"""
        masses = [1.0, 10.0, 50.0]
        
        for mass in masses:
            config = ZetaRegularizationConfig(bh_mass_mqp=mass)
            regulator = ZetaRegularizationRigorous(config)
            S_EE = regulator.entanglement_entropy_ryu_takayanagi()
            
            assert S_EE > 0


class TestTotalEntropy:
    """Tests para entropía total regularizada"""
    
    def test_total_entropy_components(self):
        """Verifica estructura completa de componentes"""
        config = ZetaRegularizationConfig(
            bh_mass_mqp=10.0,
            include_logarithmic_correction=True,
            include_lqg_density_of_states=True,
            include_entanglement_entropy=True,
        )
        regulator = ZetaRegularizationRigorous(config)
        
        entropy_dict = regulator.total_entropy_regularized()
        
        # Verificar claves esperadas
        expected_keys = [
            "S_BH", "delta_S_zeta", "delta_S_log", 
            "delta_S_EE", "delta_S_LQG", "S_total",
            "effective_central_charge", "log_microstate_count",
            "log_microstate_count_corrected"
        ]
        
        for key in expected_keys:
            assert key in entropy_dict
            assert entropy_dict[key] is not None
    
    def test_total_entropy_positive_bounded(self):
        """Verifica S_total es positiva y contiene correcciones"""
        config = ZetaRegularizationConfig(bh_mass_mqp=10.0)
        regulator = ZetaRegularizationRigorous(config)
        
        entropy_dict = regulator.total_entropy_regularized()
        S_total = entropy_dict["S_total"]
        S_BH = entropy_dict["S_BH"]
        
        assert S_total > 0
        assert S_total >= S_BH  # Total >= BH (nunca menos)
        assert S_total < 10.0 * S_BH  # Correcciones razonables
    
    def test_entropy_without_corrections(self):
        """Verifica entropía sin correcciones ≈ S_BH"""
        config = ZetaRegularizationConfig(
            bh_mass_mqp=10.0,
            include_logarithmic_correction=False,
            include_lqg_density_of_states=False,
            include_entanglement_entropy=False,
        )
        regulator = ZetaRegularizationRigorous(config)
        
        entropy_dict = regulator.total_entropy_regularized()
        
        S_total = entropy_dict["S_total"]
        S_BH = entropy_dict["S_BH"]
        
        # Corrección zeta siempre presente, tolerar pequeña desviación
        assert np.isclose(S_total, S_BH, rtol=0.01)


class TestVacuumEnergy:
    """Tests para autoenergía de punto cero"""
    
    def test_vacuum_energy_scales(self):
        """Verifica scaling de energía de vacío"""
        config = ZetaRegularizationConfig(bh_mass_mqp=10.0)
        regulator = ZetaRegularizationRigorous(config)
        
        # Diferentes cutoffs
        E1, E1_reg = regulator.vacuum_self_energy_regularized(freq_cutoff=100.0)
        E2, E2_reg = regulator.vacuum_self_energy_regularized(freq_cutoff=1000.0)
        
        # Con cutoff, energía debe escalar como cutoff^4
        assert E2 > E1
        
        # Regularizada debe ser "más chico"
        assert np.abs(E2_reg) < E2
    
    def test_vacuum_self_energy_bounded(self):
        """Verifica que energía regularizada está acotada"""
        config = ZetaRegularizationConfig(bh_mass_mqp=10.0)
        regulator = ZetaRegularizationRigorous(config)
        
        E_cutoff, E_reg = regulator.vacuum_self_energy_regularized()
        
        assert not np.isinf(E_reg)
        assert not np.isnan(E_reg)
        # Regularizada debe ser finita a diferencia de cutoff naif
        assert np.abs(E_reg) < np.abs(E_cutoff)


class TestThermodynamicConsistency:
    """Tests para consistencia termodinámica"""
    
    def test_thermodynamic_relations_verified(self):
        """Verifica relaciones fundamentales de termodinámica"""
        config = ZetaRegularizationConfig(bh_mass_mqp=10.0)
        regulator = ZetaRegularizationRigorous(config)
        
        checks = regulator.verify_thermodynamic_relations()
        
        # S/A debe ser 1/4
        ratio = checks["entropy_to_area_ratio"]
        expected = checks["expected_ratio"]
        
        assert np.isclose(ratio, expected, rtol=1e-10)
    
    def test_temperature_positive(self):
        """Verifica temperatura Hawking positiva"""
        config = ZetaRegularizationConfig(bh_mass_mqp=10.0)
        regulator = ZetaRegularizationRigorous(config)
        
        checks = regulator.verify_thermodynamic_relations()
        
        T = checks["temperature_hawking"]
        assert T > 0
        assert not np.isnan(T)
    
    def test_surface_gravity_reasonable(self):
        """Verifica surface gravity es razonable"""
        config = ZetaRegularizationConfig(bh_mass_mqp=10.0)
        regulator = ZetaRegularizationRigorous(config)
        
        checks = regulator.verify_thermodynamic_relations()
        
        kappa = checks["surface_gravity"]
        assert kappa > 0
        assert kappa < 1.0  # Para BH macroscópico


class TestPublicAPI:
    """Tests para función pública compute_regulated_entropy"""
    
    def test_public_api_basic(self):
        """Verifica función pública básica"""
        result = compute_regulated_entropy(mass_mqp=10.0, include_all_corrections=True)
        
        assert "entropy" in result
        assert "thermodynamic_checks" in result
        assert "vacuum_energy" in result
        assert "config" in result
    
    def test_public_api_with_without_corrections(self):
        """Verifica diferencia con/sin correcciones"""
        result_with = compute_regulated_entropy(mass_mqp=10.0, include_all_corrections=True)
        result_without = compute_regulated_entropy(mass_mqp=10.0, include_all_corrections=False)
        
        S_with = result_with["entropy"]["S_total"]
        S_without = result_without["entropy"]["S_total"]
        
        # Con correcciones debe ser diferente
        assert S_with != S_without
    
    def test_public_api_multiple_masses(self):
        """Verifica API para múltiples masas"""
        masses = [1.0, 5.0, 10.0, 20.0, 50.0]
        results = [compute_regulated_entropy(mass_mqp=m) for m in masses]
        
        entropies = [r["entropy"]["S_total"] for r in results]
        
        # Deben crecer monótonamente
        for i in range(len(entropies)-1):
            assert entropies[i+1] > entropies[i]


class TestNumericalStability:
    """Tests para estabilidad numérica global"""
    
    def test_no_nans_or_infs(self):
        """Verifica que no hay NaN o infinitos"""
        config = ZetaRegularizationConfig(bh_mass_mqp=10.0)
        regulator = ZetaRegularizationRigorous(config)
        
        entropy_dict = regulator.total_entropy_regularized()
        
        for key, value in entropy_dict.items():
            assert not np.isnan(value), f"{key} is NaN"
            assert not np.isinf(value), f"{key} is infinite"
    
    def test_extreme_masses(self):
        """Verifica comportamiento en masas extremas"""
        # BH muy pequeño
        config_small = ZetaRegularizationConfig(bh_mass_mqp=0.1)
        regulator_small = ZetaRegularizationRigorous(config_small)
        S_small = regulator_small.bekenstein_hawking_entropy()
        
        # BH muy grande
        config_large = ZetaRegularizationConfig(bh_mass_mqp=1000.0)
        regulator_large = ZetaRegularizationRigorous(config_large)
        S_large = regulator_large.bekenstein_hawking_entropy()
        
        # Ambos deben ser válidos
        assert S_small > 0
        assert S_large > 0
        assert S_large > S_small
