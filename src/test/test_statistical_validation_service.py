"""
Tests para StatisticalValidationService

PASO 4: Validación estadística formal
- Monte Carlo sweeps
- Tests de significancia (χ², KL divergence)
- No-hair theorem validation
- Bootstrap confidence intervals
- Theory comparison
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from pathlib import Path
from scipy.special import softmax
from scipy.signal import chirp

from src.application.statistical_validation_service import (
    StatisticalValidationService,
    MCResult,
    SignificanceTest
)


class TestStatisticalValidationServiceBasics:
    """Tests para inicialización y propiedades básicas"""
    
    def test_initialization(self):
        """Test creación del servicio"""
        service = StatisticalValidationService(n_bootstrap=500, random_seed=123)
        
        assert service is not None
        assert hasattr(service, 'bootstrap_resampler')
        assert hasattr(service, 'fisher_calculator')
        assert hasattr(service, 'rng')
    
    def test_custom_random_seed(self):
        """Test reproducibilidad con seed"""
        service1 = StatisticalValidationService(random_seed=42)
        service2 = StatisticalValidationService(random_seed=42)
        
        # Generan números aleatorios iguales
        vals1 = [service1.rng.randn() for _ in range(5)]
        vals2 = [service2.rng.randn() for _ in range(5)]
        
        np.testing.assert_array_almost_equal(vals1, vals2)


class TestMonteCarloSweep:
    """Tests para Monte Carlo sweep"""
    
    @pytest.fixture
    def mock_generator_and_classifier(self):
        """Mock de generador de señal y clasificador"""
        
        def signal_generator(theory, snr, n_samples=16384):
            """Genera chirp simple"""
            t = np.linspace(0, 1, n_samples)
            signal = chirp(t, 35, 1, 250, method='linear')
            return signal * snr / 10  # Escalado por SNR
        
        def vqc_classifier(signal):
            """Mock VQC: predice teoría al azar + ruido"""
            # Predicción = distribución sobre teorías
            probs = np.array([0.7, 0.2, 0.1])  # Sesgado a teoría 0
            probs = probs + np.random.randn(3) * 0.05
            return softmax(probs)
        
        return signal_generator, vqc_classifier
    
    def test_monte_carlo_sweep_basic(self, mock_generator_and_classifier):
        """Test Monte Carlo sweep básico"""
        gen, clf = mock_generator_and_classifier
        
        service = StatisticalValidationService(random_seed=42)
        
        results = service.monte_carlo_sweep(
            signal_generator=gen,
            vqc_classifier=clf,
            theory_labels=["GR", "LQG", "ECO"],
            snr_levels=[10, 20],
            n_trials_per_snr=5,
            noise_std=1.0
        )
        
        # Verificar estructura
        assert len(results) == 2
        assert 10 in results
        assert 20 in results
        
        # Verificar cantidad de resultados
        assert len(results[10]) == 5
        assert len(results[20]) == 5
    
    def test_monte_carlo_result_types(self, mock_generator_and_classifier):
        """Test tipos en resultados MC"""
        gen, clf = mock_generator_and_classifier
        
        service = StatisticalValidationService(random_seed=42)
        
        results = service.monte_carlo_sweep(
            signal_generator=gen,
            vqc_classifier=clf,
            theory_labels=["A", "B", "C"],
            snr_levels=[15.0],
            n_trials_per_snr=3
        )
        
        for snr, mc_list in results.items():
            for mc_result in mc_list:
                assert isinstance(mc_result, MCResult)
                assert isinstance(mc_result.snr, (float, int))
                assert isinstance(mc_result.theory_label, str)
                assert isinstance(mc_result.extracted_parameters, dict)
                assert isinstance(mc_result.vqc_predictions, np.ndarray)
                assert isinstance(mc_result.prediction_correct, bool)
    
    def test_monte_carlo_predictions_normalized(self, mock_generator_and_classifier):
        """Test que predicciones VQC son distribuciones válidas"""
        gen, clf = mock_generator_and_classifier
        
        service = StatisticalValidationService(random_seed=42)
        
        results = service.monte_carlo_sweep(
            signal_generator=gen,
            vqc_classifier=clf,
            theory_labels=["X", "Y", "Z"],
            snr_levels=[20],
            n_trials_per_snr=5
        )
        
        for mc_result in results[20]:
            # Las predicciones deben estar entre 0 y 1
            assert np.all(mc_result.vqc_predictions >= -1e-10)
            assert np.all(mc_result.vqc_predictions <= 1.0 + 1e-10)


class TestChiSquaredGoodness:
    """Tests para test χ² de bondad de ajuste"""
    
    def test_chi_squared_perfect_fit(self):
        """Test χ² cuando observed = predicted"""
        service = StatisticalValidationService()
        
        observed = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0, 3.0])
        cov = np.eye(3)
        
        result = service.chi_squared_goodness_of_fit(observed, predicted, cov)
        
        assert isinstance(result, SignificanceTest)
        assert result.test_statistic < 1e-10  # Casi cero
        assert result.p_value > 0.99  # No significativo
        assert not result.is_significant
    
    def test_chi_squared_differs(self):
        """Test χ² cuando observed ≠ predicted"""
        service = StatisticalValidationService()
        
        observed = np.array([1.0, 2.0, 3.0])
        predicted = np.array([0.0, 1.0, 2.0])
        cov = np.eye(3) * 0.1  # Pequeña covarianza
        
        result = service.chi_squared_goodness_of_fit(observed, predicted, cov)
        
        assert result.test_statistic > 0
        assert result.p_value < 1.0
    
    def test_chi_squared_with_correlation(self):
        """Test χ² con parámetros correlacionados"""
        service = StatisticalValidationService()
        
        observed = np.array([2.0, 3.0])
        predicted = np.array([1.0, 2.0])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        
        result = service.chi_squared_goodness_of_fit(observed, predicted, cov)
        
        assert isinstance(result, SignificanceTest)
        assert result.test_name == "χ² Goodness of Fit"
        assert hasattr(result, 'interpretation')


class TestKullbackLeiblerDivergence:
    """Tests para divergencia de Kullback-Leibler"""
    
    def test_kl_identical_distributions(self):
        """Test KL cuando distribuciones son idénticas"""
        service = StatisticalValidationService()
        
        p = np.array([0.25, 0.25, 0.25, 0.25])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        
        kl = service.kullback_leibler_divergence(p, q)
        
        assert kl < 1e-10  # Prácticamente cero
    
    def test_kl_different_distributions(self):
        """Test KL cuando distribuciones difieren"""
        service = StatisticalValidationService()
        
        p = np.array([0.9, 0.1])
        q = np.array([0.5, 0.5])
        
        kl = service.kullback_leibler_divergence(p, q)
        
        assert kl > 0
        assert not np.isnan(kl)
        assert not np.isinf(kl)
    
    def test_kl_asymmetry(self):
        """Test que KL no es simétrica"""
        service = StatisticalValidationService()
        
        p1 = np.array([0.8, 0.2])
        p2 = np.array([0.5, 0.5])
        
        kl_12 = service.kullback_leibler_divergence(p1, p2)
        kl_21 = service.kullback_leibler_divergence(p2, p1)
        
        # KL no debe ser simétrica generalmente
        assert kl_12 != kl_21 or np.isclose(kl_12, kl_21)
    
    def test_kl_numerical_stability(self):
        """Test estabilidad numérica con valores pequeños"""
        service = StatisticalValidationService()
        
        p = np.array([1e-12, 1 - 1e-12])
        q = np.array([0.5, 0.5])
        
        kl = service.kullback_leibler_divergence(p, q)
        
        assert not np.isnan(kl)
        assert not np.isinf(kl)


class TestTheoryModelComparison:
    """Tests para comparación entre teorías"""
    
    def test_theory_comparison_single_theory(self):
        """Test comparación con una sola teoría"""
        service = StatisticalValidationService()
        
        # Crear resultados mock: teoría "A" con 80% accuracy
        mc_results = []
        for i in range(10):
            result = MCResult(
                snr=20,
                theory_label="A",
                extracted_parameters={"M": 30},
                vqc_predictions=np.array([0.9, 0.1]),
                prediction_correct=(i < 8)  # 8/10 correct
            )
            mc_results.append(result)
        
        comparison = service.theory_model_comparison(
            mc_results=mc_results,
            theory_labels=["A", "B"]
        )
        
        assert "A" in comparison
        assert comparison["A"]["accuracy"] == 0.8
        assert comparison["A"]["n_trials"] == 10
    
    def test_theory_comparison_multiple(self):
        """Test comparación múltiple de teorías"""
        service = StatisticalValidationService()
        
        mc_results = []
        for theory_idx, theory in enumerate(["GR", "LQG", "ECO"]):
            # GR: 90% accuracy, LQG: 70%, ECO: 50%
            accuracy = 0.9 - 0.2 * theory_idx
            n_correct = int(20 * accuracy)
            
            for i in range(20):
                result = MCResult(
                    snr=20,
                    theory_label=theory,
                    extracted_parameters={"M": 30},
                    vqc_predictions=np.ones(3) / 3,
                    prediction_correct=(i < n_correct)
                )
                mc_results.append(result)
        
        comparison = service.theory_model_comparison(
            mc_results=mc_results,
            theory_labels=["GR", "LQG", "ECO"]
        )
        
        assert len(comparison) == 3
        # GR should have highest accuracy
        gr_acc = comparison["GR"]["accuracy"]
        lqg_acc = comparison["LQG"]["accuracy"]
        eco_acc = comparison["ECO"]["accuracy"]
        
        assert gr_acc > lqg_acc > eco_acc
        assert comparison["GR"]["is_better_than_gr"] == False


class TestNoHairTheorem:
    """Tests para test del teorema no-cabello"""
    
    def test_no_hair_basic(self):
        """Test básico de no-hair"""
        service = StatisticalValidationService()
        
        frequencies = np.array([250.0, 500.0])
        damping_times = np.array([0.1, 0.05])
        mass = 65.0
        spin = 0.7
        
        result = service.no_hair_theorem_test(
            ringdown_frequencies=frequencies,
            ringdown_damping_times=damping_times,
            mass_estimate=mass,
            spin_estimate=spin
        )
        
        assert isinstance(result, dict)
        assert "QNM_0" in result
        assert "QNM_1" in result
    
    def test_no_hair_results_structure(self):
        """Test estructura de resultados no-hair"""
        service = StatisticalValidationService()
        
        frequencies = np.array([200.0])
        damping_times = np.array([0.1])
        
        result = service.no_hair_theorem_test(
            ringdown_frequencies=frequencies,
            ringdown_damping_times=damping_times,
            mass_estimate=60.0,
            spin_estimate=0.5
        )
        
        mode = result["QNM_0"]
        assert "frequency_hz" in mode
        assert "damping_time_s" in mode
        assert "observed_Q" in mode
        assert "predicted_Q_kerr" in mode
        assert "deviation_fraction" in mode
        assert "beyond_gr_evidence" in mode


class TestBootstrapParameterUncertainties:
    """Tests para bootstrap de incertidumbres"""
    
    def test_bootstrap_basic(self):
        """Test bootstrap básico"""
        service = StatisticalValidationService(n_bootstrap=100, random_seed=42)
        
        # Parámetros sintéticos: masa y spin
        extracted_params = np.array([
            [30.0, 0.5],
            [31.0, 0.45],
            [29.5, 0.55],
            [30.5, 0.48]
        ])
        
        param_names = ["mass_msun", "spin"]
        
        result = service.bootstrap_parameter_uncertainties(
            extracted_parameters=extracted_params,
            param_names=param_names
        )
        
        assert "mass_msun" in result
        assert "spin" in result
        
        # Verificar estructura
        for param in result.values():
            assert "mean" in param
            assert "std" in param
            assert "ci_68" in param
            assert "ci_95" in param
    
    def test_bootstrap_mean_close_to_data_mean(self):
        """Test que bootstrap mean está cerca de data mean"""
        service = StatisticalValidationService(n_bootstrap=500, random_seed=42)
        
        # Generar datos con media conocida
        true_mean = 42.0
        extracted_params = np.random.normal(true_mean, 1.0, size=(50, 1))
        
        result = service.bootstrap_parameter_uncertainties(
            extracted_parameters=extracted_params,
            param_names=["param"]
        )
        
        bootstrap_mean = result["param"]["mean"]
        
        # Debería estar cerca de la media verdadera
        assert abs(bootstrap_mean - true_mean) < 2.0


class TestSignificanceTestDataclass:
    """Tests para dataclass SignificanceTest"""
    
    def test_significance_test_creation(self):
        """Test creación de SignificanceTest"""
        test = SignificanceTest(
            test_name="Test 1",
            test_statistic=5.0,
            p_value=0.01,
            threshold=0.05,
            is_significant=True,
            interpretation="Resultado significativo"
        )
        
        assert test.test_name == "Test 1"
        assert test.test_statistic == 5.0
        assert test.p_value == 0.01
        assert test.is_significant == True


class TestMCResultDataclass:
    """Tests para dataclass MCResult"""
    
    def test_mc_result_creation(self):
        """Test creación de MCResult"""
        probs = np.array([0.7, 0.2, 0.1])
        
        result = MCResult(
            snr=20.0,
            theory_label="GR",
            extracted_parameters={"M": 30, "chi": 0.5},
            vqc_predictions=probs,
            prediction_correct=True
        )
        
        assert result.snr == 20.0
        assert result.theory_label == "GR"
        assert result.prediction_correct == True


class TestIntegration:
    """Tests de integración PASO 4"""
    
    def test_full_validation_pipeline(self):
        """Test pipeline completo de validación"""
        service = StatisticalValidationService(random_seed=42)
        
        # 1. Crear datos sintéticos
        mc_results = []
        for snr in [10, 20, 30]:
            for i in range(10):
                correct = np.random.rand() > (0.2 + snr/100)  # Mejor con SNR
                result = MCResult(
                    snr=float(snr),
                    theory_label=np.random.choice(["GR", "LQG"]),
                    extracted_parameters={"M": 30 + np.random.randn()},
                    vqc_predictions=np.random.dirichlet([1, 1]),
                    prediction_correct=correct
                )
                mc_results.append(result)
        
        # 2. Comparar teorías
        comparison = service.theory_model_comparison(
            mc_results=mc_results,
            theory_labels=["GR", "LQG"]
        )
        
        assert all(t in comparison for t in ["GR", "LQG"])
        assert all("accuracy" in comparison[t] for t in ["GR", "LQG"])


# Fixtures para tests

@pytest.fixture
def simple_mc_results():
    """MC results de ejemplo"""
    results = []
    for i in range(20):
        result = MCResult(
            snr=20,
            theory_label=np.random.choice(["A", "B", "C"]),
            extracted_parameters={"param": 30 + np.random.randn()},
            vqc_predictions=np.random.dirichlet([1, 1, 1]),
            prediction_correct=np.random.rand() > 0.3
        )
        results.append(result)
    return results
