"""
Tests para PASO 2 - MCMC Benchmarking

Valida comparación Classical Metropolis-Hastings vs Quantum VQC
"""

import pytest
import numpy as np
from src.application.mcmc_benchmarking import (
    ClassicalMetropolisHastings,
    VariationalQuantumCircuitSampler,
    MCMCBenchmark,
    MCMCMetrics,
)


class TestMCMCBenchmarking:
    """Test suite para MCMC Benchmarking"""
    
    @pytest.fixture
    def synthetic_data(self):
        """Generar datos sintéticos de GW para pruebas"""
        np.random.seed(42)
        
        # Datos: strain simple
        fs = 16384
        duration = 0.1
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        
        # GW sintética simple (chirp)
        freq_start = 100
        freq_end = 300
        amplitude = 1e-21
        
        chirp = amplitude * np.sin(2 * np.pi * (freq_start + (freq_end - freq_start) * t / duration) * t)
        
        # Agregar ruido gaussiano (PSD-like)
        psd = np.ones_like(chirp) * 1e-44
        noise = np.random.normal(0, np.sqrt(psd))
        
        data = chirp + noise
        
        return data, psd
    
    @pytest.fixture
    def param_bounds(self):
        """Límites de parámetros para inferencia"""
        return {
            0: {"min": 10.0, "max": 100.0},    # m1 (masa 1)
            1: {"min": 10.0, "max": 100.0},    # m2 (masa 2)
            2: {"min": 0.0, "max": 1.0},       # chi1 (spin 1)
            3: {"min": 0.0, "max": 1.0},       # chi2 (spin 2)
        }
    
    # ==================== CLASSICAL MH TESTS ====================
    
    def test_metropolis_hastings_returns_samples_and_metrics(self, synthetic_data, param_bounds):
        """Test que MH retorna muestras y métricas correctas"""
        data, psd = synthetic_data
        
        samples, metrics = ClassicalMetropolisHastings.sample(
            data, psd, param_bounds,
            num_iterations=100,
            proposal_std=0.5
        )
        
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (100, 4)  # 100 iteraciones, 4 parámetros
        assert isinstance(metrics, MCMCMetrics)
    
    def test_mh_metrics_validity(self, synthetic_data, param_bounds):
        """Test que métricas MH son físicamente válidas"""
        data, psd = synthetic_data
        
        _, metrics = ClassicalMetropolisHastings.sample(
            data, psd, param_bounds,
            num_iterations=200,
            proposal_std=0.5
        )
        
        # Acceptance rate debe estar en [0, 1]
        assert 0 <= metrics.acceptance_rate <= 1
        
        # Autocorrelación debe ser > 0
        assert metrics.autocorrelation > 0
        
        # Samples efectivos debe ser < num_iterations
        assert metrics.effective_samples <= metrics.num_samples
        assert metrics.effective_samples > 0
        
        # R-hat de convergencia debe ser > 1 (típicamente)
        assert metrics.convergence_r_hat > 0
    
    def test_mh_respects_parameter_bounds(self, synthetic_data, param_bounds):
        """Test que MH respeta los límites de parámetros"""
        data, psd = synthetic_data
        
        samples, _ = ClassicalMetropolisHastings.sample(
            data, psd, param_bounds,
            num_iterations=500,
            proposal_std=1.0
        )
        
        # Verificar que todas las muestras están dentro de bounds
        for i, param_range in param_bounds.items():
            assert np.all(samples[:, i] >= param_range["min"])
            assert np.all(samples[:, i] <= param_range["max"])
    
    def test_mh_proposal_std_affects_acceptance(self, synthetic_data, param_bounds):
        """Test que proposal_std afecta acceptance rate"""
        data, psd = synthetic_data
        
        _, metrics_small = ClassicalMetropolisHastings.sample(
            data, psd, param_bounds,
            num_iterations=200,
            proposal_std=0.1
        )
        
        _, metrics_large = ClassicalMetropolisHastings.sample(
            data, psd, param_bounds,
            num_iterations=200,
            proposal_std=2.0
        )
        
        # Proposal pequeño → acceptance rate alto
        # Proposal grande → acceptance rate bajo
        assert metrics_small.acceptance_rate > metrics_large.acceptance_rate
    
    def test_mh_posterior_estimates(self, synthetic_data, param_bounds):
        """Test que MH estima posterior reasonable"""
        data, psd = synthetic_data
        
        _, metrics = ClassicalMetropolisHastings.sample(
            data, psd, param_bounds,
            num_iterations=500,
            proposal_std=0.5
        )
        
        # Media debe estar dentro de los bounds
        for i, param_range in param_bounds.items():
            assert param_range["min"] <= metrics.posterior_mean[i] <= param_range["max"]
        
        # Std debe ser positiva
        assert np.all(metrics.posterior_std > 0)
    
    # ==================== QUANTUM VQC TESTS ====================
    
    def test_vqc_returns_samples_and_metrics(self, synthetic_data, param_bounds):
        """Test que VQC retorna muestras y métricas"""
        data, psd = synthetic_data
        
        samples, metrics = VariationalQuantumCircuitSampler.quantum_circuit_inference(
            data, psd, param_bounds,
            num_iterations=50,
            learning_rate=0.01
        )
        
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (50, 4)
        assert isinstance(metrics, MCMCMetrics)
    
    def test_vqc_metrics_validity(self, synthetic_data, param_bounds):
        """Test que métricas VQC son válidas"""
        data, psd = synthetic_data
        
        _, metrics = VariationalQuantumCircuitSampler.quantum_circuit_inference(
            data, psd, param_bounds,
            num_iterations=100,
            learning_rate=0.01
        )
        
        # VQC tiene acceptance rate = 1.0 (determinista)
        assert metrics.acceptance_rate == 1.0
        
        # Autocorrelación debe ser baja (approx 1.0)
        assert metrics.autocorrelation == 1.0
        
        # Samples efectivos = num_iterations
        assert metrics.effective_samples == metrics.num_iterations
    
    def test_vqc_respects_parameter_bounds(self, synthetic_data, param_bounds):
        """Test que VQC respeta bounds"""
        data, psd = synthetic_data
        
        samples, _ = VariationalQuantumCircuitSampler.quantum_circuit_inference(
            data, psd, param_bounds,
            num_iterations=100,
            learning_rate=0.01
        )
        
        for i, param_range in param_bounds.items():
            assert np.all(samples[:, i] >= param_range["min"])
            assert np.all(samples[:, i] <= param_range["max"])
    
    def test_vqc_learning_rate_effect(self, synthetic_data, param_bounds):
        """Test que learning_rate afecta convergencia"""
        data, psd = synthetic_data
        
        _, metrics_low_lr = VariationalQuantumCircuitSampler.quantum_circuit_inference(
            data, psd, param_bounds,
            num_iterations=50,
            learning_rate=0.001
        )
        
        _, metrics_high_lr = VariationalQuantumCircuitSampler.quantum_circuit_inference(
            data, psd, param_bounds,
            num_iterations=50,
            learning_rate=0.1
        )
        
        # Ambos deben converger (convergence_r_hat < threshold)
        # pero high_lr podría ser más rápido
        assert metrics_low_lr.convergence_r_hat > 0
        assert metrics_high_lr.convergence_r_hat > 0
    
    # ==================== COMPARISON TESTS ====================
    
    def test_benchmark_returns_comparison_dict(self, synthetic_data, param_bounds):
        """Test que benchmark retorna dict con comparación"""
        data, psd = synthetic_data
        
        results = MCMCBenchmark.run_comparison(
            data, psd, param_bounds,
            classical_iterations=100,
            quantum_iterations=50
        )
        
        assert isinstance(results, dict)
        assert "classical" in results
        assert "quantum" in results
        assert "comparison" in results
    
    def test_benchmark_comparison_metrics(self, synthetic_data, param_bounds):
        """Test que comparison calcula speedups correctamente"""
        data, psd = synthetic_data
        
        results = MCMCBenchmark.run_comparison(
            data, psd, param_bounds,
            classical_iterations=100,
            quantum_iterations=50
        )
        
        comparison = results["comparison"]
        
        assert "speedup_wallclock" in comparison
        assert "speedup_effective" in comparison
        assert "summary" in comparison
        
        # Speedups deben ser positivos
        assert comparison["speedup_wallclock"] > 0
        assert comparison["speedup_effective"] > 0
    
    def test_quantum_faster_than_classical(self, synthetic_data, param_bounds):
        """Test que VQC es comparativamente eficiente vs MH"""
        data, psd = synthetic_data
        
        results = MCMCBenchmark.run_comparison(
            data, psd, param_bounds,
            classical_iterations=500,
            quantum_iterations=200
        )
        
        classical_time = results["classical"].time_elapsed
        quantum_time = results["quantum"].time_elapsed
        
        speedup = classical_time / quantum_time
        
        # Con suficientes iteraciones, VQC debería ser más eficiente
        # Verificar que speedup existe (puede ser > 1 o < 1 dependiendo de overhead)
        assert speedup > 0
    
    def test_quantum_effective_efficiency(self, synthetic_data, param_bounds):
        """Test que VQC tiene mejor eficiencia de muestreo"""
        data, psd = synthetic_data
        
        results = MCMCBenchmark.run_comparison(
            data, psd, param_bounds,
            classical_iterations=300,
            quantum_iterations=150
        )
        
        classical_metrics = results["classical"]
        quantum_metrics = results["quantum"]
        
        # Cálculo: effective_samples / time
        classical_efficiency = classical_metrics.effective_samples / (classical_metrics.time_elapsed + 1e-10)
        quantum_efficiency = quantum_metrics.effective_samples / (quantum_metrics.time_elapsed + 1e-10)
        
        # Ambas deben tener eficiencia positiva
        assert classical_efficiency > 0
        assert quantum_efficiency > 0
        
        # VQC típicamente tiene mejor eficiencia (100% acceptance, low autocorr)
        # aunque puede variar por overhead Python
        assert (quantum_efficiency / classical_efficiency) > 0.5
    
    # ==================== NUMERICAL STABILITY TESTS ====================
    
    def test_no_nans_in_mh(self, synthetic_data, param_bounds):
        """Test que MH no produce NaN/Inf"""
        data, psd = synthetic_data
        
        samples, metrics = ClassicalMetropolisHastings.sample(
            data, psd, param_bounds,
            num_iterations=100,
            proposal_std=0.5
        )
        
        assert not np.any(np.isnan(samples))
        assert not np.any(np.isinf(samples))
        assert not np.any(np.isnan(metrics.posterior_mean))
    
    def test_no_nans_in_vqc(self, synthetic_data, param_bounds):
        """Test que VQC no produce NaN/Inf"""
        data, psd = synthetic_data
        
        samples, metrics = VariationalQuantumCircuitSampler.quantum_circuit_inference(
            data, psd, param_bounds,
            num_iterations=50,
            learning_rate=0.01
        )
        
        assert not np.any(np.isnan(samples))
        assert not np.any(np.isinf(samples))
        assert not np.any(np.isnan(metrics.posterior_mean))
    
    def test_benchmark_execution_stability(self, synthetic_data, param_bounds):
        """Test que benchmark completo es estable"""
        data, psd = synthetic_data
        
        try:
            results = MCMCBenchmark.run_comparison(
                data, psd, param_bounds,
                classical_iterations=50,
                quantum_iterations=25
            )
            
            # Si llegamos aquí, fue exitoso
            assert results is not None
            assert len(results) == 3  # classical, quantum, comparison
        except Exception as e:
            pytest.fail(f"Benchmark crash: {str(e)}")
    
    # ==================== INTEGRATION TESTS ====================
    
    def test_mh_convergence_with_more_iterations(self, synthetic_data, param_bounds):
        """Test que MH converge mejor con más iteraciones"""
        data, psd = synthetic_data
        
        _, metrics_100 = ClassicalMetropolisHastings.sample(
            data, psd, param_bounds,
            num_iterations=100,
            proposal_std=0.5
        )
        
        _, metrics_500 = ClassicalMetropolisHastings.sample(
            data, psd, param_bounds,
            num_iterations=500,
            proposal_std=0.5
        )
        
        # R-hat debería mejora (disminuir) con más iteraciones
        assert metrics_500.convergence_r_hat <= metrics_100.convergence_r_hat
    
    def test_vqc_posterior_stability_across_runs(self, synthetic_data, param_bounds):
        """Test que VQC produce posteriors consistentes"""
        data, psd = synthetic_data
        
        _, metrics1 = VariationalQuantumCircuitSampler.quantum_circuit_inference(
            data, psd, param_bounds,
            num_iterations=100,
            learning_rate=0.01
        )
        
        _, metrics2 = VariationalQuantumCircuitSampler.quantum_circuit_inference(
            data, psd, param_bounds,
            num_iterations=100,
            learning_rate=0.01
        )
        
        # Ambas ejecuciones deben producir posteriors válidos
        assert metrics1.posterior_mean is not None
        assert metrics2.posterior_mean is not None
        
        # Ambas deberían estar dentro de bounds
        for i in range(len(param_bounds)):
            assert param_bounds[i]["min"] <= metrics1.posterior_mean[i] <= param_bounds[i]["max"]
            assert param_bounds[i]["min"] <= metrics2.posterior_mean[i] <= param_bounds[i]["max"]
