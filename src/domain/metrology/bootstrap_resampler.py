"""
Domain: Bootstrap Resampling for Confidence Intervals
======================================================

Implementa bootstrap estatístico riguroso para:
- Estimar intervalos de confianza (68%, 95%, 99.7%)
- Calcular sesgo y varianza
- Evaluar robustez a ruido

Por Rodrigo Gil-Merino (adapted for QNIM)
"""

import numpy as np
from typing import Tuple, Dict, List, Callable
from dataclasses import dataclass


@dataclass
class BootstrapResult:
    """Resultado de bootstrap resampling."""
    mean_estimate: float
    std_estimate: float
    ci_68: Tuple[float, float]  # ±1σ
    ci_95: Tuple[float, float]  # ±2σ
    ci_99_7: Tuple[float, float]  # ±3σ
    bias: float
    all_estimates: np.ndarray


class BootstrapResampler:
    """
    Bootstrap resampling para estimar incertidumbres en parámetros
    extraídos del análisis de ondas gravitacionales.
    """
    
    def __init__(self, n_bootstrap: int = 1000, random_seed: int = 42):
        """
        Args:
            n_bootstrap: Número de remuestreos (>= 1000 para rigor)
            random_seed: Semilla para reproducibilidad
        """
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(random_seed)
        
    def _resample_with_replacement(self, data: np.ndarray) -> np.ndarray:
        """
        Resamplea WITH replacement (bootstrap estándar).
        
        Args:
            data: Array [N, d] o [N]
            
        Returns:
            Array remuestreado de mismo shape
        """
        n_samples = data.shape[0]
        indices = self.rng.choice(n_samples, size=n_samples, replace=True)
        return data[indices]
    
    def bootstrap_parameter(
        self,
        data: np.ndarray,
        estimator_func: Callable[[np.ndarray], float],
        ci_levels: List[float] = [0.68, 0.95, 0.997]
    ) -> BootstrapResult:
        """
        Calcula bootstrap de un parámetro escalar.
        
        Args:
            data: Array de observaciones [N,]
            estimator_func: Función que computa parámetro (ej: np.mean, np.median)
            ci_levels: Niveles de confianza a calcular
            
        Returns:
            BootstrapResult con estadísticas completas
        """
        bootstrap_estimates = []
        
        # Generar B remuestreos
        for _ in range(self.n_bootstrap):
            resampled = self._resample_with_replacement(data)
            estimate = estimator_func(resampled)
            bootstrap_estimates.append(estimate)
        
        bootstrap_estimates = np.array(bootstrap_estimates)
        
        # Statisticas
        mean_est = np.mean(bootstrap_estimates)
        std_est = np.std(bootstrap_estimates, ddof=1)
        true_value = estimator_func(data)
        bias = mean_est - true_value
        
        # Intervalos de confianza (percentile method)
        cis = {}
        for ci_level in ci_levels:
            lower_percentile = (1 - ci_level) / 2 * 100
            upper_percentile = (1 + ci_level) / 2 * 100
            ci_lower = np.percentile(bootstrap_estimates, lower_percentile)
            ci_upper = np.percentile(bootstrap_estimates, upper_percentile)
            cis[ci_level] = (ci_lower, ci_upper)
        
        return BootstrapResult(
            mean_estimate=float(mean_est),
            std_estimate=float(std_est),
            ci_68=cis[0.68],
            ci_95=cis[0.95],
            ci_99_7=cis[0.997],
            bias=float(bias),
            all_estimates=bootstrap_estimates
        )
    
    def bootstrap_multiparameter(
        self,
        data: np.ndarray,
        estimator_func: Callable[[np.ndarray], np.ndarray],
        param_names: List[str],
        ci_level: float = 0.68
    ) -> Dict[str, Dict]:
        """
        Bootstrap para múltiples parámetros simultáneamente.
        
        Args:
            data: Array [N, d]
            estimator_func: Función que retorna [n_params]
            param_names: Nombres de parámetros
            ci_level: Nivel de confianza
            
        Returns:
            Dict {param_name: BootstrapResult}
        """
        bootstrap_estimates_list = []
        
        # Generar B remuestreos
        for _ in range(self.n_bootstrap):
            resampled = self._resample_with_replacement(data)
            estimates = estimator_func(resampled)
            bootstrap_estimates_list.append(estimates)
        
        bootstrap_matrix = np.array(bootstrap_estimates_list)  # [B, n_params]
        
        results = {}
        for i, param_name in enumerate(param_names):
            param_estimates = bootstrap_matrix[:, i]
            
            mean_est = np.mean(param_estimates)
            std_est = np.std(param_estimates, ddof=1)
            true_value = estimator_func(data)[i]
            bias = mean_est - true_value
            
            lower_percentile = (1 - ci_level) / 2 * 100
            upper_percentile = (1 + ci_level) / 2 * 100
            ci_lower = np.percentile(param_estimates, lower_percentile)
            ci_upper = np.percentile(param_estimates, upper_percentile)
            
            results[param_name] = {
                "mean": float(mean_est),
                "std": float(std_est),
                "bias": float(bias),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "all_estimates": param_estimates
            }
        
        return results
    
    def bootstrap_snr_robustness(
        self,
        signal: np.ndarray,
        noise_std: float,
        snr_levels: List[float],
        estimator_func: Callable[[np.ndarray], float],
        n_trials_per_snr: int = 50
    ) -> Dict[float, Dict]:
        """
        Evalúa robustez del estimador a diferentes SNR.
        
        Args:
            signal: Señal clean [N,]
            noise_std: Desviación estándar del ruido
            snr_levels: Lista de SNRs a probar (ej: [5, 10, 15, 20])
            estimator_func: Función que estima parámetro
            n_trials_per_snr: Trials por SNR
            
        Returns:
            {snr: {mean, std, ci_68, bias}}
        """
        results = {}
        
        for snr in snr_levels:
            print(f"  📊 Testing SNR = {snr:.1f}...", end=" ", flush=True)
            
            # Calcular amplitud de ruido para este SNR
            signal_power = np.mean(signal ** 2)
            noise_power = signal_power / (snr ** 2)
            noise_std_adjusted = np.sqrt(noise_power)
            
            # Multiple trials
            estimates = []
            for _ in range(n_trials_per_snr):
                # Generar realización de ruido
                noise = self.rng.normal(0, noise_std_adjusted, len(signal))
                noisy_signal = signal + noise
                
                # Estimar parámetro
                est = estimator_func(noisy_signal)
                estimates.append(est)
            
            estimates = np.array(estimates)
            mean_est = np.mean(estimates)
            std_est = np.std(estimates, ddof=1)
            true_value = estimator_func(signal)
            bias = mean_est - true_value
            
            ci_lower = np.percentile(estimates, 16)
            ci_upper = np.percentile(estimates, 84)
            
            results[snr] = {
                "mean": float(mean_est),
                "std": float(std_est),
                "bias": float(bias),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "all_estimates": estimates
            }
            
            print(f"✓ (std={std_est:.4f})")
        
        return results
    
    def jackknife_estimate(
        self,
        data: np.ndarray,
        estimator_func: Callable[[np.ndarray], float]
    ) -> Dict:
        """
        Jacknife resampling (complementario a bootstrap).
        Deja out each data point one at a time.
        
        Args:
            data: Array [N,]
            estimator_func: Función estimadora
            
        Returns:
            {mean, std, ci_68, bias}
        """
        n = len(data)
        jackknife_estimates = []
        
        for i in range(n):
            # Leave-one-out
            data_without_i = np.delete(data, i)
            estimate = estimator_func(data_without_i)
            jackknife_estimates.append(estimate)
        
        jackknife_estimates = np.array(jackknife_estimates)
        
        mean_est = np.mean(jackknife_estimates)
        std_est = np.std(jackknife_estimates, ddof=1)
        true_value = estimator_func(data)
        bias = mean_est - true_value
        
        # Using jackknife standard error
        # SE_jack = sqrt((n-1)/n * sum((θ_i - mean)^2))
        se_jack = np.sqrt((n - 1) / n * np.sum((jackknife_estimates - mean_est) ** 2))
        
        ci_lower = mean_est - 1.96 * se_jack
        ci_upper = mean_est + 1.96 * se_jack
        
        return {
            "mean": float(mean_est),
            "std": float(std_est),
            "std_error": float(se_jack),
            "bias": float(bias),
            "ci_95_lower": float(ci_lower),
            "ci_95_upper": float(ci_upper),
            "all_estimates": jackknife_estimates
        }


def bootstrap_parameter_covariance(
    parameters: np.ndarray,  # [n_samples, n_params]
    n_bootstrap: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula matriz de covarianza bootstrap de parámetros.
    
    Args:
        parameters: Array [n_samples, n_params]
        n_bootstrap: Número de remuestreos
        
    Returns:
        (mean_params, covariance_matrix)
    """
    rng = np.random.RandomState(42)
    n_samples, n_params = parameters.shape
    
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = parameters[indices]
        bootstrap_means.append(np.mean(bootstrap_sample, axis=0))
    
    bootstrap_means = np.array(bootstrap_means)  # [B, n_params]
    
    # Covariance matrix from bootstrap
    cov_matrix = np.cov(bootstrap_means.T)
    mean_params = np.mean(bootstrap_means, axis=0)
    
    return mean_params, cov_matrix
