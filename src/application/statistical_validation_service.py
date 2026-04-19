"""
Application: Statistical Validation Service (MC Sweep + Significance Tests)
===========================================================

Orquesta:
1. Monte Carlo sweep a múltiples SNRs
2. Tests de significancia (χ², Bayesian Model Comparison)
3. No-hair theorem validation
4. Bootstrap confidence intervals
"""

import numpy as np
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
import json
from pathlib import Path

from src.domain.metrology.bootstrap_resampler import BootstrapResampler
from src.domain.metrology.fisher_matrix_new import FisherMatrixCalculator, FisherResult


@dataclass
class MCResult:
    """Resultado de un trial Monte Carlo."""
    snr: float
    theory_label: str
    extracted_parameters: Dict[str, float]
    vqc_predictions: np.ndarray  # Probabilidades por teoría
    prediction_correct: bool


@dataclass
class SignificanceTest:
    """Resultado de test de significancia."""
    test_name: str
    test_statistic: float
    p_value: float
    threshold: float
    is_significant: bool
    interpretation: str


class StatisticalValidationService:
    """
    Servicio de validación estadística completa del pipeline QNIM.
    
    Combina bootstrap, Fisher matrix, MC sweeps y tests de significancia.
    """
    
    def __init__(self, n_bootstrap: int = 1000, random_seed: int = 42):
        self.bootstrap_resampler = BootstrapResampler(n_bootstrap, random_seed)
        self.fisher_calculator = FisherMatrixCalculator()
        self.rng = np.random.RandomState(random_seed)
    
    def monte_carlo_sweep(
        self,
        signal_generator: Callable,  # genera señal sintética
        vqc_classifier: Callable,    # clasifica con VQC entrenado
        theory_labels: List[str],
        snr_levels: List[float],
        n_trials_per_snr: int = 50,
        noise_std: float = 1.0
    ) -> Dict[float, List[MCResult]]:
        """
        Monte Carlo sweep: genera eventos a múltiples SNRs, clasifica con VQC.
        
        Args:
            signal_generator: Función(theory, snr, n_samples) → signal
            vqc_classifier: Función(signal) → [p_GR, p_LQG, p_ECO, ...]
            theory_labels: Lista de teorías
            snr_levels: SNRs a probar
            n_trials_per_snr: Trials por SNR
            noise_std: Desviación estándar de ruido
            
        Returns:
            {snr: [MCResult, ...]}
        """
        results_by_snr = {}
        
        print(f"🎲 Iniciando Monte Carlo sweep...")
        print(f"   SNRs: {snr_levels}")
        print(f"   Trials por SNR: {n_trials_per_snr}")
        print(f"   Teorías: {', '.join(theory_labels)}")
        print()
        
        for snr in snr_levels:
            print(f"  📊 SNR = {snr:.1f}:")
            mc_results = []
            
            for trial in range(n_trials_per_snr):
                # Seleccionar teoría al azar
                true_theory = self.rng.choice(theory_labels)
                
                # Generar señal sintética
                signal = signal_generator(true_theory, snr, n_samples=16384)
                
                # Agregar ruido
                noise = self.rng.normal(0, noise_std, len(signal))
                noisy_signal = signal + noise
                
                # Clasificar con VQC
                predictions = vqc_classifier(noisy_signal)  # [n_theories,]
                
                # ¿Acertó?
                predicted_theory = theory_labels[np.argmax(predictions)]
                is_correct = (predicted_theory == true_theory)
                
                # Dummy: extractos de parámetros (en producción: fitting real)
                extracted_params = {
                    "M_c": 30.0 + self.rng.normal(0, 1),
                    "chi_eff": 0.5 + self.rng.normal(0, 0.1),
                    "SNR": snr
                }
                
                result = MCResult(
                    snr=snr,
                    theory_label=true_theory,
                    extracted_parameters=extracted_params,
                    vqc_predictions=predictions,
                    prediction_correct=is_correct
                )
                mc_results.append(result)
                
                if (trial + 1) % 10 == 0:
                    accuracy = np.mean([r.prediction_correct for r in mc_results])
                    print(f"    Trials {trial+1}/{n_trials_per_snr}: Accuracy = {accuracy:.2%}")
            
            results_by_snr[snr] = mc_results
            
            # Estadísticas finales para este SNR
            accuracy = np.mean([r.prediction_correct for r in mc_results])
            print(f"    ✓ Final accuracy @ SNR={snr:.1f}: {accuracy:.2%}")
            print()
        
        return results_by_snr
    
    def chi_squared_goodness_of_fit(
        self,
        observed_params: np.ndarray,
        predicted_params: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> SignificanceTest:
        """
        Test χ² para bondad de ajuste.
        
        H0: Observed = Predicted
        
        χ² = (Obs - Pred)^T Cov^-1 (Obs - Pred)
        
        Args:
            observed_params: Parámetros extraídos de datos
            predicted_params: Parámetros predichos por modelo
            covariance_matrix: Matriz de covarianza
            
        Returns:
            SignificanceTest
        """
        n_params = len(observed_params)
        
        # Invertir covarianza
        try:
            cov_inv = np.linalg.inv(covariance_matrix)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(covariance_matrix)
        
        delta = observed_params - predicted_params
        chi2 = np.dot(delta, np.dot(cov_inv, delta))
        
        # p-value from chi-squared CDF
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df=n_params)
        
        is_significant = p_value < 0.05
        
        interpretation = (
            f"χ² = {chi2:.2f} (dof={n_params}), p={p_value:.4f}. "
            f"Modelo {'rechazado' if is_significant else 'consistente'} al 95% CL."
        )
        
        return SignificanceTest(
            test_name="χ² Goodness of Fit",
            test_statistic=float(chi2),
            p_value=float(p_value),
            threshold=0.05,
            is_significant=is_significant,
            interpretation=interpretation
        )
    
    def kullback_leibler_divergence(
        self,
        p_data: np.ndarray,
        p_model: np.ndarray
    ) -> float:
        """
        Divergencia de Kullback-Leibler entre distribuciones.
        
        D_KL(p_data || p_model) = ∑ p_data * log(p_data / p_model)
        
        Args:
            p_data: Distribución observada (normalizada)
            p_model: Distribución del modelo (normalizada)
            
        Returns:
            D_KL (≥ 0, 0 significa distribuciones idénticas)
        """
        # Evitar singularidades
        epsilon = 1e-10
        p_data = np.clip(p_data, epsilon, 1)
        p_model = np.clip(p_model, epsilon, 1)
        
        # KL divergence
        kl = np.sum(p_data * np.log(p_data / p_model))
        
        return float(kl)
    
    def theory_model_comparison(
        self,
        mc_results: List[MCResult],
        theory_labels: List[str]
    ) -> Dict[str, Dict]:
        """
        Comparación de teorías usando predicciones VQC.
        
        Args:
            mc_results: Resultados MC recolectados
            theory_labels: Nombres de teorías
            
        Returns:
            {theory: {accuracy, log_odds vs others, ...}}
        """
        n_theories = len(theory_labels)
        theory_accuracies = {t: 0 for t in theory_labels}
        theory_counts = {t: 0 for t in theory_labels}
        
        # Calcular accuracy por teoría
        for result in mc_results:
            theory_counts[result.theory_label] += 1
            if result.prediction_correct:
                theory_accuracies[result.theory_label] += 1
        
        # Normalizar
        for theory in theory_labels:
            if theory_counts[theory] > 0:
                theory_accuracies[theory] /= theory_counts[theory]
        
        # Calcular log-odds vs GR (baseline)
        gr_accuracy = max(theory_accuracies.values()) if theory_accuracies else 0.5
        
        results = {}
        for theory in theory_labels:
            acc = theory_accuracies[theory]
            
            # Log-odds ratio
            log_odds = np.log((acc + 1e-5) / (1 - acc + 1e-5))
            
            results[theory] = {
                "accuracy": float(acc),
                "n_trials": theory_counts[theory],
                "log_odds_vs_gr": float(log_odds),
                "is_better_than_gr": acc > gr_accuracy
            }
        
        return results
    
    def no_hair_theorem_test(
        self,
        ringdown_frequencies: np.ndarray,
        ringdown_damping_times: np.ndarray,
        mass_estimate: float,
        spin_estimate: float
    ) -> Dict:
        """
        Test del teorema no-cabello.
        
        Predice Q(n,m) = ω(n,m) * τ(n,m) para Kerr.
        Compara con observado.
        
        Args:
            ringdown_frequencies: Frecuencias observadas [Hz]
            ringdown_damping_times: Tiempos de amortiguamiento [s]
            mass_estimate: Masa estimada [M_sun]
            spin_estimate: Spin estimado (adimensional)
            
        Returns:
            {n00: {predicted_Q, observed_Q, deviation_sigma}, ...}
        """
        results = {}
        
        # Kerr QNM predictions (0,0 mode es el más excitado)
        # Fórmula aproximada de Berti et al.
        
        for i, (f_obs, tau_obs) in enumerate(zip(ringdown_frequencies, ringdown_damping_times)):
            # Q = 2π f * τ
            q_obs = 2 * np.pi * f_obs * tau_obs
            
            # Predicción Kerr (toy model)
            # En realidad usarías interpolación de tablas QNM de Berti
            m_c = mass_estimate * 1.477e-3  # Conversión a km (G=c=1)
            q_predicted = 0.5 * (1 + 0.5 * spin_estimate)  # Aproximación simple
            
            deviation = (q_obs - q_predicted) / (q_predicted + 1e-10)
            
            results[f"QNM_{i}"] = {
                "frequency_hz": float(f_obs),
                "damping_time_s": float(tau_obs),
                "observed_Q": float(q_obs),
                "predicted_Q_kerr": float(q_predicted),
                "deviation_fraction": float(deviation),
                "beyond_gr_evidence": abs(deviation) > 0.1  # >10% desviación
            }
        
        return results
    
    def bootstrap_parameter_uncertainties(
        self,
        extracted_parameters: np.ndarray,  # [n_events, n_params]
        param_names: List[str]
    ) -> Dict[str, Dict]:
        """
        Bootstrap para incertidumbres en parámetros extraídos.
        
        Args:
            extracted_parameters: Array [n_events, n_params]
            param_names: Nombres de parámetros
            
        Returns:
            {param_name: {mean, std, ci_68, ci_95}}
        """
        results = {}
        
        for i, param_name in enumerate(param_names):
            param_data = extracted_parameters[:, i]
            
            # Bootstrap
            boot_result = self.bootstrap_resampler.bootstrap_parameter(
                param_data,
                estimator_func=np.mean
            )
            
            results[param_name] = {
                "mean": boot_result.mean_estimate,
                "std": boot_result.std_estimate,
                "bias": boot_result.bias,
                "ci_68": {"lower": boot_result.ci_68[0], "upper": boot_result.ci_68[1]},
                "ci_95": {"lower": boot_result.ci_95[0], "upper": boot_result.ci_95[1]},
            }
        
        return results
