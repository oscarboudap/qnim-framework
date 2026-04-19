"""
Domain: Fisher Matrix Calculator for GW Parameter Estimation
===========================================================

Calcula matriz de Fisher para:
- Estimar cotas de Cramér-Rao en parámetros
- Evaluar poder discriminatorio de teorías
- Estimar precision bounds (mejor caso Bayesiano)

Basado en:
- Vallisneri (2008) arXiv:gr-qc/0610110
- Moore, Cole, Berry (2014) Classical and Quantum Gravity
"""

import numpy as np
from typing import Dict, Tuple, Callable, List
from dataclasses import dataclass


@dataclass
class FisherResult:
    """Resultado de análisis de matriz de Fisher."""
    fisher_matrix: np.ndarray  # [n_params, n_params]
    param_names: List[str]
    parameter_errors: np.ndarray  # √(F^-1 diagonal)
    parameter_correlations: np.ndarray  # Correlación normalizada
    degeneracies: List[Tuple[str, str, float]]  # (param1, param2, correlation)
    cumulative_snr: float


class FisherMatrixCalculator:
    """
    Calcula matriz de Fisher para análisis de ondas gravitacionales.
    
    Usa aproximación lineal:
    F_ij = ∫ (∂h+/∂θ_i * ∂h+/∂θ_j + ∂h×/∂θ_i * ∂h×/∂θ_j) / S_n(f) df
    """
    
    def __init__(self, 
                 sample_rate: float = 16384,  # sampleo [Hz]
                 duration: float = 1.0,       # duración [s]
                 noise_psd_func: Callable = None):
        """
        Args:
            sample_rate: Frecuencia de sampleo [Hz]
            duration: Duración de la observación [s]
            noise_psd_func: PSD del ruido S_n(f)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        
        # Por defecto: LIGO noise curve
        if noise_psd_func is None:
            self.noise_psd = self._ligo_noise_psd
        else:
            self.noise_psd = noise_psd_func
    
    @staticmethod
    def _ligo_noise_psd(f: np.ndarray) -> np.ndarray:
        """
        LIGO noise power spectral density (O2, aLIGO).
        
        Aproximación analítica de Vallisneri et al.
        """
        # Parámetros de ajuste de O2
        f_0 = 150.0  # Reference frequency
        S_0 = 1e-49  # Reference PSD (Hz^-1)
        
        # Seismic wall (low f)
        seismic = (f / f_0) ** (-4.14)
        
        # Thermal noise (mid f)
        thermal = 0.4 * (f_0 / f) ** 0.5
        
        # Radiation pressure (high f)
        rad_pressure = (f / f_0) ** 2.5
        
        s_n_f = S_0 * (seismic + thermal + rad_pressure)
        
        # Piso mínimo de ruido
        s_n_f = np.maximum(s_n_f, 1e-50)
        
        return s_n_f
    
    def compute_fisher_matrix(
        self,
        signal: np.ndarray,        # [n_samples, n_detectors]
        strain_sampler_func: Callable,  # Función que genera strain
        parameter_dict: Dict,      # Parámetros nominales
        param_names: List[str],    # Nombres de parámetros a variar
        delta_frac: float = 1e-3   # Fracción para derivada numérica
    ) -> FisherResult:
        """
        Calcula matriz de Fisher por diferencias finitas.
        
        F_ij ≈ ∂ln(L)/∂θ_i * ∂ln(L)/∂θ_j
        
        Donde ln(L) es la log-likelihood del VQC.
        
        Args:
            signal: Datos observados [N, n_det] (puede ser predicción VQC)
            strain_sampler_func: Función(params, n_samples) → [n_samples, n_det]
            parameter_dict: Diccionario de parámetros nominales
            param_names: Parámetros sobre los que variar
            delta_frac: Fracción para derivada numérica (típicamente 1e-3)
            
        Returns:
            FisherResult con matriz y análisis
        """
        n_params = len(param_names)
        n_samples = signal.shape[0]
        
        # Derivadas numéricas (finite differences)
        derivatives = np.zeros((n_params, n_samples))
        
        print(f"  🔢 Calculando derivadas numéricas ({n_params} parámetros)...")
        
        for i, param_name in enumerate(param_names):
            # h(θ + δ)
            params_plus = parameter_dict.copy()
            delta = delta_frac * abs(params_plus[param_name]) if params_plus[param_name] != 0 else delta_frac
            params_plus[param_name] += delta
            strain_plus = strain_sampler_func(params_plus, n_samples)
            
            # h(θ - δ)
            params_minus = parameter_dict.copy()
            params_minus[param_name] -= delta
            strain_minus = strain_sampler_func(params_minus, n_samples)
            
            # Derivada por diferencia central
            dh_dtheta = (strain_plus - strain_minus) / (2 * delta)
            
            # Podría ser [N, n_det], promediamos sobre detectores
            if dh_dtheta.ndim > 1:
                dh_dtheta = np.mean(dh_dtheta, axis=1)
            
            derivatives[i] = dh_dtheta
            
            print(f"    ∂h/∂{param_name}: ✓")
        
        # Calcular matriz de Fisher
        # F_ij = <∂h/∂θ_i * ∂h/∂θ_j> / σ_noise²
        
        print(f"  🔢 Construyendo matriz de Fisher {n_params}x{n_params}...")
        
        # Ruido RMS de los datos
        signal_mean = np.mean(signal)
        noise_array = signal - signal_mean
        sigma_noise_sq = np.mean(noise_array ** 2)
        
        fisher_matrix = np.zeros((n_params, n_params))
        
        for i in range(n_params):
            for j in range(n_params):
                # Producto interno normalizado por ruido
                fisher_matrix[i, j] = np.dot(derivatives[i], derivatives[j]) / sigma_noise_sq
        
        # Análisis de la matriz de Fisher
        print(f"  📊 Analizando matriz de Fisher...")
        
        # Invertir para obtener covariance
        try:
            fisher_inv = np.linalg.inv(fisher_matrix)
            parameter_errors = np.sqrt(np.diag(fisher_inv))
        except np.linalg.LinAlgError:
            print("    ⚠️  Matriz de Fisher singular, usando pseudo-inversa")
            fisher_inv = np.linalg.pinv(fisher_matrix)
            parameter_errors = np.sqrt(np.abs(np.diag(fisher_inv)))
        
        # Correlaciones
        normed_cov = np.zeros_like(fisher_inv)
        for i in range(n_params):
            for j in range(n_params):
                if parameter_errors[i] > 0 and parameter_errors[j] > 0:
                    normed_cov[i, j] = fisher_inv[i, j] / (parameter_errors[i] * parameter_errors[j])
        
        # Identificar degeneracies (correlaciones altas)
        degeneracies = []
        for i in range(n_params):
            for j in range(i+1, n_params):
                corr = abs(normed_cov[i, j])
                if corr > 0.9:  # Muy correlacionado
                    degeneracies.append((param_names[i], param_names[j], float(corr))
        
        # Cumulative SNR^2 = Trace(F)
        cumulative_snr = float(np.sqrt(np.trace(fisher_matrix)))\n        \n        # Imprimir resumen\n        print(f\"\\n  📈 Fisher Matrix Summary:\")\n        print(f\"     Cumulative SNR: {cumulative_snr:.2f}\")\n        print(f\"\\n     Parameter Errors (1σ):\")\n        for param_name, error in zip(param_names, parameter_errors):\n            print(f\"       σ({param_name:<12}): {error:.6e}\")\n        \n        if degeneracies:\n            print(f\"\\n     ⚠️  Degeneracies detected (corr > 0.9):\")\n            for p1, p2, corr in degeneracies:\n                print(f\"       {p1} ↔ {p2}: ρ = {corr:.3f}\")\n        \n        return FisherResult(\n            fisher_matrix=fisher_matrix,\n            param_names=param_names,\n            parameter_errors=parameter_errors,\n            parameter_correlations=normed_cov,\n            degeneracies=degeneracies,\n            cumulative_snr=cumulative_snr\n        )\n    \n    def cramér_rao_bounds(\n        self,\n        fisher_result: FisherResult\n    ) -> Dict[str, Dict]:\n        \"\"\"\n        Calcula las cotas de Cramér-Rao (best-case lower bounds).\n        \n        Teorema: Para estimador insesgado θ̂,\n        Var(θ̂_i) >= [F^-1]_ii\n        \n        Args:\n            fisher_result: Resultado de compute_fisher_matrix\n            \n        Returns:\n            {param_name: {cr_bound, relative_precision}}\n        \"\"\"\n        fisher_inv = np.linalg.pinv(fisher_result.fisher_matrix)\n        cr_bounds = np.sqrt(np.diag(fisher_inv))\n        \n        results = {}\n        for i, param_name in enumerate(fisher_result.param_names):\n            results[param_name] = {\n                \"cramér_rao_bound\": float(cr_bounds[i]),\n                \"relative_precision\": float(cr_bounds[i])  # Como % si queremos\n            }\n        \n        return results\n    \n    def theory_discrimination_power(\n        self,\n        fisher_matrix: np.ndarray,\n        theory_diff_vector: np.ndarray\n    ) -> float:\n        \"\"\"\n        Evalúa poder para discriminar dos teorías.\n        \n        Usa métrica: d = sqrt(Δθ^T F Δθ)\n        \n        Si d > 1 → discriminación posible a nivel 1σ\n        Si d > 3 → discriminación robusta a nivel 3σ\n        \n        Args:\n            fisher_matrix: Matriz de Fisher compuesta\n            theory_diff_vector: Diferencia de parámetros entre teorías\n            \n        Returns:\n            Distance metric d (adimensional)\n        \"\"\"\n        if fisher_matrix.shape[0] != len(theory_diff_vector):\n            raise ValueError(\"Dimensiones incompatibles\")\n        \n        # d = sqrt(ΔΘ^T F ΔΘ)\n        d_squared = np.dot(theory_diff_vector, np.dot(fisher_matrix, theory_diff_vector))\n        d = np.sqrt(max(d_squared, 0))  # Evitar sqrt de negativo\n        \n        return float(d)\n