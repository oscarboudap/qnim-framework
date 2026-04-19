"""Calculadora de Matriz de Fisher (metrología de ondas gravitacionales).

Nivel Postdoctoral: Calcula la cota inferior de Cramér-Rao (CRLB) para
los parámetros de nuevas teorías de gravedad.

El producto escalar ponderado por ruido en el espacio de Hilbert es:
    ⟨h₁|h₂⟩ = 4 Re ∫ h₁*(f) h₂(f) / S_n(f) df

La matriz de Fisher es:
    Γ_ij = ⟨∂_i h | ∂_j h⟩

Soteriología:
- Simétrica y semidefinida positiva
- Inversa = covarianza mínima (Cramér-Rao bound)
- Determinante proporcional a volumen de incertidumbre conjunta
"""

import numpy as np
from .value_objects import PowerSpectralDensity, FisherMatrix


class FisherMatrixCalculator:
    """
    Servicio de dominio stateless para cálculos metrológicos.
    
    Todos los métodos son estáticos (sin estado de instancia).
    Cada cálculo es puro: mismas entradas → misma salida.
    """
    
    @staticmethod
    def compute_inner_product(h1_f: np.ndarray, h2_f: np.ndarray,
                            psd: PowerSpectralDensity) -> float:
        """
        Producto escalar en el espacio de Hilbert ponderado por ruido.
        
        ⟨h₁|h₂⟩_noise = 4 Re ∫ h₁*(f) h₂(f) / S_n(f) df
        
        Args:
            h1_f: Strain en frecuencia (array complejo)
            h2_f: Strain en frecuencia (array complejo)
            psd: Power Spectral Density del detector
            
        Returns:
            Producto escalar (número real)
        """
        if len(h1_f) != len(h2_f):
            raise ValueError("h1_f y h2_f deben tener la misma longitud")
        
        integrand = (np.conj(h1_f) * h2_f) / psd.psd_values
        return 4 * np.real(np.sum(integrand)) * psd.delta_f
    
    @staticmethod
    def calculate_snr(h_f: np.ndarray, psd: PowerSpectralDensity) -> float:
        """
        Signal-to-Noise Ratio.
        
        SNR = √⟨h|h⟩
        
        Args:
            h_f: Strain en frecuencia
            psd: Power Spectral Density
            
        Returns:
            SNR (sin dimensiones, típicamente 8-250 para LIGO)
        """
        inner_prod = FisherMatrixCalculator.compute_inner_product(h_f, h_f, psd)
        return np.sqrt(max(inner_prod, 0.0))  # Guard against numerical errors
    
    @staticmethod
    def build_fisher_matrix(h_f: np.ndarray, derivatives: list,
                           psd: PowerSpectralDensity,
                           parameter_names: tuple) -> FisherMatrix:
        """
        Construye la matriz de información de Fisher.
        
        Γ_ij = ⟨∂_i h | ∂_j h⟩ con producto escalar ponderado por ruido.
        
        Args:
            h_f: Strain base en frecuencia
            derivatives: Lista de derivadas [∂h/∂θ₁, ∂h/∂θ₂, ...] en frecuencia
            psd: Power Spectral Density del detector
            parameter_names: Nombres de parámetros (ej: ('M_c', 'eta', 'a', ...))
            
        Returns:
            FisherMatrix value object (inmutable, validado)
            
        Raises:
            ValueError: Si la matriz no es semidefinida positiva
        """
        n_params = len(derivatives)
        if n_params != len(parameter_names):
            raise ValueError("Número de derivadas != número de nombres de parámetros")
        
        matrix = np.zeros((n_params, n_params), dtype=float)
        
        for i in range(n_params):
            for j in range(i, n_params):
                val = FisherMatrixCalculator.compute_inner_product(
                    derivatives[i], derivatives[j], psd
                )
                matrix[i, j] = val
                matrix[j, i] = val
        
        # Verificar semidefinitude
        eigvals = np.linalg.eigvalsh(matrix)
        if np.any(eigvals < -1e-10):  # Tolerancia numérica
            raise ValueError("Matriz de Fisher no es semidefinida positiva")
        
        snr = FisherMatrixCalculator.calculate_snr(h_f, psd)
        
        return FisherMatrix(
            matrix=matrix,
            parameter_names=parameter_names,
            snr=snr
        )