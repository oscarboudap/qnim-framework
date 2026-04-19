"""Calculadora de Significancia de Gravedad Cuántica.

Nivel Postdoctoral: Evalúa si desviaciones medidas de Relatividad General
alcanzan significancia de descubrimiento (5-sigma).

Teoría de Errores en Ondas Gravitacionales:
  σ_evento = √[(σ_medida)² + (σ_modelo)² + (σ_sistema)²]
  
Umbral de Descubrimiento (Partícula Physics):
  - 2σ: Indicador (1.4% falsa alarma)
  - 3σ: Observación (0.1% falsa alarma)
  - 5σ: Descubrimiento (0.00003% falsa alarma)

Escala de Planck & Nueva Física:
  - l_P = √(ℏG/c³) ≈ 1.616e-35 m
  - t_P ≈ 5.4e-44 s
  - Efectos visibles en metrología GW si desviación >> ruido
"""

import scipy.stats as stats
from .value_objects import QuantumGravitySignificance


class QuantumGravitySignificanceCalculator:
    """
    Servicio de dominio stateless para significancia estadística de nueva física.
    
    Detecta anomalías que indican gravedad cuántica o BSM (Beyond Standard Model).
    """
    
    # Umbral de descubrimiento universal en física de partículas
    DISCOVERY_THRESHOLD_SIGMA = 5.0
    OBSERVATION_THRESHOLD_SIGMA = 3.0
    INDICATION_THRESHOLD_SIGMA = 2.0
    
    @staticmethod
    def calculate_discovery_significance(
        background_noise_level: float,
        quantum_signal_strength: float,
        discovery_threshold_sigma: float = None
    ) -> QuantumGravitySignificance:
        """
        Calcula significancia estadística de desviación de RG.
        
        Implementa Z-score gaussian: σ = (signal - background) / σ_total
        
        Args:
            background_noise_level: Ruido de detector + errores sistemáticos
            quantum_signal_strength: Amplitud de anomalía medida
            discovery_threshold_sigma: Umbral personalizado (def: 5.0)
            
        Returns:
            QuantumGravitySignificance: Resultado con p-value e interpretación
            
        Raises:
            ValueError: Si inputs unphysical
        """
        if background_noise_level <= 0:
            raise ValueError("Ruido de fondo debe ser positivo")
        if quantum_signal_strength < 0:
            raise ValueError("Señal no puede ser negativa")
        
        if discovery_threshold_sigma is None:
            discovery_threshold_sigma = QuantumGravitySignificanceCalculator.DISCOVERY_THRESHOLD_SIGMA
        
        # Estadística Gaussiana (Z-score)
        sigma_value = quantum_signal_strength / background_noise_level
        
        # p-value bilateral (2-sided test)
        # Probabilidad de observar σ o más extremo por puro azar
        p_value = 2 * stats.norm.sf(sigma_value)
        
        is_discovery = sigma_value >= discovery_threshold_sigma
        
        # Interpretación física
        if sigma_value >= discovery_threshold_sigma:
            conclusion = "🎉 Nueva Física Confirmada (5-Sigma Descubrimiento)"
        elif sigma_value >= QuantumGravitySignificanceCalculator.OBSERVATION_THRESHOLD_SIGMA:
            conclusion = "📊 Observación Estadística Válida (3-Sigma)"
        elif sigma_value >= QuantumGravitySignificanceCalculator.INDICATION_THRESHOLD_SIGMA:
            conclusion = "💭 Indicador de Anomalía (2-Sigma)"
        else:
            conclusion = "❌ Fluctuación Estadística (< 2-Sigma)"
        
        return QuantumGravitySignificance(
            sigma_value=round(sigma_value, 2),
            p_value=float(p_value),
            is_discovery=is_discovery,
            conclusion=conclusion,
            discovery_threshold_sigma=discovery_threshold_sigma
        )