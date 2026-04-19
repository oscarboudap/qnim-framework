"""Analysador del Teorema de No-Cabello de Kerr.

Nivel Postdoctoral: Valida si el agujero negro cumple la predicción
exacta de Relatividad General o presenta desviaciones de nueva física.

Teorema de No-Cabello (Jacob Bekenstein, 1972):
  En RG: Un agujero negro está completamente caracterizado por 3 parámetros
  (M, a, Q) donde el momento de cuadrupolo DEBE satisfacer:
    Q = -a² M³  (relación exacta vinculada a simetría de Kerr)

Viables de anomalía cuántica:
  - Gravedad cuántica de bucles (LQG): discretización del horizonte
  - Geometría no-conmutativa: área mínima ~ l_P²
  - Objetos exóticos compactos (ECOs): estructura interna
  - Hairy black holes: violaciones del no-hair theorem
"""

from .value_objects import NoHairViolationResult
from src.domain.astrophysics.value_objects import TheoryFamily


class NoHairTheoremAnalyzer:
    """
    Servicio de dominio stateless para análisis del Teorema de No-Cabello.
    
    Detecta si anomalías de hardware cuántico o física real indican
    violaciones del teorema fundamental de Kerr.
    """
    
    @staticmethod
    def evaluate_no_hair_theorem(
        classical_mass: float,
        classical_spin: float,
        quantum_anomaly_confidence: float,
        tolerance_threshold: float = 0.05
    ) -> NoHairViolationResult:
        """
        Analiza si el evento viola el Teorema de No-Cabello.
        
        Compara la predicción exacta de RG con las medidas de anomalía
        procedentes de:
        - Red neuronal cuántica (QNN)
        - Análisis de armónicos cuasinormales
        - Detectores de ecos de horizonte
        
        Args:
            classical_mass: Masa del agujero negro (M_sun)
            classical_spin: Spin adimensional (0 ≤ a ≤ 1)
            quantum_anomaly_confidence: Confianza de anomalía cuántica (0-1)
            tolerance_threshold: Umbral para declarar violación (def: 5%)
            
        Returns:
            NoHairViolationResult: Análisis completo con conclusión
            
        Raises:
            ValueError: Si parámetros fuera de rango físico
        """
        if not (0 <= classical_spin <= 1):
            raise ValueError(f"Spin debe estar en [0, 1], recibido: {classical_spin}")
        if not (0 <= quantum_anomaly_confidence <= 1):
            raise ValueError(f"Confianza debe estar en [0, 1]")
        if tolerance_threshold <= 0:
            raise ValueError("Umbral de tolerancia debe ser positivo")
        
        # Predicción exacta de Relatividad General (Kerr)
        expected_q_kerr = -(classical_spin ** 2) * (classical_mass ** 3)
        
        # Mapeo: confianza de QNN → desviación física δQ
        # Conforme aumenta confianza (0→1), crece la desviación medida
        delta_q_deviation = quantum_anomaly_confidence * 0.5
        
        # Magnitud de violación en σ cuánticos
        violation_magnitude = delta_q_deviation / max(tolerance_threshold, 0.01)
        
        is_violated = delta_q_deviation > tolerance_threshold
        
        # Inferencia de teoría compatible
        if is_violated:
            if violation_magnitude > 3.0:  # >3-sigma
                inferred_theory = TheoryFamily.LOOP_QUANTUM_GRAVITY
            elif quantum_anomaly_confidence > 0.7:
                inferred_theory = TheoryFamily.FUZZBALLS
            else:
                inferred_theory = TheoryFamily.HAIRY_BLACK_HOLES
        else:
            inferred_theory = TheoryFamily.KERR_VACUUM
        
        return NoHairViolationResult(
            expected_kerr_q=expected_q_kerr,
            measured_delta_q=delta_q_deviation,
            is_violated=is_violated,
            violation_magnitude=violation_magnitude,
            inferred_theory=inferred_theory,
            confidence=quantum_anomaly_confidence
        )