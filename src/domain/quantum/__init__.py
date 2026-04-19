"""
API pública del dominio cuántico (domain/quantum).

Este módulo expone la configuración cuántica, problemas de optimización
y resultados de computación cuántica necesarios para ejecutar circuitos
(VQC) y recocido cuántico en la detección de ondas gravitacionales.

Estructura:
-----------
CONFIGURACIÓN CUÁNTICA
    VQCTopology: Topología de la red neuronal variacional
    QuantumCircuitConfig: Configuración de ejecución

OPTIMIZACIÓN CUÁNTICA
    QUBOProblem: Problema de optimización binaria
    AnnealingResult: Resultado de recocido cuántico
    TemplateSignal: Plantilla para template matching

SERVICIOS DE DOMINIO (Stateless)
    TemplateMatchingQUBO: Formulador de problemas QUBO

INTERFACES (Puertos)
    IQuantumAnnealer: Contrato para recocido cuántico
    IGateBasedQuantumComputer: Contrato para compuertas cuánticas
"""

# ============================================================================
# VALUE OBJECTS: CONFIGURACIÓN
# ============================================================================

from .value_objects import (
    VQCTopology,
    QuantumCircuitConfig,
)

# ============================================================================
# VALUE OBJECTS: OPTIMIZACIÓN
# ============================================================================

from .value_objects import (
    QUBOProblem,
    AnnealingResult,
    TemplateSignal,
)

# ============================================================================
# SERVICIOS DE DOMINIO
# ============================================================================

from .qubo_formulator import TemplateMatchingQUBO

# ============================================================================
# INTERFACES (PUERTOS)
# ============================================================================

from .interfaces import (
    IQuantumAnnealer,
    IGateBasedQuantumComputer,
)

# ============================================================================
# EXPORTACIONES PÚBLICAS (Backward Compatibility)
# ============================================================================

# Legacy imports (deprecated, pero soportado)
from .entities import (  # DEPRECATED, usa value_objects.py
    QUBOProblem as _QUBOProblem_legacy,
    AnnealingResult as _AnnealingResult_legacy,
    VQCTopology as _VQCTopology_legacy,
)

__all__ = [
    # Value Objects (6 exports)
    "VQCTopology",
    "QuantumCircuitConfig",
    "QUBOProblem",
    "AnnealingResult",
    "TemplateSignal",
    
    # Domain Services (1 export)
    "TemplateMatchingQUBO",
    
    # Interfaces (2 exports)
    "IQuantumAnnealer",
    "IGateBasedQuantumComputer",
]

__version__ = "1.0.0"
__author__ = "Postdoctoral Research, Quantum Computing Group"

