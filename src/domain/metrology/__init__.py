"""
API pública del dominio de metrología.

Este módulo expone todos los servicios y value objects de metrología
para ondas gravitacionales según Domain-Driven Design.

Estructura:
-----------
INSTRUMENTACIÓN (PSD y ruido)
    PowerSpectralDensity: Ruido del detector

SERVICIOS COSMOLÓGICOS (Capa 4)
    HubbleCosmologyCalculator: Inferencia de H₀

SERVICIOS DE HORIZONTE (Capa 6)
    NoHairTheoremAnalyzer: Test del teorema de no-cabello

SERVICIOS DE FÍSICA CUÁNTICA (Capa 7)
    QuantumGravitySignificanceCalculator: Significancia de nueva física

METROLOGÍA FUNDAMENTAL
    FisherMatrixCalculator: Cotas de Cramér-Rao
    
VALUE OBJECTS
    HubbleConstant, NoHairViolationResult, QuantumGravitySignificance,
    PowerSpectralDensity, FisherMatrix
"""

# ============================================================================
# VALUE OBJECTS
# ============================================================================

from .value_objects import (
    # Capa 4
    HubbleConstant,
    # Capa 6
    NoHairViolationResult,
    # Capa 7
    QuantumGravitySignificance,
    # Instrumentación
    PowerSpectralDensity,
    FisherMatrix,
)

# ============================================================================
# SERVICIOS DE DOMINIO: COSMOLOGÍA
# ============================================================================

from .hubble_metrology import HubbleCosmologyCalculator

# ============================================================================
# SERVICIOS DE DOMINIO: HORIZONTE CUÁNTICO
# ============================================================================

from .multipole_validator import NoHairTheoremAnalyzer

# ============================================================================
# SERVICIOS DE DOMINIO: FÍSICA CUÁNTICA
# ============================================================================

from .planck_error_bounds import QuantumGravitySignificanceCalculator

# ============================================================================
# SERVICIOS DE DOMINIO: METROLOGÍA FUNDAMENTAL
# ============================================================================

from .fisher_matrix_calculator import FisherMatrixCalculator


# ============================================================================
# EXPORTACIONES PÚBLICAS
# ============================================================================

__all__ = [
    # Value Objects (5 exports)
    "HubbleConstant",
    "NoHairViolationResult",
    "QuantumGravitySignificance",
    "PowerSpectralDensity",
    "FisherMatrix",
    
    # Domain Services (4 exports)
    "HubbleCosmologyCalculator",
    "NoHairTheoremAnalyzer",
    "QuantumGravitySignificanceCalculator",
    "FisherMatrixCalculator",
]

__version__ = "1.0.0"
__author__ = "Postdoctoral Research, GW Metrology Group"

