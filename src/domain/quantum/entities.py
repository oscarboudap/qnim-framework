"""DEPRECATED: Entidades cuánticas movidas a value_objects.py

Este archivo se mantiene por compatibilidad backwards, pero todas las
definiciones han sido reubicadas a value_objects.py con validación
y métodos adicionales.

USAR IMPORTACIONES DE:
    from src.domain.quantum.value_objects import (
        QUBOProblem, AnnealingResult, VQCTopology
    )
"""

from src.domain.quantum.value_objects import (
    QUBOProblem,
    AnnealingResult,
    VQCTopology,
)

__all__ = [
    "QUBOProblem",
    "AnnealingResult",
    "VQCTopology",
]