"""
Presentation Layer: DTO Mappers
===============================

Convertir DTOs de application layer a formatos tipados para display.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

from src.application.dto import (
    ClassificationResult,
    InferenceResult,
    TrainingMetrics,
    ConfusionMatrixData,
)
from src.presentation.configuration import CLI_CONFIG
from src.presentation.exceptions import DTOConversionException


@dataclass(frozen=True)
class FormattedClassification:
    """Formato tipado para mostrar resultado de clasificación."""
    
    theory: str
    confidence: float  # 0.0-1.0
    beyond_gr_detected: bool
    capas_detectadas: int
    
    def confidence_percentage(self) -> str:
        """Retorna confianza como porcentaje formateado."""
        return f"{self.confidence * 100:.{CLI_CONFIG.PERCENTAGE_PRECISION}f}%"


@dataclass(frozen=True)
class FormattedInference:
    """Formato tipado para mostrar resultado de inferencia completa."""
    
    classification: FormattedClassification
    execution_time_seconds: float
    circuit_depth: Optional[int] = None
    method: Optional[str] = None
    classical_parameters: Optional[Dict[str, float]] = None
    
    def formatted_time(self) -> str:
        """Retorna tiempo en formato legible."""
        return f"{self.execution_time_seconds:.{CLI_CONFIG.FLOAT_PRECISION}f}s"


@dataclass(frozen=True)
class FormattedTrainingMetrics:
    """Formato tipado para mostrar métricas de entrenamiento."""
    
    loss: float
    accuracy: float
    iterations: int
    execution_time_seconds: float
    
    def formatted_loss(self) -> str:
        return f"{self.loss:.{CLI_CONFIG.FLOAT_PRECISION}f}"
    
    def formatted_accuracy(self) -> str:
        return f"{self.accuracy * 100:.{CLI_CONFIG.PERCENTAGE_PRECISION}f}%"
    
    def formatted_time(self) -> str:
        return f"{self.execution_time_seconds:.{CLI_CONFIG.FLOAT_PRECISION}f}s"


def map_classification_result(result: ClassificationResult) -> FormattedClassification:
    """
    Convierte ClassificationResult (application DTO) a FormattedClassification (presentation).
    
    Args:
        result: DTO de application layer
    
    Returns:
        Formato tipado para presentación CLI
    
    Raises:
        DTOConversionException: Si datos inválidos o incompletos
    """
    try:
        if result is None:
            raise ValueError("Result cannot be None")
        
        if result.confidence < 0 or result.confidence > 1:
            raise ValueError(f"Confidence must be 0-1, got {result.confidence}")
        
        return FormattedClassification(
            theory=result.theory,
            confidence=result.confidence,
            beyond_gr_detected=result.beyond_gr_detected,
            capas_detectadas=result.layers_detected
        )
    except Exception as e:
        raise DTOConversionException(
            f"Error convirtiendo ClassificationResult: {str(e)}"
        )


def map_inference_result(result: InferenceResult) -> FormattedInference:
    """
    Convierte InferenceResult a FormattedInference.
    
    Args:
        result: DTO de application layer
    
    Returns:
        Formato tipado para presentación
    
    Raises:
        DTOConversionException: Si datos inválidos
    """
    try:
        if result is None:
            raise ValueError("Result cannot be None")
        
        formatted_classification = map_classification_result(result.classification)
        
        return FormattedInference(
            classification=formatted_classification,
            execution_time_seconds=result.execution_time_seconds,
            circuit_depth=result.circuit_depth,
            method=result.method,
            classical_parameters=result.classical_parameters
        )
    except DTOConversionException:
        raise
    except Exception as e:
        raise DTOConversionException(
            f"Error convirtiendo InferenceResult: {str(e)}"
        )


def map_training_metrics(metrics: TrainingMetrics) -> FormattedTrainingMetrics:
    """
    Convierte TrainingMetrics a FormattedTrainingMetrics.
    
    Args:
        metrics: DTO de application layer
    
    Returns:
        Formato tipado para presentación
    
    Raises:
        DTOConversionException: Si datos inválidos
    """
    try:
        if metrics is None:
            raise ValueError("Metrics cannot be None")
        
        if metrics.loss < 0:
            raise ValueError(f"Loss cannot be negative: {metrics.loss}")
        if metrics.accuracy < 0 or metrics.accuracy > 1:
            raise ValueError(f"Accuracy must be 0-1: {metrics.accuracy}")
        if metrics.iterations < 0:
            raise ValueError(f"Iterations cannot be negative: {metrics.iterations}")
        if metrics.execution_time_seconds < 0:
            raise ValueError(f"Time cannot be negative: {metrics.execution_time_seconds}")
        
        return FormattedTrainingMetrics(
            loss=metrics.loss,
            accuracy=metrics.accuracy,
            iterations=metrics.iterations,
            execution_time_seconds=metrics.execution_time_seconds
        )
    except Exception as e:
        raise DTOConversionException(
            f"Error convirtiendo TrainingMetrics: {str(e)}"
        )


def validate_confusion_matrix(cm: ConfusionMatrixData) -> None:
    """
    Valida que ConfusionMatrixData sea válida.
    
    Args:
        cm: Datos de matriz de confusión
    
    Raises:
        DTOConversionException: Si datos inválidos
    """
    try:
        if cm is None:
            raise ValueError("ConfusionMatrixData cannot be None")
        
        if any(val < 0 for val in [cm.tp, cm.tn, cm.fp, cm.fn]):
            raise ValueError("Confusion matrix values cannot be negative")
        
        if cm.tp + cm.fp + cm.tn + cm.fn == 0:
            raise ValueError("Confusion matrix is empty (all zeros)")
    except Exception as e:
        raise DTOConversionException(
            f"Error validando ConfusionMatrixData: {str(e)}"
        )
