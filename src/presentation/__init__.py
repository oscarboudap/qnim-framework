"""
Presentation Layer
==================

Capa de Presentación - Formatea resultados para el usuario final.
Usa DTOs de application layer (NO domain entities).
Encapsula visualización (matplotlib) y CLI.

Arquitectura:
    - CLIPresenter: Formato CLI para resultados
    - TrainingVisualizationPresenter: Gráficos matplotlib/seaborn
    - Excepciones: Framework de errores específicos de presentation
    - Configuration: Constantes centralizadas
    - DTOMappers: Conversión application DTOs → formatos display

Principios:
    ✅ Hexagonal Boundary: Presentation solo depende de Application layer
    ✅ Dependency Inversion: Inyección de dependencias
    ✅ Type Safety: 100% type hints
    ✅ Error Handling: Custom exceptions
    ✅ No Business Logic: Solo formatea y muestra
"""

# ============================================================================
# EXCEPTION FRAMEWORK
# ============================================================================

from src.presentation.exceptions import (
    PresentationException,
    CLIException,
    VisualizationException,
    DTOConversionException,
    FormattingException,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

from src.presentation.configuration import (
    CLIConfig,
    VisualizationConfig,
    CLI_CONFIG,
    VIZ_CONFIG,
)

# ============================================================================
# PRESENTERS
# ============================================================================

from src.presentation.cli_presenter import CLIPresenter

from src.presentation.visualize_results import (
    TrainingVisualizationPresenter,
    plot_training_results,        # Legacy
    plot_corner_results,          # Legacy
)

# ============================================================================
# DTO MAPPERS
# ============================================================================

from src.presentation.dto_mappers import (
    FormattedClassification,
    FormattedInference,
    FormattedTrainingMetrics,
    map_classification_result,
    map_inference_result,
    map_training_metrics,
    validate_confusion_matrix,
)

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Exceptions (7)
    "PresentationException",
    "CLIException",
    "VisualizationException",
    "DTOConversionException",
    "FormattingException",
    
    # Configuration (4)
    "CLIConfig",
    "VisualizationConfig",
    "CLI_CONFIG",
    "VIZ_CONFIG",
    
    # Presenters (2)
    "CLIPresenter",
    "TrainingVisualizationPresenter",
    
    # DTO Mappers (7)
    "FormattedClassification",
    "FormattedInference",
    "FormattedTrainingMetrics",
    "map_classification_result",
    "map_inference_result",
    "map_training_metrics",
    "validate_confusion_matrix",
    
    # Legacy (2)
    "plot_training_results",
    "plot_corner_results",
]
