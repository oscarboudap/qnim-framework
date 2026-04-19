"""
Presentation Layer: CLI Presenter
=================================

Formatea y presenta resultados en terminal.
Usa DTOs de application layer (NOT domain entities).
"""

from src.application.dto import ClassificationResult, InferenceResult
from src.presentation.configuration import CLI_CONFIG
from src.presentation.dto_mappers import (
    map_classification_result,
    map_inference_result,
    FormattedClassification,
    FormattedInference,
)
from src.presentation.exceptions import CLIException


class CLIPresenter:
    """
    Presentador CLI para mostrar resultados al usuario.
    
    Usa DTOs tipados de application layer.
    Todas las salidas son formateadas y seguros de tipo.
    """
    
    @staticmethod
    def show_welcome() -> None:
        """
        Muestra pantalla de bienvenida.
        
        Raises:
            CLIException: Si hay error de encoding o output
        """
        try:
            print(CLI_CONFIG.DIVIDER_CHAR * CLI_CONFIG.HEADER_WIDTH)
            print(f"🚀 {CLI_CONFIG.PROJECT_NAME}")
            print(f"      {CLI_CONFIG.PROJECT_SUBTITLE}")
            print(CLI_CONFIG.DIVIDER_CHAR * CLI_CONFIG.HEADER_WIDTH)
        except Exception as e:
            raise CLIException(f"Error mostrando bienvenida: {str(e)}")
    
    @staticmethod
    def show_classification_result(result: ClassificationResult) -> None:
        """
        Muestra resultado de clasificación de teoría.
        
        Args:
            result: DTO de application layer con resultado clasificación
        
        Raises:
            CLIException: Si datos inválidos o error de output
        """
        try:
            formatted = map_classification_result(result)
            
            print("\n" + CLI_CONFIG.SECTION_DIVIDER_CHAR * CLI_CONFIG.SECTION_DIVIDER_WIDTH)
            print("📊 RESULTADO DE CLASIFICACIÓN")
            print(CLI_CONFIG.SECTION_DIVIDER_CHAR * CLI_CONFIG.SECTION_DIVIDER_WIDTH)
            print(f"Teoría Predicha: {formatted.theory}")
            print(f"Confianza: {formatted.confidence_percentage()}")
            print(f"Beyond-GR Detectado: {'✅ Sí' if formatted.beyond_gr_detected else '❌ No'}")
            print(f"Capas Detectadas: {formatted.capas_detectadas}")
            print(CLI_CONFIG.SECTION_DIVIDER_CHAR * CLI_CONFIG.SECTION_DIVIDER_WIDTH + "\n")
        
        except Exception as e:
            raise CLIException(f"Error mostrando resultado clasificación: {str(e)}")
    
    @staticmethod
    def show_inference_result(result: InferenceResult) -> None:
        """
        Muestra resultado completo de inferencia.
        
        Args:
            result: DTO de application layer con resultado completo
        
        Raises:
            CLIException: Si datos inválidos o error de output
        """
        try:
            formatted = map_inference_result(result)
            
            print("\n" + CLI_CONFIG.SECTION_DIVIDER_CHAR * CLI_CONFIG.SECTION_DIVIDER_WIDTH)
            print("🎯 RESULTADO DE INFERENCIA COMPLETA")
            print(CLI_CONFIG.SECTION_DIVIDER_CHAR * CLI_CONFIG.SECTION_DIVIDER_WIDTH)
            
            # Clasificación
            print(f"Teoría: {formatted.classification.theory}")
            print(f"Confianza: {formatted.classification.confidence_percentage()}")
            
            # Metadata si disponible
            if formatted.circuit_depth is not None:
                print(f"Profundidad del Circuito: {formatted.circuit_depth}")
            if formatted.method is not None:
                print(f"Motor: {formatted.method}")
            
            # Parámetros clásicos si disponibles
            if formatted.classical_parameters:
                print("\n📐 Parámetros Clásicos (D-Wave):")
                for key, value in formatted.classical_parameters.items():
                    print(f"  {key}: {value:.{CLI_CONFIG.FLOAT_PRECISION}f}")
            
            # Tiempo de ejecución
            print(f"\n⏱️  Tiempo de Ejecución: {formatted.formatted_time()}")
            print(CLI_CONFIG.SECTION_DIVIDER_CHAR * CLI_CONFIG.SECTION_DIVIDER_WIDTH + "\n")
        
        except Exception as e:
            raise CLIException(f"Error mostrando resultado inferencia: {str(e)}")
    
    @staticmethod
    def show_message(message: str, level: str = "info") -> None:
        """
        Muestra mensaje formateado con nivel (info, warning, error).
        
        Args:
            message: Texto del mensaje
            level: "info", "warning", "error", "success"
        
        Raises:
            CLIException: Si hay error de output
        """
        try:
            emoji_map = {
                "info": "ℹ️",
                "warning": "⚠️",
                "error": "❌",
                "success": "✅",
            }
            emoji = emoji_map.get(level, "•")
            print(f"{emoji} {message}")
        except Exception as e:
            raise CLIException(f"Error mostrando mensaje: {str(e)}")