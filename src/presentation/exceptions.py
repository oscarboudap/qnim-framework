"""
Presentation Layer: Exception Framework
========================================

Excepciones específicas para la capa de presentación.
"""


class PresentationException(Exception):
    """
    Excepción base para errores en la capa de presentación.
    
    Se usa cuando hay problemas al formatear o mostrar datos
    para el usuario final.
    """
    pass


class CLIException(PresentationException):
    """
    Excepción para errores en la presentación CLI.
    
    Ejemplos:
        - Datos inválidos para formatear
        - Terminal encoding issues
        - Output stream errors
    """
    pass


class VisualizationException(PresentationException):
    """
    Excepción para errores en visualización/gráficos.
    
    Ejemplos:
        - Matplotlib falla al crear figura
        - Error al guardar archivo PNG/PDF
        - Directorio de salida no accesible
        - Datos insuficientes para gráfico
    """
    pass


class DTOConversionException(PresentationException):
    """
    Excepción cuando conversión de DTO → formato display falla.
    
    Ejemplos:
        - DTO tiene campos None inesperadamente
        - Formato no soportado
        - Validación de datos fallida
    """
    pass


class FormattingException(PresentationException):
    """
    Excepción para errores al formatear texto/números.
    
    Ejemplos:
        - Formato de número inválido
        - Template de string no exists
        - Encoding issues
    """
    pass
