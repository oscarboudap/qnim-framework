"""
Infrastructure Layer: Custom Exceptions
========================================

Excepciones específicas de la capa de infraestructura.
Permite a application layer manejar errores de manera presisa sin importar
(sin conocer details de frameworks específicos).
"""


class InfrastructureException(Exception):
    """Excepción base para la capa de infraestructura."""
    pass


class DataLoaderException(InfrastructureException):
    """Error al cargar datos (ficheros, parsing HDF5, etc.)."""
    pass


class StorageException(InfrastructureException):
    """Error al guardar/persistir datos (permisos, espacio en disco, etc.)."""
    pass


class QuantumComputeException(InfrastructureException):
    """Error en computación cuántica (backend offline, jobId invalido, etc.)."""
    pass


class TrainingException(InfrastructureException):
    """Error durante entrenamiento de modelo cuántico."""
    pass


class PreprocessingException(InfrastructureException):
    """Error durante preprocesamiento/transformación de datos."""
    pass


class ReportingException(InfrastructureException):
    """Error durante generación de reportes/visualizaciones."""
    pass
