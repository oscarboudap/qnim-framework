# src/domain/shared/exceptions.py

class DomainException(Exception):
    """Clase base para excepciones del dominio."""
    pass

class SignalProcessingError(DomainException):
    """Error ocurrido durante el procesamiento de la señal."""
    pass

class QuantumEmbeddingError(DomainException):
    """Error al intentar mapear datos clásicos a qubits."""
    pass