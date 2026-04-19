class QNIMBaseException(Exception):
    """Excepción base para el framework."""
    pass

class SignalProcessingError(QNIMBaseException):
    """Error en el blanqueo o filtrado de la señal."""
    pass

class QuantumBackendError(QNIMBaseException):
    """Error al conectar con IBM Quantum o D-Wave."""
    pass

class PhysicsConsistencyError(QNIMBaseException):
    """Error cuando los parámetros inferidos violan leyes fundamentales."""
    pass