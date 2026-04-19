"""
Script Layer Exception Hierarchy

Provides typed exception handling for script operations, separate from
domain/application/infrastructure exceptions.

Inheritance:
  ScriptException (BASE)
    ├── ScriptConfigurationException (config/init)
    ├── ScriptExecutionException (runtime)
    ├── ScriptDataException (data loading/validation)
    ├── ScriptComputationException (algo/model exec)
    └── ScriptIOException (file/output)
"""

from typing import Optional


class ScriptException(Exception):
    """
    Base exception for all script layer errors.
    
    Allows catching/handling script-specific exceptions without affecting
    domain/application/infrastructure layer exceptions.
    """
    
    def __init__(self, message: str, context: Optional[dict] = None) -> None:
        """
        Initialize script exception.
        
        Args:
          message: Human-readable error message
          context: Optional dict with additional error context
        """
        self.message = message
        self.context = context or {}
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format exception message with context."""
        if self.context:
            context_str = " | ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} [{context_str}]"
        return self.message


class ScriptConfigurationException(ScriptException):
    """
    Raised when script configuration is invalid.
    
    Causes:
      - Invalid config parameter values
      - Missing required configuration
      - Conflicting configuration options
    """
    pass


class ScriptExecutionException(ScriptException):
    """
    Raised when script execution fails at runtime.
    
    Causes:
      - Dependency injection container failure
      - Application layer use case failure
      - Unexpected execution flow
    """
    pass


class ScriptDataException(ScriptException):
    """
    Raised when data loading/validation fails.
    
    Causes:
      - File not found or corrupted
      - Data shape/type mismatch
      - Insufficient data for operation
    """
    pass


class ScriptComputationException(ScriptException):
    """
    Raised when computation/model execution fails.
    
    Causes:
      - Training failure
      - Inference error
      - Numerical instability
    """
    pass


class ScriptIOException(ScriptException):
    """
    Raised when file I/O operations fail.
    
    Causes:
      - Cannot read input file
      - Cannot write output file
      - Permission denied
    """
    pass
