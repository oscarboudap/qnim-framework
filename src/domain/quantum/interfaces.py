# src/domain/quantum/interfaces.py
from abc import ABC, abstractmethod
from ..astrophysics.entities import GWSignal
from .entities import InferenceResult

class IGWRepository(ABC):
    """Contrato para la carga de datos de ondas gravitacionales."""
    @abstractmethod
    def get_signal_by_detector(self, detector_name: str) -> GWSignal:
        pass

class IQuantumEmbedder(ABC):
    """Contrato para transformar señales físicas en circuitos cuánticos."""
    @abstractmethod
    def encode(self, signal: GWSignal) -> any: # 'any' porque el circuito depende de la librería
        pass

class IQuantumClassifier(ABC):
    """Contrato para la inferencia cuántica."""
    @abstractmethod
    def predict(self, encoded_data: any) -> InferenceResult:
        pass