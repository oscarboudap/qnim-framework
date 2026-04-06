from abc import ABC, abstractmethod
import numpy as np
from src.domain.astrophysics.entities import GWSignal
from .entities import InferenceResult

class IQuantumClassifier(ABC):
    @abstractmethod
    def predict(self, encoded_data: np.ndarray) -> InferenceResult: pass

class IGWRepository(ABC):
    @abstractmethod
    def get_signal_by_detector(self, detector_name: str) -> GWSignal: pass

# Para el IQuantumEmbedder de tu diagrama
class IQuantumEmbedder(ABC):
    @abstractmethod
    def encode(self, signal: GWSignal) -> np.ndarray: pass