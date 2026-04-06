from dataclasses import dataclass
from enum import Enum
import numpy as np

class DetectorType(Enum):
    H1 = "Hanford"
    L1 = "Livingston"
    V1 = "Virgo"

@dataclass
class GWSignal:
    strain: np.ndarray
    detector: DetectorType
    sample_rate: int
    gps_start: float # Representa GPSTime del diagrama