from src.domain.quantum.interfaces import IGWRepository, IQuantumClassifier
from .signal_preprocessing_service import SignalPreprocessingService
from .quantum_mapping_service import QuantumMappingService
from .anomaly_generator_service import AnomalyGeneratorService

class ProcessEventUseCase:
    def __init__(self, repository, classifier, preprocessor, mapper, anomaly_generator):
        self._repo = repository
        self._classifier = classifier
        self._preprocessor = preprocessor
        self._mapper = mapper
        self._anomaly_gen = anomaly_generator

    def execute_comparison(self, detector_name: str, epsilon: float = 0.1):
        # Flujo principal según tu diagrama
        raw = self._repo.get_signal_by_detector(detector_name)
        clean = self._preprocessor.clean_signal(raw)
        anomalous = self._anomaly_gen.apply_quantum_drift(clean, epsilon)
        
        data_rg = self._mapper.prepare_for_embedding(clean)
        data_lqg = self._mapper.prepare_for_embedding(anomalous)
        
        return self._classifier.predict(data_rg), self._classifier.predict(data_lqg)