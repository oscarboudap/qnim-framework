from src.domain.quantum.interfaces import IGWRepository, IQuantumClassifier
from .signal_preprocessing_service import SignalPreprocessingService
from .quantum_mapping_service import QuantumMappingService
from .anomaly_generator_service import AnomalyGeneratorService

class ProcessEventUseCase:
    def __init__(self, repository, classifier, preprocessor, mapper, anomaly_generator):
        self._repo = repository
        self._classifier = classifier
        self._preprocessor = preprocessor # Ahora con Whitening
        self._mapper = mapper
        self._anomaly_generator = anomaly_generator

    def execute_comparison(self, detector_name: str, epsilon: float = 0.1):
        # 1. EXTRACCIÓN Y LIMPIEZA (Foco en el Punto 4.3 de tu índice)
        raw_signal = self._repo.get_signal_by_detector(detector_name)
        
        # Primero blanqueamos para eliminar el ruido de color de LIGO
        white_signal = self._preprocessor.whitening(raw_signal)
        # Luego filtramos para quedarnos con la banda de los QNM (30-500Hz)
        clean_signal = self._preprocessor.bandpass_filter(white_signal)
        
        # 2. GENERACIÓN DE HIPÓTESIS (Punto 2.2.2.1)
        # La anomalía se inyecta sobre la señal limpia para simular correcciones de Planck
        anomalous_signal = self._anomaly_generator.apply_quantum_drift(clean_signal, epsilon)
        
        # 3. MAPPING Y EMBEDDING CUÁNTICO (Punto 3.3)
        data_rg = self._mapper.prepare_for_embedding(clean_signal)
        data_lqg = self._mapper.prepare_for_embedding(anomalous_signal)
        
        # 4. INFERENCIA NATIVA (Punto 6.1)
        res_rg = self._classifier.predict(data_rg)
        res_lqg = self._classifier.predict(data_lqg)
        
        return res_rg, res_lqg