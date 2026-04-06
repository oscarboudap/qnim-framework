from src.domain.quantum.interfaces import IGWRepository, IQuantumClassifier
from .signal_preprocessing_service import SignalPreprocessingService
from .quantum_mapping_service import QuantumMappingService
from .anomaly_generator_service import AnomalyGeneratorService

class ProcessEventUseCase:
    def __init__(
        self, 
        repository: IGWRepository, 
        classifier: IQuantumClassifier,
        preprocessor: SignalPreprocessingService,
        mapper: QuantumMappingService,
        anomaly_generator: AnomalyGeneratorService
    ):
        self._repository = repository
        self._classifier = classifier
        self._preprocessor = preprocessor
        self._mapper = mapper
        self._anomaly_generator = anomaly_generator

    def execute_comparison(self, detector_name: str):
        # 1. Obtener y limpiar señal de LIGO (Relatividad General)
        raw_signal = self._repository.get_signal_by_detector(detector_name)
        clean_signal = self._preprocessor.clean_signal(raw_signal)
        
        # 2. Generar versión con Gravedad Cuántica (LQG)
        # Ahora el método existe con este nombre en el servicio
        anomalous_signal = self._anomaly_generator.apply_quantum_drift(clean_signal)
        
        # 3. Preparar datos para los Qubits
        data_rg = self._mapper.prepare_for_embedding(clean_signal)
        data_lqg = self._mapper.prepare_for_embedding(anomalous_signal)
        
        # 4. Inferencia en Hardware Real (IBM Quantum)
        # Primero enviamos la hipótesis estándar
        print(f"[QNIM] Enviando Hipótesis RG a la QPU...")
        res_rg = self._classifier.predict(data_rg)
        
        # Luego enviamos la hipótesis alternativa
        print(f"[QNIM] Enviando Hipótesis LQG a la QPU...")
        res_lqg = self._classifier.predict(data_lqg)
        
        return res_rg, res_lqg