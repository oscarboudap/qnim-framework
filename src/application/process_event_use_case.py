from src.domain.quantum.interfaces import IGWRepository, IQuantumClassifier
from .signal_preprocessing_service import SignalPreprocessingService
from .quantum_mapping_service import QuantumMappingService
from .anomaly_generator_service import AnomalyGeneratorService

class ProcessEventUseCase:
    def __init__(self, repository, classifier, preprocessor, mapper, anomaly_generator):
        # Vinculamos las dependencias a la instancia (self)
        self.repository = repository
        self.classifier = classifier
        self.preprocessor = preprocessor
        self.mapper = mapper
        self.anomaly_generator = anomaly_generator

    def execute_comparison(self, detector_name: str):
        # 1. Obtención de datos crudos desde el repositorio
        # Aquí es donde fallaba: ahora self.repository existe
        raw_signal = self.repository.get_signal_by_detector(detector_name)
        
        # 2. Preprocesamiento (Whitening + Bandpass)
        white_signal = self.preprocessor.whitening(raw_signal)
        clean_rg = self.preprocessor.bandpass_filter(white_signal)

        # 3. Inyección de Física de Campo Fuerte (Cuerdas/LQG)
        # Usamos el nuevo método de teoría que definimos
        clean_lqg = self.anomaly_generator.apply_theory_drift(
            clean_rg, 
            theory="STRING_FUZZBALL", 
            epsilon=0.25
        )

        # 4. Quantum Embedding (Mapping Multitarea: Inspiral + Ringdown)
        data_rg = self.mapper.prepare_multitask_embedding(clean_rg)
        data_lqg = self.mapper.prepare_multitask_embedding(clean_lqg)

        # 5. Inferencia Cuántica (Envío a IBM Quantum si está activo)
        print(f"📡 [QNIM] Enviando circuitos al backend para inferencia...")
        res_rg = self.classifier.predict(data_rg)
        res_lqg = self.classifier.predict(data_lqg)

        return res_rg, res_lqg