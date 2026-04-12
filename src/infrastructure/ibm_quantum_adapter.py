import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from src.domain.quantum.interfaces import IQuantumClassifier
from src.domain.quantum.entities import InferenceResult

class IBMQuantumClassifier(IQuantumClassifier):
    def __init__(self, token: str, backend_name: str = "ibm_fez"):
        self.service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        self.backend = self.service.backend(backend_name)
        
        # Rigor Doctoral: 16 parámetros para capturar la métrica completa
        self.input_dim = 16 
        
        # Definimos los circuitos que faltaban
        self.feature_map = ZZFeatureMap(self.input_dim, reps=1, entanglement='linear')
        self.ansatz = RealAmplitudes(self.input_dim, reps=1)
        self.weights = np.random.uniform(0, 2*np.pi, self.ansatz.num_parameters)
        
        print(f"✅ [IBM-READY] Configurado para {backend_name} con 16-dim embedding.")

    def predict(self, encoded_data: np.ndarray) -> InferenceResult:
        # Construcción del circuito ISA para hardware real
        full_circuit = self.feature_map.compose(self.ansatz)
        full_circuit.measure_all()
        
        # Vinculación de parámetros físicos
        bound_circuit = full_circuit.assign_parameters(np.concatenate([encoded_data, self.weights]))
        
        # Transpilación optimizada para la topología del chip real
        pm = generate_preset_pass_manager(optimization_level=3, backend=self.backend)
        isa_circuit = pm.run(bound_circuit)
        
        # Ejecución en la QPU de IBM
        sampler = Sampler(mode=self.backend)
        job = sampler.run([isa_circuit], shots=8192)
        print(f"📡 Job enviado a IBM: {job.job_id()}. Esperando resultados...")
        
        result = job.result()[0]
        counts = result.data.meas.get_counts()
        
        # Procesamiento de paridad y metrología
        total = sum(counts.values())
        even = sum(v for k, v in counts.items() if k.count('1') % 2 == 0)
        prob = even / total
        
        return InferenceResult(
            probabilities={0: prob, 1: 1 - prob},
            predicted_class=0 if prob > 0.5 else 1,
            metadata={"job_id": job.job_id(), "backend": self.backend.name}
        )