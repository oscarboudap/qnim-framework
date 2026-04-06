from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from src.domain.quantum.interfaces import IQuantumClassifier
from src.domain.quantum.entities import InferenceResult
import numpy as np

class IBMQuantumClassifier(IQuantumClassifier):
    def __init__(self, token: str, backend_name: str = "ibm_kingston", n_qubits: int = 8):
        self.service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance="ibm-q/open/main")
        self.backend = self.service.backend(backend_name)
        self.n_qubits = n_qubits
        self.feature_map = ZZFeatureMap(n_qubits, reps=2, entanglement='circular')
        self.ansatz = RealAmplitudes(n_qubits, reps=4)
        self.weights = np.random.uniform(0, 2*np.pi, self.ansatz.num_parameters)

    def predict(self, encoded_data: np.ndarray) -> InferenceResult:
        full_circuit = self.feature_map.compose(self.ansatz)
        full_circuit.measure_all()
        bound_circuit = full_circuit.assign_parameters(np.concatenate([encoded_data, self.weights]))
        
        pm = generate_preset_pass_manager(optimization_level=3, backend=self.backend)
        isa_circuit = pm.run(bound_circuit)
        
        sampler = Sampler(mode=self.backend)
        job = sampler.run([isa_circuit])
        print(f"🚀 Job enviado: {job.job_id()}")
        
        result = job.result()[0]
        counts = result.data.meas.get_counts()
        total = sum(counts.values())
        even = sum(v for k, v in counts.items() if k.count('1') % 2 == 0)
        prob_0 = even / total
        
        return InferenceResult({0: prob_0, 1: 1-prob_0}, 0 if prob_0 > 0.5 else 1, {"job_id": job.job_id()})