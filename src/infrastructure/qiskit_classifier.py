from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from src.domain.quantum.interfaces import IQuantumClassifier
from src.domain.quantum.entities import InferenceResult
import numpy as np

class QiskitClassifier(IQuantumClassifier):
    def __init__(self, n_qubits: int = 8, iterations: int = 5):
        self.device_backend = FakeSherbrooke() # Digital Twin de IBM
        self.simulator = AerSimulator.from_backend(self.device_backend) # AerSimulator del diagrama
        self.n_qubits = n_qubits
        self.iterations = iterations
        self.feature_map = ZZFeatureMap(n_qubits, reps=2)
        self.ansatz = RealAmplitudes(n_qubits, reps=4)
        self.weights = np.random.uniform(0, 2*np.pi, self.ansatz.num_parameters)

    def predict(self, encoded_data: np.ndarray) -> InferenceResult:
        full_circuit = self.feature_map.compose(self.ansatz)
        full_circuit.measure_all()
        params = np.concatenate([encoded_data, self.weights])
        bound = full_circuit.assign_parameters(params)
        
        pm = generate_preset_pass_manager(optimization_level=3, backend=self.device_backend)
        isa_circuit = pm.run(bound)
        
        results = []
        for _ in range(self.iterations):
            job = self.simulator.run(isa_circuit, shots=1024)
            counts = job.result().get_counts()
            p0 = sum(v for k, v in counts.items() if k.count('1') % 2 == 0) / 1024
            results.append(p0)
            
        mean_p = np.mean(results)
        return InferenceResult({0: mean_p, 1: 1-mean_p}, 0 if mean_p > 0.5 else 1, {"std": np.std(results)})