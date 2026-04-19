import os
import numpy as np
from dotenv import load_dotenv

# Qiskit Core & Runtime
from qiskit import transpile
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.primitives import StatevectorSampler
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

# Domain
from src.domain.quantum.interfaces import IGateBasedQuantumComputer
from src.domain.quantum.entities import VQCTopology

class IBMQuantumAdapter(IGateBasedQuantumComputer):
    def __init__(self, weights_path: str):
        load_dotenv()
        self.use_real_hardware = os.getenv("USE_REAL_HARDWARE", "False") == "True"
        self.token = os.getenv("IBM_QUANTUM_TOKEN")
        self.backend_name = os.getenv("IBM_BACKEND_NAME", "ibm_kingston")
        
        self.weights_path = weights_path
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"❌ Pesos no encontrados en {weights_path}.")
        self.loaded_weights = np.load(weights_path)

    def execute_circuit(self, circuit_topology: VQCTopology, features: np.ndarray) -> np.ndarray:
        if self.use_real_hardware:
            return self._execute_on_real_qpu(circuit_topology, features)
        return self._execute_on_faithful_simulator(circuit_topology, features)

    def _execute_on_faithful_simulator(self, topology: VQCTopology, features: np.ndarray) -> np.ndarray:
        print(f"🔬 Ejecutando simulación local de alta fidelidad...")
        fmap = ZZFeatureMap(topology.num_qubits, reps=topology.feature_map_reps)
        ansatz = RealAmplitudes(num_qubits=topology.num_qubits, reps=topology.ansatz_reps)
        
        vqc = VQC(
            feature_map=fmap,
            ansatz=ansatz,
            sampler=StatevectorSampler()
        )
        # Inicialización obligatoria para cargar pesos
        vqc.fit(np.zeros((1, topology.num_qubits)), np.array([[1, 0]]))
        return vqc.neural_network.forward(features, self.loaded_weights)

    def _execute_on_real_qpu(self, topology: VQCTopology, features: np.ndarray) -> np.ndarray:
        """
        Ejecución real ISA-compliant (transpilada) y compatible con Plan Open.
        """
        print(f"🛰️  Conectando con IBM Quantum Platform...")
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=self.token)
        backend = service.backend(self.backend_name)
        
        # 1. Construcción del circuito
        fmap = ZZFeatureMap(topology.num_qubits, reps=topology.feature_map_reps)
        ansatz = RealAmplitudes(topology.num_qubits, reps=topology.ansatz_reps)
        qc = fmap.compose(ansatz)
        
        combined_params = np.concatenate([features.flatten(), self.loaded_weights])
        bound_circuit = qc.assign_parameters(combined_params)
        bound_circuit.measure_all()

        # 2. TRANSPILACIÓN (Crítico para evitar el error de hardware en ibm_fez)
        print(f"🔧 Transpilando para {backend.name}...")
        isa_circuit = transpile(bound_circuit, backend=backend, optimization_level=1)

        # 3. EJECUCIÓN mediante SamplerV2 (Formato PUB)
        # No pasamos 'options' en el constructor para evitar errores de Pydantic
        sampler = SamplerV2(mode=backend)
        
        # Pasamos los disparos (shots) directamente en el envío del Job
        print(f"⏳ Enviando Job a la cola real de {backend.name}...")
        job = sampler.run([(isa_circuit, None, 4096)])
        print(f"🆔 Job ID: {job.job_id()}")
        
        result = job.result()
        
        # Acceso a resultados en formato Primitivas V2
        counts = result[0].data.meas.get_counts()
        shots = sum(counts.values())
        p1 = counts.get('1', 0) / shots
        return np.array([[1 - p1, p1]])

    def get_backend_properties(self) -> dict:
        """Cumple el contrato de la interfaz IGateBasedQuantumComputer."""
        try:
            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=self.token)
            backend = service.backend(self.backend_name)
            return {
                'backend_name': backend.name,
                'num_qubits': backend.num_qubits,
                'status': backend.status().status_msg
            }
        except Exception:
            return {'backend_name': self.backend_name, 'num_qubits': 127}