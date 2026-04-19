import os
import numpy as np
from dotenv import load_dotenv
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.primitives import StatevectorSampler
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from src.domain.quantum.interfaces import IGateBasedQuantumComputer
from src.domain.quantum.entities import VQCTopology

class IBMQuantumAdapter(IGateBasedQuantumComputer):
    """
    Adaptador QNIM: Ejecución en hardware real respetando el .env
    """
    
    def __init__(self, weights_path: str):
        load_dotenv() 
        self.use_real_hardware = os.getenv("USE_REAL_HARDWARE", "False") == "True"
        self.token = os.getenv("IBM_QUANTUM_TOKEN")
        # Leemos explícitamente el backend de tu .env
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
        print(f"🔬 Ejecutando simulación local (Modo: {os.getenv('MODE', 'SYNTHETIC')})...")
        fmap = ZZFeatureMap(topology.num_qubits, reps=topology.feature_map_reps)
        ansatz = RealAmplitudes(num_qubits=topology.num_qubits, reps=topology.ansatz_reps)
        vqc = VQC(feature_map=fmap, ansatz=ansatz, sampler=StatevectorSampler())
        vqc.fit(np.zeros((1, topology.num_qubits)), np.array([[1, 0]]))
        return vqc.neural_network.forward(features, self.loaded_weights)

    def _execute_on_real_qpu(self, topology: VQCTopology, features: np.ndarray) -> np.ndarray:
        """
        Ejecución directa en el backend especificado en el .env
        """
        # 1. Autenticación
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=self.token)
        
        # 2. SELECCIÓN FIJA DEL BACKEND (Tu .env manda)
        print(f"🛰️  Conectando al backend configurado: {self.backend_name}...")
        backend = service.backend(self.backend_name)
        
        # 3. Construcción del circuito
        fmap = ZZFeatureMap(topology.num_qubits, reps=topology.feature_map_reps)
        ansatz = RealAmplitudes(topology.num_qubits, reps=topology.ansatz_reps)
        qc = fmap.compose(ansatz)
        
        # 4. Asignación de parámetros
        params = np.concatenate([features.flatten(), self.loaded_weights])
        bound_circuit = qc.assign_parameters(params)
        bound_circuit.measure_all()

        # 5. Ejecución en modo directo (SamplerV2 con sintaxis PUB)
        sampler = SamplerV2(mode=backend)
        
        print(f"⏳ Enviando Job a {self.backend_name}. Esperando resultados...")
        # (circuito, parámetros, disparos)
        job = sampler.run([(bound_circuit, None, 4096)])
        print(f"🆔 Job ID: {job.job_id()}")
        
        # Sincronización: El script esperará aquí
        result = job.result()
        
        # 6. Extracción de probabilidades
        pub_result = result[0]
        counts = pub_result.data.meas.get_counts()
        shots = sum(counts.values())
        p1 = counts.get('1', 0) / shots
        
        return np.array([[1 - p1, p1]])