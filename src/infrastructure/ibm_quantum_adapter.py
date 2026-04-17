import os
import numpy as np
from dotenv import load_dotenv
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.primitives import StatevectorSampler # Para simulación fiel
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Session

from src.domain.quantum.interfaces import IGateBasedQuantumComputer
from src.domain.quantum.entities import VQCTopology

class IBMQuantumAdapter(IGateBasedQuantumComputer):
    """
    Adaptador de nivel de investigación para computación cuántica basada en puertas.
    Gestiona la ejecución híbrida: Simulación local de alta fidelidad vs QPU Real.
    """
    
    def __init__(self, weights_path: str):
        load_dotenv() # Carga las variables del archivo .env
        self.use_real_hardware = os.getenv("USE_REAL_HARDWARE", "False") == "True"
        self.token = os.getenv("IBM_QUANTUM_TOKEN")
        self.backend_name = os.getenv("IBM_BACKEND_NAME", "ibm_kingston")
        
        self.weights_path = weights_path
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"❌ Pesos no encontrados en {weights_path}. Entrena el modelo primero.")
        self.loaded_weights = np.load(weights_path)

    def execute_circuit(self, circuit_topology: VQCTopology, features: np.ndarray) -> np.ndarray:
        """
        Punto de entrada único que bifurca la ejecución según el entorno configurado.
        """
        if self.use_real_hardware:
            return self._execute_on_real_qpu(circuit_topology, features)
        else:
            return self._execute_on_faithful_simulator(circuit_topology, features)

    def _execute_on_faithful_simulator(self, topology: VQCTopology, features: np.ndarray) -> np.ndarray:
        """
        Simulación de alta fidelidad (Statevector) para validación científica local.
        """
        print(f"🔬 Ejecutando simulación local de alta fidelidad (Modo: {os.getenv('MODE')})...")
        
        fmap = ZZFeatureMap(topology.num_qubits, reps=topology.feature_map_reps)
        ansatz = RealAmplitudes(num_qubits=topology.num_qubits, reps=topology.ansatz_reps)
        
        # Usamos VQC con StatevectorSampler para la máxima precisión teórica
        vqc = VQC(
            feature_map=fmap,
            ansatz=ansatz,
            sampler=StatevectorSampler()
        )
        
        # Inicialización fantasma obligatoria en Qiskit ML para cargar la red
        vqc.fit(np.zeros((1, topology.num_qubits)), np.array([[1, 0]]))
        
        # Forward pass con los pesos entrenados
        return vqc.neural_network.forward(features, self.loaded_weights)

    def _execute_on_real_qpu(self, topology: VQCTopology, features: np.ndarray) -> np.ndarray:
        """
        Ejecución real en hardware de IBM (ibm_kingston) con mitigación de errores.
        """
        print(f"🛰️  Enviando Job a IBM Quantum: {self.backend_name}...")
        
        service = QiskitRuntimeService(channel="ibm_quantum", token=self.token)
        backend = service.backend(self.backend_name)
        
        # Construcción del circuito
        fmap = ZZFeatureMap(topology.num_qubits, reps=topology.feature_map_reps)
        ansatz = RealAmplitudes(topology.num_qubits, reps=topology.ansatz_reps)
        qc = fmap.compose(ansatz)
        
        # Asignación de parámetros (features + weights)
        # Combinamos ambos arrays para mapearlos a los parámetros del circuito
        combined_params = np.concatenate([features.flatten(), self.loaded_weights])
        bound_circuit = qc.assign_parameters(combined_params)
        bound_circuit.measure_all()

        # Ejecución mediante Sesión de Runtime para optimizar la latencia
        with Session(service=service, backend=backend) as session:
            sampler = SamplerV2(session=session)
            sampler.options.resilience_level = 1 # Mitigación de errores activa (Readout)
            
            job = sampler.run([bound_circuit], shots=4096)
            result = job.result()
            
            # Procesamiento de counts a probabilidades (Clasificación binaria)
            counts = result[0].data.meas.get_counts()
            shots = sum(counts.values())
            # Probabilidad de la clase 1 (Anomalía)
            p1 = counts.get('1', 0) / shots
            return np.array([[1 - p1, p1]])