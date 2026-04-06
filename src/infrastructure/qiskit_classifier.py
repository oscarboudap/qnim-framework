import numpy as np
from qiskit_aer import AerSimulator
# Cambiamos a Sherbrooke para máxima compatibilidad entre versiones del SDK
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke 
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from src.domain.quantum.interfaces import IQuantumClassifier
from src.domain.quantum.entities import InferenceResult

class QiskitClassifier(IQuantumClassifier):
    """
    Simulador de Alta Fidelidad con Modelo de Ruido.
    Utiliza el Digital Twin de FakeSherbrooke (127 qubits) para mimetizar hardware real.
    """
    def __init__(self, n_qubits: int = 8, iterations: int = 5):
        # 1. Cargamos el backend falso y su modelo de ruido asociado
        try:
            self.device_backend = FakeSherbrooke()
            self.simulator = AerSimulator.from_backend(self.device_backend)
            print(f"📡 [QNIM-SIM] Modelo de ruido cargado: FakeSherbrooke (Eagle R3)")
        except Exception as e:
            # Fallback a simulador ideal si falla el modelo de ruido (Plan B de robustez)
            self.simulator = AerSimulator()
            print(f"⚠️ [QNIM-SIM] Usando simulador ideal. Error cargando ruido: {e}")
        
        self.n_qubits = n_qubits
        self.iterations = iterations # Número de ejecuciones Monte Carlo
        
        # 2. Arquitectura del Circuito Variacional
        self.feature_map = ZZFeatureMap(n_qubits, reps=2, entanglement='circular')
        self.ansatz = RealAmplitudes(n_qubits, reps=4)
        
        # Pesos fijos para el barrido de sensibilidad
        self.weights = np.random.uniform(0, 2*np.pi, self.ansatz.num_parameters)

    def predict(self, encoded_data: np.ndarray) -> InferenceResult:
        """Inferencia estadística mediante muestreo con ruido."""
        
        # 3. Construcción del circuito ISA
        full_circuit = self.feature_map.compose(self.ansatz)
        full_circuit.measure_all()
        
        all_params = np.concatenate([encoded_data, self.weights])
        bound_circuit = full_circuit.assign_parameters(all_params)
        
        # Transpilación optimizada para el backend elegido
        # Nivel 3 para intentar mitigar el ruido simulado
        pm = generate_preset_pass_manager(optimization_level=3, backend=self.device_backend)
        isa_circuit = pm.run(bound_circuit)
        
        # 4. Ejecución Monte Carlo (Análisis de Robustez)
        results_list = []
        for i in range(self.iterations):
            # Simulamos el ruido real en cada iteración
            job = self.simulator.run(isa_circuit, shots=1024)
            counts = job.result().get_counts()
            
            # Cálculo de paridad (Mitigación de errores post-selección)
            total_shots = sum(counts.values())
            even_parity = sum(v for k, v in counts.items() if k.count('1') % 2 == 0)
            results_list.append(even_parity / total_shots)
        
        # 5. Métricas de salida
        mean_prob = np.mean(results_list)
        std_dev = np.std(results_list)
        
        return InferenceResult(
            probabilities={0: mean_prob, 1: 1 - mean_prob},
            predicted_class=0 if mean_prob > 0.5 else 1,
            metadata={
                "method": "Local Noise Simulation (Sherbrooke)",
                "std_dev": std_dev,
                "iterations": self.iterations,
                "isa_depth": isa_circuit.depth(),
                "qubits": self.n_qubits
            }
        )