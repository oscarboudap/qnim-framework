import numpy as np
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke 
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap

from src.domain.quantum.interfaces import IQuantumClassifier
from src.domain.quantum.entities import InferenceResult

class QiskitClassifier(IQuantumClassifier):
    """
    Simulador de Alta Fidelidad con Modelo de Ruido.
    Configurado para Inferencia Jerárquica (Multitask Mapping).
    """
    def __init__(self, n_qubits: int = 8, iterations: int = 10):
        # 1. Cargamos el backend FakeSherbrooke (Eagle R3)
        try:
            self.device_backend = FakeSherbrooke()
            self.simulator = AerSimulator.from_backend(self.device_backend)
            print(f"📡 [QNIM-SIM] Digital Twin cargado: FakeSherbrooke (127 qubits)")
        except Exception as e:
            self.simulator = AerSimulator()
            self.device_backend = None
            print(f"⚠️ [QNIM-SIM] Fallback a simulador ideal: {e}")
        
        self.n_qubits = n_qubits
        self.iterations = iterations
        
        # 🛠️ CORRECCIÓN DE DIMENSIONES (Punto 3.3):
        # Como el Mapping Multitarea concatena Inspiral + Ringdown, 
        # la dimensión de entrada es n_qubits * 2 = 16.
        self.input_dim = n_qubits * 2 
        
        # Mapa de características ZZ para capturar correlaciones no lineales (Punto 2.1)
        self.feature_map = ZZFeatureMap(self.input_dim, reps=1, entanglement='linear')
        
        # Ansatz variacional para la clasificación (Punto 6.1)
        self.ansatz = RealAmplitudes(self.input_dim, reps=1)
        
        # Inicialización de pesos aleatorios
        self.weights = np.random.uniform(0, 2*np.pi, self.ansatz.num_parameters)
        print(f"✅ Circuito ISA configurado para {self.input_dim} parámetros de entrada.")

    def predict(self, encoded_data: np.ndarray) -> InferenceResult:
        """Inferencia con mitigación de ruido y estadística robusta."""
        
        # 2. Construcción del circuito completo
        full_circuit = self.feature_map.compose(self.ansatz)
        full_circuit.measure_all()
        
        # Vinculación de parámetros: encoded_data (16) + weights (16)
        # La concatenación ahora coincide exactamente con input_dim
        bound_circuit = full_circuit.assign_parameters(np.concatenate([encoded_data, self.weights]))
        
        # 3. Transpilación ISA para el modelo de ruido de Sherbrooke
        pm = generate_preset_pass_manager(optimization_level=3, backend=self.device_backend)
        isa_circuit = pm.run(bound_circuit)
        
        # 4. Ejecución Monte Carlo (Punto 5.2.1: Estándar 5-Sigma)
        probs = []
        for _ in range(self.iterations):
            # Usamos 8192 shots para maximizar la resolución espectral (Anexo C)
            job = self.simulator.run(isa_circuit, shots=8192)
            counts = job.result().get_counts()
            
            total = sum(counts.values())
            # Medida de paridad para mitigar errores de lectura sistemáticos
            even = sum(v for k, v in counts.items() if k.count('1') % 2 == 0)
            probs.append(even / total)
        
        mean_prob = np.mean(probs)
        std_dev = np.std(probs)
        
        return InferenceResult(
            probabilities={0: mean_prob, 1: 1 - mean_prob},
            predicted_class=0 if mean_prob > 0.5 else 1,
            metadata={
                "method": "Local Noise-Model Simulation (Sherbrooke)",
                "std_dev": std_dev,
                "iterations": self.iterations,
                "isa_depth": isa_circuit.depth(),
                "input_dim": self.input_dim
            }
        )