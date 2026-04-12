import numpy as np
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.circuit.library import real_amplitudes
from qiskit.primitives import StatevectorSampler # <--- Sampler V2 nativo local
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as RuntimeSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

class CloudTranspilerSampler(RuntimeSampler):
    """Interceptor que transpila al vuelo para enviar 1 solo Job a IBM."""
    def __init__(self, backend, **kwargs):
        super().__init__(mode=backend, **kwargs)
        self.pm = generate_preset_pass_manager(optimization_level=3, target=backend.target)

    def run(self, pubs, **kwargs):
        print("⚙️  Transpilando circuito a instrucciones nativas (ISA)...")
        isa_pubs = []
        for pub in pubs:
            if isinstance(pub, tuple):
                isa_circuit = self.pm.run(pub[0])
                isa_pubs.append((isa_circuit, *pub[1:]))
            else:
                isa_circuit = self.pm.run(pub.circuit)
                params = getattr(pub, 'parameter_values', None)
                shots = getattr(pub, 'shots', None)
                isa_pubs.append((isa_circuit, params, shots))
                
        print("🛰️  Enviando 1 ÚNICO JOB de predicción a IBM Quantum...")
        return super().run(isa_pubs, **kwargs)

class VQCClassifier:
    def __init__(self, n_qubits=6, token=None, backend_name="ibm_kingston", use_real_hardware=False):
        self.n_qubits = n_qubits
        self.ansatz = real_amplitudes(num_qubits=n_qubits, reps=2)
        self.use_real_hardware = use_real_hardware
        self.real_backend = None
        
        if token:
            try:
                service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
                self.real_backend = service.backend(backend_name)
            except Exception as e:
                print(f"⚠️ Error al conectar con IBM: {e}")

    def initialize_classifier(self, feature_map, initial_weights=None):
        # Usamos StatevectorSampler (V2) para el dummy fit local.
        # COBYLA(maxiter=0) asegura que no intente optimizar nada.
        self.vqc = VQC(
            feature_map=feature_map,
            ansatz=self.ansatz,
            optimizer=COBYLA(maxiter=0),
            sampler=StatevectorSampler(), # <--- LA CLAVE PARA EVITAR EL ERROR
            initial_point=initial_weights
        )
        return self.vqc