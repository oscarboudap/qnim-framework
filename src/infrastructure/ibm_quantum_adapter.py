from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from src.domain.quantum.interfaces import IQuantumClassifier
from src.domain.quantum.entities import InferenceResult
import numpy as np
import os

class IBMQuantumClassifier(IQuantumClassifier):
    """
    Adaptador de Infraestructura para Hardware Cuántico Real (IBM Quantum).
    Implementa el Punto 3.2: Integración de bajo nivel y gestión de latencia.
    """
    def __init__(self, token: str, backend_name: str = "ibm_boston", n_qubits: int = 8):
        # 1. Autenticación Forzada y Limpia
        try:
            print(f"🚀 [QNIM-INFRA] Forzando conexión con instancia ibm-q/open/main...")
            
            # Pasamos el token Y la instancia explícita. 
            # Esto evita que el SDK intente buscar filtros en tu disco.
            self.service = QiskitRuntimeService(
                channel="ibm_quantum_platform",
                token=token,
                instance="ibm-q/open/main"  # <--- CLAVE: Esto elimina el error de 'No matching instances'
            )
            
            # Selección del Backend
            self.backend = self.service.backend(backend_name)
            
            print(f"✅ [QNIM-INFRA] Conexión establecida con éxito.")
            print(f"📡 [QNIM-INFRA] Backend: {backend_name}")
            
        except Exception as e:
            # Si falla con la instancia explícita, probamos una última vez sin ella 
            # pero forzando una sesión nueva.
            try:
                print("⚠️ [QNIM-INFRA] Reintentando sin instancia explícita...")
                self.service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
                self.backend = self.service.backend(backend_name)
                print("✅ [QNIM-INFRA] Conectado en segundo intento.")
            except:
                raise ConnectionError(
                    f"Error crítico de infraestructura: El token no tiene acceso a backends. "
                    f"Causa probable: Tu cuenta de IBM Quantum no tiene backends asignados. "
                    f"Detalle: {e}"
                )

        self.n_qubits = n_qubits

    def predict(self, encoded_data: np.ndarray) -> InferenceResult:
        """Ejecuta la inferencia en la QPU real gestionando la transpilación ISA."""
        
        # 3. Composición del circuito de Inferencia
        full_circuit = self.feature_map.compose(self.ansatz)
        full_circuit.measure_all()
        
        # Vinculación de datos de entrada y pesos del modelo
        all_params = np.concatenate([encoded_data, self.weights])
        bound_circuit = full_circuit.assign_parameters(all_params)
        
        # 4. Transpilación Crítica (Optimización para hardware NISQ)
        # El nivel 3 minimiza errores de compuerta CNOT, vital para ondas gravitacionales
        pm = generate_preset_pass_manager(optimization_level=3, backend=self.backend)
        isa_circuit = pm.run(bound_circuit)
        
        # 5. Ejecución en Hardware Real (Sampler V2)
        sampler = Sampler(mode=self.backend)
        sampler.options.default_shots = 1024 
        
        print(f"\n[QNIM-INFRA] Transpilación completada con éxito.")
        print(f"[QNIM-INFRA] Profundidad ISA: {isa_circuit.depth()} compuertas.")
        print(f"[QNIM-INFRA] Enviando Job a la QPU {self.backend.name}...")
        
        # Envío asíncrono
        job = sampler.run([isa_circuit])
        print(f"🚀 [QNIM-INFRA] Job ID: {job.job_id()}")
        print(f"🔗 [QNIM-INFRA] Seguimiento: https://quantum.ibm.com/jobs/{job.job_id()}")
        
        # 6. Recuperación de Resultados (Bloqueo síncrono hasta que termine la cola)
        result = job.result()[0]
        counts = result.data.meas.get_counts()
        
        # 7. Post-procesamiento: Medición de Paridad (Mitigación de Ruido 5.1)
        total_shots = sum(counts.values())
        even_parity_count = sum(val for b, val in counts.items() if b.count('1') % 2 == 0)
        
        # Probabilidad de la hipótesis de Relatividad General (Clase 0)
        prob_rg = even_parity_count / total_shots
        
        return InferenceResult(
            probabilities={0: prob_rg, 1: 1 - prob_rg},
            predicted_class=0 if prob_rg > 0.5 else 1,
            metadata={
                "backend": self.backend.name,
                "job_id": job.job_id(),
                "isa_depth": isa_circuit.depth(),
                "method": "Real QPU Parity Inference"
            }
        )