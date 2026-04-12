import os
import numpy as np
from pathlib import Path

from qiskit.circuit.library import ZZFeatureMap, real_amplitudes
from qiskit_algorithms.optimizers import SPSA
from qiskit.primitives import StatevectorSampler

from src.infrastructure.storage.quantum_dataloader import QuantumDatasetLoader
from src.domain.quantum.vqc_classifier import VQCClassifier

class ModelTrainingService:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data" / "synthetic"
        self.models_dir = self.project_root / "models"
        self.loader = QuantumDatasetLoader(target_samples=64)
        self.iteration = 0 # Contador para los logs

    def execute_training(self, max_events=20):
        print("\n🧠 --- INICIANDO ENTRENAMIENTO QML (12 CÚBITS) ---")
        
        try:
            latest_batch = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith("2026")])[-1]
        except IndexError:
            print("❌ Error: No se encontraron carpetas.")
            return

        print(f"📂 Cargando dataset de entrenamiento desde: {latest_batch.name}")
        X, y = self._prepare_dataset(latest_batch, max_events)
        
        if len(X) == 0: return

        print(f"📊 Dataset cargado: {len(X)} muestras.")
        print("⚙️ Configurando topología a 12 Cúbits...")
        
        fmap = ZZFeatureMap(12, reps=1)
        ansatz = real_amplitudes(num_qubits=12, reps=2)
        
        # --- SISTEMA DE LOGS (CALLBACK) ---
        def training_callback(nfev, params, obj_func, stepsize, accepted):
            # Usamos \r y flush=True para que la línea se reescriba y parezca una barra de carga
            print(f"\r🚀 [SPSA] Evaluaciones: {nfev} | Función de Coste (Pérdida): {obj_func:.4f} ", end="", flush=True)
            
        from qiskit_machine_learning.algorithms.classifiers import VQC
        vqc_engine = VQC(
            feature_map=fmap,
            ansatz=ansatz,
            optimizer=SPSA(maxiter=100),
            sampler=StatevectorSampler(),
            callback=training_callback # Inyectamos el logger corregido
        )

        print(f"🛰️  Optimizando parámetros cuánticos (SPSA)...")
        vqc_engine.fit(X, y)
        
        os.makedirs(self.models_dir, exist_ok=True)
        weights_path = self.models_dir / "qnim_vqc_weights.npy"
        np.save(weights_path, vqc_engine.weights)
        
        print("\n\n✅ ENTRENAMIENTO COMPLETADO.")
        print(f"💾 Cerebro cuántico guardado exitosamente en: {weights_path}")

    def _prepare_dataset(self, batch_path, max_events):
        X, y = [], []
        import h5py
        
        for file in list(batch_path.glob("*.h5"))[:max_events]: 
            # Cortamos a 12 características (coincidiendo con los cúbits)
            X.append(self.loader.prepare_for_quantum(str(file))[:12]) 
            
            with h5py.File(file, 'r') as f:
                theory = f.attrs.get("true_theory", b"Kerr")
                if isinstance(theory, bytes): theory = theory.decode()
                
                if theory == "Kerr" or theory == "RG":
                    y.append([1, 0])
                else:
                    y.append([0, 1])
                    
        return np.array(X), np.array(y)