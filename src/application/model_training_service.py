import os
import numpy as np
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

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
        self.loader = QuantumDatasetLoader(target_samples=16384) 

    def execute_training(self, max_events=200):
        print("\n🧠 --- INICIANDO ENTRENAMIENTO QML (12 CÚBITS + PIPELINE RIGUROSO) ---")
        
        try:
            latest_batch = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith("2026")])[-1]
        except IndexError:
            print("❌ Error: No se encontraron carpetas.")
            return

        print(f"📂 Cargando dataset de entrenamiento desde: {latest_batch.name}")
        X_raw, y = self._prepare_dataset(latest_batch, max_events)
        
        if len(X_raw) == 0: return
        print(f"📊 Dataset cargado: {len(X_raw)} muestras.")
        
        # --- EL PIPELINE DE MÁXIMO RIGOR ---
        print("🗜️ Construyendo Pipeline Cuántico (StandardScaler -> PCA -> MinMax[-π, π])...")
        quantum_pipeline = Pipeline([
            ('scaler', StandardScaler()), # Normaliza el ruido gravitacional
            ('pca', PCA(n_components=12)), # Extrae los 12 tensores principales
            ('angle_mapper', MinMaxScaler(feature_range=(-np.pi, np.pi))) # Mapea a rotaciones de cúbits
        ])
        
        X = quantum_pipeline.fit_transform(X_raw)
        
        os.makedirs(self.models_dir, exist_ok=True)
        # Guardamos el pipeline completo, no solo el PCA
        joblib.dump(quantum_pipeline, self.models_dir / "qnim_preprocessing_pipeline.pkl")
        print("💾 Pipeline de preprocesamiento guardado correctamente.")
        # -----------------------------------

        print("⚙️ Configurando topología a 12 Cúbits con Angle Encoding...")
        fmap = ZZFeatureMap(12, reps=1)
        ansatz = real_amplitudes(num_qubits=12, reps=2)
        
        def training_callback(nfev, params, obj_func, stepsize, accepted):
            print(f"\r🚀 [SPSA] Evaluaciones: {nfev} | Función de Coste (Pérdida): {obj_func:.4f} ", end="", flush=True)
            
        from qiskit_machine_learning.algorithms.classifiers import VQC
        vqc_engine = VQC(
            feature_map=fmap,
            ansatz=ansatz,
            optimizer=SPSA(maxiter=100),
            sampler=StatevectorSampler(),
            callback=training_callback
        )

        print(f"\n🛰️  Optimizando parámetros cuánticos (SPSA)...")
        vqc_engine.fit(X, y)
        
        weights_path = self.models_dir / "qnim_vqc_weights.npy"
        np.save(weights_path, vqc_engine.weights)
        
        print("\n\n✅ ENTRENAMIENTO COMPLETADO.")

    def _prepare_dataset(self, batch_path, max_events):
        X, y = [], []
        
        for file in list(batch_path.glob("*.h5"))[:max_events]: 
            signal = self.loader.prepare_for_quantum(str(file))
            X.append(signal) 
            
            # --- EL TRUCO DEL NOMBRE DE ARCHIVO ---
            # file.stem saca "event_00150", split('_')[1] saca "00150", int() lo hace 150
            file_num = int(file.stem.split('_')[1])
            
            if file_num < 100:
                y.append([1, 0]) # RG (Kerr)
            else:
                y.append([0, 1]) # LQG (Anomalía)
            # --------------------------------------
                    
        return np.array(X), np.array(y)