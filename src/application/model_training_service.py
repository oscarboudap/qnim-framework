import os
import sys
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from src.domain.quantum.vqc_architecture import QNIMQuantumCircuit
from qiskit_algorithms.optimizers import SPSA
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms.classifiers import VQC

class ModelTrainingService:
    """
    Caso de Uso: Entrenar el Cerebro Cuántico (VQC).
    Implementa el pipeline de mitigación de la maldición de dimensionalidad.
    """
    def __init__(self, dataloader_port, models_dir: str):
        self.loader = dataloader_port # Puerto de lectura de datos
        self.models_dir = Path(models_dir)
        self.topology = QNIMQuantumCircuit.get_standard_topology()
        self.current_iter = 0 # Inicializamos el contador de iteraciones

    def execute_training(self, dataset_path: Path, max_events: int = 200):
        print("🗜️ Construyendo Pipeline Cuántico (StandardScaler -> PCA -> MinMax)...")
        quantum_pipeline = Pipeline([
            ('scaler', StandardScaler()), 
            ('pca', PCA(n_components=self.topology.num_qubits)), 
            ('angle_mapper', MinMaxScaler(feature_range=(-np.pi, np.pi))) 
        ])
        
        # 1. Cargar datos (Infraestructura)
        X_raw, y = self._load_data(dataset_path, max_events)
        
        # 2. Comprimir datos
        X_compressed = quantum_pipeline.fit_transform(X_raw)
        
        # Guardamos el pipeline clásico
        os.makedirs(self.models_dir, exist_ok=True)
        joblib.dump(quantum_pipeline, self.models_dir / "qnim_preprocessing_pipeline.pkl")

        # 3. Configurar y Entrenar la Red Cuántica
        from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
        fmap = ZZFeatureMap(self.topology.num_qubits, reps=self.topology.feature_map_reps)
        ansatz = RealAmplitudes(num_qubits=self.topology.num_qubits, reps=self.topology.ansatz_reps, entanglement=self.topology.entanglement_strategy)
        
        max_iter = 100
        self.current_iter = 0
        
        # --- EL TRACKER DE PROGRESO ---
        def training_callback(*args):
            self.current_iter += 1
            
            # Qiskit manda 2 o 5 parámetros dependiendo del optimizador/versión
            if len(args) == 5:
                # SPSA manda: (evaluaciones, parametros, coste, stepsize, aceptado)
                obj_func_eval = args[2] 
            elif len(args) == 2:
                # VQC estándar manda: (pesos, coste)
                obj_func_eval = args[1]
            else:
                obj_func_eval = 0.0 # Seguridad
                
            progress = (self.current_iter / max_iter) * 100
            # El \r sobrescribe la línea
            sys.stdout.write(f"\r⏳ [QML Training] Progreso: {progress:5.1f}% | Iteración: {self.current_iter}/{max_iter} | Función de Coste: {obj_func_eval:.4f}")
            sys.stdout.flush()
            if self.current_iter == max_iter:
                print() # Salto de línea limpio al terminar
        # ------------------------------
        vqc_engine = VQC(
            feature_map=fmap,
            ansatz=ansatz,
            optimizer=SPSA(maxiter=max_iter),
            sampler=StatevectorSampler(),
            callback=training_callback # Inyectamos el callback
        )

        print(f"🛰️ Optimizando topología de {self.topology.num_qubits} cúbits en el Espacio de Hilbert...")
        vqc_engine.fit(X_compressed, y)
        
        # 4. Guardar los pesos cuánticos
        np.save(self.models_dir / "qnim_vqc_weights.npy", vqc_engine.weights)
        print("✅ Entrenamiento Híbrido Completado.")

    def _load_data(self, batch_path: Path, max_events: int):
        X, y = [], []
        files = list(batch_path.glob("*.h5"))[:max_events]
        for idx, file in enumerate(files):
            signal = self.loader.prepare_for_quantum(str(file))
            X.append(signal)
            # El orden del generador es 50% RG (idx < mitad), 50% LQG
            if idx < (len(files) // 2):
                y.append([1, 0]) # Relatividad General
            else:
                y.append([0, 1]) # Anomalía Cuántica
        return np.array(X), np.array(y)