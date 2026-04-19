"""
Application Service: Model Training Use Case
==============================================

Orquesta el entrenamiento del modelo cuántico delegando a puertos de infraestructura.

Responsibilities:
- Coordinar carga de datos via IDataLoaderPort
- Delegar preprocesamiento a IPreprocessingPort
- Delegar ML training a IQuantumMLTrainerPort
- Notificar progreso via ITrainingProgressObserver

No contiene: sklearn, Qiskit, joblib - todo inyectado.
Stateless: Sin contadores de iteración internos.
"""

from pathlib import Path
from typing import Optional

from src.application.ports import (
    IDataLoaderPort,
    IQuantumMLTrainerPort,
    IPreprocessingPort,
    ITrainingProgressObserver,
)
from src.application.dto import TrainingMetrics


class ModelTrainingUseCase:
    """
    Caso de Uso: Entrenar el Cerebro Cuántico (VQC).
    
    Delega TODO a puertos - application solo orquesta.
    Sin dependencia directa a frameworks específicos.
    """
    
    def __init__(self,
                 data_loader: IDataLoaderPort,
                 vqc_trainer: IQuantumMLTrainerPort,
                 preprocessing: IPreprocessingPort,
                 progress_observer: Optional[ITrainingProgressObserver] = None):
        """
        Args:
            data_loader: Puerto para cargar datos
            vqc_trainer: Puerto para entrenar VQC
            preprocessing: Puerto para pipeline clásico
            progress_observer: Observer de progreso (opcional)
        """
        self.data_loader = data_loader
        self.trainer = vqc_trainer
        self.preprocessor = preprocessing
        self.observer = progress_observer
    
    def execute(self,
                dataset_path: Path,
                max_events: int = 200,
                max_iterations: int = 100,
                output_models_dir: str = "models/") -> TrainingMetrics:
        """
        Ejecuta el entrenamiento completo del pipeline cuántico.
        
        Args:
            dataset_path: Ruta al dataset de entrenamiento
            max_events: Máximo de eventos a cargar
            max_iterations: Iteraciones del optimizador VQC
            output_models_dir: Donde guardar pesos entrenados
            
        Returns:
            TrainingMetrics: Métricas finales (loss, accuracy, tiempo, etc.)
        """
        # 1. Load data (sin infraestructure code en application)
        print("📂 Cargando datos vía puerto IDataLoaderPort...")
        X_raw, y = self._load_data(dataset_path, max_events)
        
        # 2. Preprocess (StandardScaler + PCA + Mapper)
        print("🗜️ Preprocesando mediante IPreprocessingPort...")
        X_compressed = self.preprocessor.fit_transform(X_raw)
        
        # 3. Save preprocessing pipeline (en formato agnóstico)
        self.preprocessor.save(f"{output_models_dir}/preprocessing_pipeline.pkl")
        
        # 4. Train VQC (sin Qiskit imports aquí)
        print(f"🛰️ Entrenando VQC con {max_iterations} iteraciones...")
        training_result = self.trainer.train_vqc(
            X_train=X_compressed,
            y_train=y,
            num_qubits=12,  # QNIM estándar
            max_iterations=max_iterations,
            optimizer_name="SPSA"
        )
        
        # 5. Notify observer of completion
        if self.observer:
            self.observer.on_training_complete(training_result)
        
        # 6. Save weights
        self.trainer.save_weights(
            weights=training_result['weights'],
            path=f"{output_models_dir}/qnim_vqc_weights.npy"
        )
        
        # 7. Return typed metrics (no dict!)
        return TrainingMetrics(
            final_training_loss=training_result['training_loss'],
            final_validation_accuracy=training_result['validation_accuracy'],
            num_iterations_completed=training_result['iterations'],
            estimated_time_seconds=training_result['execution_time_seconds'],
            model_checkpoint_path=f"{output_models_dir}/qnim_vqc_weights.npy"
        )
    
    def _load_data(self, batch_path: Path, max_events: int) -> tuple:
        """
        Carga datos usando puerto IDataLoaderPort.
        
        Returns:
            (X, y) arrays para entrenamiento
        """
        X, y = [], []
        
        # Listar archivos HDF5
        files = list(batch_path.glob("*.h5"))[:max_events]
        
        for idx, file in enumerate(files):
            # Call port (no I/O aquí en application)
            signal = self.data_loader.prepare_for_quantum(str(file))
            X.append(signal)
            
            # 50% GR, 50% LQG balance
            if idx < len(files) // 2:
                y.append([1, 0])  # Relatividad General
            else:
                y.append([0, 1])  # Anomalía Cuántica
        
        import numpy as np
        return np.array(X), np.array(y)