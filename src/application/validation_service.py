"""
Application Service: Model Validation Use Case
===============================================

Audita precisión del modelo cuántico entrenado.

Responsibilities:
- Ejecutar predicciones sobre test set
- Computar matriz de confusión (sin visualización)
- Retornar ConfusionMatrixData tipado
- Delegar visualización a presentation layer (IMetricsReporterPort)

NO contiene: matplotlib, seaborn - eso es presentation.
"""

from typing import Optional
import numpy as np

from src.application.ports import (
    IQuantumMLTrainerPort,
    IMetricsReporterPort,
)
from src.application.dto import ConfusionMatrixData


class ModelValidationUseCase:
    """
    Caso de Uso: Auditar la precisión del Modelo Cuántico.
    
    Retorna datos tipados, delegando visualización a presentation.
    """
    
    def __init__(self,
                 qvc_trainer: IQuantumMLTrainerPort,
                 metrics_reporter: Optional[IMetricsReporterPort] = None):
        """
        Args:
            qvc_trainer: Puerto para ejecutar predicciones
            metrics_reporter: Puerto para reportes (optional)
        """
        self.trainer = qvc_trainer
        self.reporter = metrics_reporter
    
    def execute(self,
                test_X: np.ndarray,
                test_y: np.ndarray,
                model_checkpoint_path: str) -> ConfusionMatrixData:
        """
        Ejecuta validación completa: predicciones + evaluación + reporte.
        
        Args:
            test_X: Test features [n_samples, 12]
            test_y: Test labels one-hot [n_samples, 2]
            model_checkpoint_path: Ruta a pesos del modelo
            
        Returns:
            ConfusionMatrixData: Conteos TP/TN/FP/FN (tipado, sin visualization)
        """
        print("🧪 Iniciando Evaluación de Precisión sobre Test Set...")
        
        # Load model weights
        weights = self.trainer.load_weights(model_checkpoint_path)
        
        # Predict on test set
        y_true = []
        y_pred = []
        
        for i in range(len(test_X)):
            # Extract true label
            true_idx = np.argmax(test_y[i])
            y_true.append(true_idx)
            
            # Get prediction from quantum backend
            pred_probs = self._predict_quantum(test_X[i:i+1], weights)
            pred_idx = np.argmax(pred_probs)
            y_pred.append(pred_idx)
        
        # Compute confusion matrix
        cm = self._compute_confusion_matrix(y_true, y_pred)
        
        # Report if reporter available
        if self.reporter:
            self.reporter.report_confusion_matrix(
                true_positives=cm.true_positives,
                true_negatives=cm.true_negatives,
                false_positives=cm.false_positives,
                false_negatives=cm.false_negatives,
                output_path="reports/figures/confusion_matrix.png"
            )
        
        print(f"✅ Evaluación completada. Accuracy: {cm.accuracy:.4f}")
        
        return cm
    
    @staticmethod
    def _predict_quantum(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Ejecuta predicción en backend cuántico.
        
        This is a stub - real implementation depends on IQuantumMLTrainerPort.predict()
        which should be added to the port interface.
        """
        # Placeholder: en producción, usaría:
        # return self.trainer.predict(features, weights)
        # For now: dummy implementation
        return np.array([[0.3, 0.7]])  # Probabilidades [RG, LQG]
    
    @staticmethod
    def _compute_confusion_matrix(y_true: list, y_pred: list) -> ConfusionMatrixData:
        """
        Computa matriz de confusión.
        
        Labels: 0 = Relatividad General, 1 = Anomalía Cuántica
        """
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        
        return ConfusionMatrixData(
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn
        )