import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class ValidationService:
    """
    Caso de Uso: Auditar la precisión del Modelo Cuántico entrenado.
    """
    def __init__(self, orchestrator_port):
        self.orchestrator = orchestrator_port # Usamos el orquestador híbrido para inferir

    def run_confusion_assessment(self, test_X: np.ndarray, test_y: np.ndarray) -> str:
        print("🧪 Iniciando Evaluación de Matriz de Confusión Cuántica...")
        
        y_true = []
        y_pred = []
        
        for i in range(len(test_X)):
            # Extraemos la predicción real
            true_idx = np.argmax(test_y[i])
            y_true.append(true_idx)
            
            # Pasamos las características por la Rama de IBM del Orquestador
            prediction = self.orchestrator.execute_ibm_branch(test_X[i:i+1])
            # Si detecta Anomalía (LQG) devuelve 1, si es RG devuelve 0
            pred_idx = 1 if "LQG" in prediction['detected_theory'].value else 0
            y_pred.append(pred_idx)

        # Generar Gráfica
        self._plot_confusion_matrix(y_true, y_pred)
        
        # Generar Reporte de Texto
        target_names = ['Kerr (RG)', 'Anomalía (LQG/Fuzzball)']
        return classification_report(y_true, y_pred, target_names=target_names)

    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['RG Predicha', 'Anomalía Predicha'],
                    yticklabels=['RG Real', 'Anomalía Real'])
        plt.title('Matriz de Confusión: Clasificador QNIM')
        plt.tight_layout()
        plt.savefig('reports/figures/confusion_matrix.png')
        plt.close()