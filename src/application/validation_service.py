import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class ValidationService:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def run_confusion_assessment(self, test_X, test_y):
        """
        Evalúa el modelo entrenado contra un set de validación
        y genera la Matriz de Confusión.
        """
        print("\n🧪 Iniciando Evaluación de Matriz de Confusión Cuántica...")
        y_true = np.argmax(test_y, axis=1)
        y_pred = []

        for i, features in enumerate(test_X):
            print(f"\r🔍 Validando muestra {i+1}/{len(test_X)}...", end="")
            # Forzamos inferencia local para rapidez en la validación masiva
            pred = self.orchestrator.vqc_engine.vqc.predict(np.array([features]))
            y_pred.append(np.argmax(pred[0]))

        print("\n✅ Evaluación completada.")
        
        cm = confusion_matrix(y_true, y_pred)
        self._plot_cm(cm)
        return classification_report(y_true, y_pred, target_names=["Kerr (RG)", "Anomalía (LQG)"])

    def _plot_cm(self, cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=["RG", "LQG"], yticklabels=["RG", "LQG"])
        plt.title("Matriz de Confusión: Clasificador QNIM (12 Cúbits)")
        plt.ylabel("Teoría Real (Inyectada)")
        plt.xlabel("Teoría Predicha (QPU)")
        plt.savefig("reports/figures/confusion_matrix.png")
        print("💾 Gráfica guardada en reports/figures/confusion_matrix.png")