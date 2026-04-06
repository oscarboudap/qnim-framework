from src.domain.quantum.entities import InferenceResult

class CLIPresenter:
    """Clase para formatear la salida en la terminal."""
    
    @staticmethod
    def show_welcome():
        print("="*50)
        print("🚀 QNIM: Quantum Native Inverse Models for GW")
        print("      Framework de Inferencia Cuántica v1.0")
        print("="*50)

    @staticmethod
    def show_result(result: InferenceResult):
        print("\n" + "-"*30)
        print("📊 RESULTADO DE LA INFERENCIA CUÁNTICA")
        print("-"*30)
        print(f"Clase Predicha: {result.predicted_class} (Relatividad General)")
        print(f"Confianza: {result.probabilities[result.predicted_class]*100:.2f}%")
        print(f"Profundidad del Circuito: {result.metadata.get('circuit_depth')}")
        print(f"Motor: {result.metadata.get('method')}")
        print("-"*30 + "\n")