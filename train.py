import os
from pathlib import Path
from src.application.model_training_service import ModelTrainingService
from src.infrastructure.storage.quantum_dataloader import QuantumDatasetLoader

def main():
    print("🚀 Preparando Inyección de Dependencias para Entrenamiento...")
    project_root = Path(__file__).resolve().parent
    
    # 1. Instanciar el Puerto de Lectura (Infraestructura)
    loader = QuantumDatasetLoader(target_samples=16384)
    
    # 2. Instanciar el Servicio de Aplicación
    models_dir = project_root / "models"
    trainer = ModelTrainingService(dataloader_port=loader, models_dir=str(models_dir))
    
    # 3. Localizar dataset y ejecutar
    data_dir = project_root / "data" / "synthetic"
    try:
        latest_batch = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("2026")])[-1]
        print(f"📂 Usando dataset curado: {latest_batch.name}")
        trainer.execute_training(dataset_path=latest_batch, max_events=200)
    except IndexError:
        print("❌ No se encontró ningún dataset. Genera los datos sintéticos primero.")

if __name__ == "__main__":
    main()