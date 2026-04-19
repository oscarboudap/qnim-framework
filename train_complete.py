"""
QNIM Training Loop Completo con SPSA
=====================================

Este script implementa el entrenamiento REAL del VQC usando optimizador SPSA.

Características:
- Carga dataset sintético masivo
- Divide en train/validation
- Entrena VQC con SPSA optimizer
- Visualiza curvas de aprendizaje
- Guarda pesos y métricas
"""

import numpy as np
import joblib
import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from src.infrastructure.storage.massive_loader import MassiveDatasetLoader
from src.infrastructure.qiskit_vqc_trainer import QiskitVQCTrainer


def plot_training_history(history: dict, output_path: str = "reports/training_history.png"):
    """
    Visualiza las curvas de entrenamiento (loss, accuracy).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    epochs = range(1, len(history["training_loss"]) + 1)
    axes[0].plot(epochs, history["training_loss"], 'b-o', label='Train Loss', linewidth=2)
    if history["validation_loss"]:
        axes[0].plot(epochs, history["validation_loss"], 'r-s', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (Binary Cross-Entropy)')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history["training_accuracy"], 'b-o', label='Train Accuracy', linewidth=2)
    if history["validation_accuracy"]:
        axes[1].plot(epochs, history["validation_accuracy"], 'r-s', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].set_ylim([0, 1.05])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Gráficos de entrenamiento guardados en: {output_path}")
    plt.close()


def save_training_metrics(history: dict, output_path: str = "reports/training_metrics.json"):
    """
    Guarda métricas de entrenamiento en JSON para análisis posterior.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Convertir arrays a listas para JSON
    metrics = {
        "training_loss": [float(x) for x in history["training_loss"]],
        "training_accuracy": [float(x) for x in history["training_accuracy"]],
        "validation_loss": [float(x) for x in history["validation_loss"]] if history["validation_loss"] else None,
        "validation_accuracy": [float(x) for x in history["validation_accuracy"]] if history["validation_accuracy"] else None,
        "best_epoch": history["best_epoch"],
        "final_training_loss": float(history["final_training_loss"]),
        "final_training_accuracy": float(history["final_training_accuracy"]),
        "execution_time_seconds": float(history["execution_time_seconds"])
    }
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"💾 Métricas de entrenamiento guardadas en: {output_path}")


def main():
    print("="*70)
    print("🧠 --- QNIM: TRAINING LOOP FASE 2 CON SPSA --- 🧠")
    print("="*70)
    print()
    
    # ========== CONFIGURACIÓN ==========
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    NUM_QUBITS = 12
    MAX_EPOCHS = 50
    BATCH_SIZE = 32
    EARLY_STOPPING_PATIENCE = 10
    LEARNING_RATE = 0.05
    VAL_SPLIT = 0.2
    
    weights_path = "models/qnim_vqc_weights.npy"
    pca_path = "models/qnim_preprocessing_pipeline.pkl"
    
    # ========== 1. CARGAR DATOS ==========
    print("📦 Paso 1: Cargando dataset sintético...")
    loader = MassiveDatasetLoader()
    X, y, pca_model = loader.load_and_preprocess(n_components=NUM_QUBITS)
    
    print(f"   ✅ {len(X)} eventos sintéticos cargados")
    print(f"   ✅ Dimensionalidad: {X.shape[1]} → {NUM_QUBITS} (PCA)")
    print(f"   ✅ Clases: {y.shape[1]} teorías diferentes")
    
    # Guardar pipeline PCA (vital para inferencia después)
    joblib.dump(pca_model, pca_path)
    print(f"   💾 PCA pipeline guardado en: {pca_path}")
    print()
    
    # ========== 2. TRAIN/VAL SPLIT ==========
    print("📊 Paso 2: Dividiendo datos en train/validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=42, stratify=np.argmax(y, axis=1)
    )
    
    print(f"   ✅ Train set: {len(X_train)} muestras")
    print(f"   ✅ Validation set: {len(X_val)} muestras")
    print(f"   ✅ Ratio: {100*VAL_SPLIT:.0f}% validation")
    print()
    
    # ========== 3. ENTRENAR VQC ==========
    print("🚀 Paso 3: Entrenando VQC con optimizador SPSA...")
    print("-"*70)
    
    trainer = QiskitVQCTrainer()
    
    training_result = trainer.train_vqc(
        X_train=X_train,
        y_train=y_train,
        num_qubits=NUM_QUBITS,
        max_iterations=MAX_EPOCHS,
        optimizer_name="SPSA",
        batch_size=BATCH_SIZE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        X_val=X_val,
        y_val=y_val,
        learning_rate=LEARNING_RATE,
        callback=None
    )
    
    print("-"*70)
    print()
    
    # ========== 4. GUARDAR PESOS ==========
    print("💾 Paso 4: Guardando pesos entrenados...")
    trainer.save_weights(training_result["weights"], weights_path)
    print()
    
    # ========== 5. VISUALIZAR MÉTRICAS ==========
    print("📈 Paso 5: Guardando histórico de entrenamiento...")
    
    history_dict = {
        "training_loss": training_result["training_loss_history"],
        "training_accuracy": training_result["training_accuracy_history"],
        "validation_loss": training_result["validation_loss_history"],
        "validation_accuracy": training_result["validation_accuracy_history"],
        "best_epoch": training_result["best_epoch"],
        "final_training_loss": training_result["final_training_loss"],
        "final_training_accuracy": training_result["final_training_accuracy"],
        "execution_time_seconds": training_result["execution_time_seconds"]
    }
    
    plot_training_history(history_dict)
    save_training_metrics(history_dict)
    print()
    
    # ========== 6. RESUMEN FINAL ==========
    print("="*70)
    print("✨ RESUMEN DEL ENTRENAMIENTO")
    print("="*70)
    print(f"{'Épocas completadas':<30}: {training_result['best_epoch']}")
    print(f"{'Loss final':<30}: {training_result['final_training_loss']:.4f}")
    print(f"{'Accuracy final':<30}: {training_result['final_training_accuracy']:.4f}")
    print(f"{'Tiempo de ejecución':<30}: {training_result['execution_time_seconds']:.1f}s")
    print(f"{'Pesos guardados en':<30}: {weights_path}")
    print(f"{'Métricas guardadas en':<30}: reports/training_metrics.json")
    print(f"{'Gráficos guardados en':<30}: reports/training_history.png")
    print("="*70)
    print()
    print("✅ VQC entrenado y listo para inferencia cuántica.")
    print("   Próximo paso: Ejecutar main.py para procesar eventos reales.")


if __name__ == "__main__":
    main()
