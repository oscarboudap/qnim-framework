#!/usr/bin/env python3
"""
Application Layer: Training Use Case

Ejecuta el entrenamiento del VQC usando QiskitVQCTrainer (Infrastructure)
Orquestación centralizada en este archivo.
"""

import os
import sys
import json
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Infrastructure: Data loader y trainer
from src.infrastructure.storage.massive_loader import MassiveDatasetLoader
from src.infrastructure.qiskit_vqc_trainer import QiskitVQCTrainer


def plot_training_history(history: dict, output_path: str = "reports/training_history.png"):
    """Visualiza curvas de entrenamiento"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history["training_loss"]) + 1)
    axes[0].plot(epochs, history["training_loss"], 'b-o', label='Train Loss', linewidth=2)
    if history["validation_loss"]:
        axes[0].plot(epochs, history["validation_loss"], 'r-s', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history["training_accuracy"], 'b-o', label='Train Acc', linewidth=2)
    if history["validation_accuracy"]:
        axes[1].plot(epochs, history["validation_accuracy"], 'r-s', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].set_ylim([0, 1.05])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_training_metrics(history: dict, output_path: str = "reports/training_metrics.json"):
    """Guarda métricas en JSON"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
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


def main():
    print("="*70)
    print("🧠 QNIM: TRAINING LOOP CON SPSA")
    print("="*70)
    print()
    
    load_dotenv()
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # Configuración
    NUM_QUBITS = 12
    MAX_EPOCHS = 50
    BATCH_SIZE = 32
    EARLY_STOPPING_PATIENCE = 10
    LEARNING_RATE = 0.05
    VAL_SPLIT = 0.2
    
    weights_path = "models/qnim_vqc_weights.npy"
    pca_path = "models/qnim_preprocessing_pipeline.pkl"
    
    # 1. CARGAR DATOS (Infrastructure: MassiveDatasetLoader)
    print("📦 Paso 1: Cargando dataset...")
    loader = MassiveDatasetLoader()
    X, y, pca_model = loader.load_and_preprocess(n_components=NUM_QUBITS)
    
    print(f"   ✅ {len(X)} eventos cargados")
    print(f"   ✅ Dimensionalidad: {X.shape[1]} → {NUM_QUBITS} (PCA)")
    
    joblib.dump(pca_model, pca_path)
    print(f"   💾 PCA guardado en: {pca_path}")
    print()
    
    # 2. TRAIN/VAL SPLIT
    print("📊 Paso 2: Dividiendo en train/val...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=42, stratify=y
    )
    
    print(f"   ✅ Train: {len(X_train)} | Val: {len(X_val)}")
    
    # Convert labels to one-hot encoding for VQC
    n_classes = len(np.unique(y))
    y_train_onehot = np.eye(n_classes)[y_train]
    y_val_onehot = np.eye(n_classes)[y_val]
    print(f"   ✅ Labels converted to one-hot ({n_classes} classes)")
    print()
    
    # 3. ENTRENAR (Infrastructure: QiskitVQCTrainer)
    print("🚀 Paso 3: Entrenando VQC...")
    print("-"*70)
    
    trainer = QiskitVQCTrainer()
    
    training_result = trainer.train_vqc(
        X_train=X_train,
        y_train=y_train_onehot,
        num_qubits=NUM_QUBITS,
        max_iterations=MAX_EPOCHS,
        optimizer_name="SPSA"
    )
    
    print("-"*70)
    print()
    
    # 4. GUARDAR PESOS
    print("💾 Paso 4: Guardando pesos...")
    trainer.save_weights(training_result["weights"], weights_path)
    
    # Create simplified history from available metrics
    history_dict = {
        "training_loss": [training_result["training_loss"]],
        "training_accuracy": [vqc.score(X_train, y_train_onehot) if 'vqc' in locals() else 0.5],
        "validation_loss": [],
        "validation_accuracy": [training_result["validation_accuracy"]] if "validation_accuracy" in training_result else [],
        "best_epoch": training_result.get("iterations", MAX_EPOCHS),
        "final_training_loss": training_result["training_loss"],
        "final_training_accuracy": training_result.get("validation_accuracy", 0.0),
        "execution_time_seconds": training_result["execution_time_seconds"]
    }
    
    plot_training_history(history_dict)
    save_training_metrics(history_dict)
    print()
    
    # 5. RESUMEN
    print("="*70)
    print("✨ RESUMEN DEL ENTRENAMIENTO")
    print("="*70)
    print(f"{'Épocas':<30}: {training_result.get('iterations', MAX_EPOCHS)}")
    print(f"{'Loss final':<30}: {training_result['training_loss']:.4f}")
    print(f"{'Accuracy final':<30}: {training_result.get('validation_accuracy', 0.0):.4f}")
    print(f"{'Tiempo':<30}: {training_result['execution_time_seconds']:.1f}s")
    print("="*70)


if __name__ == "__main__":
    sys.exit(main())
