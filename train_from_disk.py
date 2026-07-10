"""
train_from_disk.py
==================
Entrena el VQC sobre el dataset pregenerado en data/dataset_full/
sin tener que regenerar los datos cada vez.

USO:
    python train_from_disk.py [--mode sim|ibm] [--max-iter 200]
"""

import argparse, sys, time, pickle, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class DiskDataset:
    """Dataset cargado desde disco, compatible con QiskitVQCTrainer."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val:   np.ndarray
    y_val:   np.ndarray
    n_classes: int
    theory_names: List[str]


THEORY_CLASSES = [
    "GR", "standard-siren", "qnm-21", "qnm-33", "pn-deformation",
    "extra-dimensions", "scalar-tensor", "graviton-mass", "chern-simons",
    "liv-alpha2", "liv-alpha4", "loop-quantum-gravity", "gup",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",              choices=["sim","ibm"], default="sim")
    p.add_argument("--data-dir",          type=str, default="data/dataset_full")
    p.add_argument("--max-iter",          type=int, default=200)
    p.add_argument("--patience",          type=int, default=40)
    p.add_argument("--learning-rate",     type=float, default=0.01)
    p.add_argument("--readout-hidden-size", type=int, default=32)
    p.add_argument("--readout-refit-every", type=int, default=10)
    p.add_argument("--batch-size",        type=int, default=512,
                   help="Batch size QNSPSA. Usar >=512 con datasets grandes.")
    p.add_argument("--shots",             type=int, default=1024)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "X_train.npy").exists():
        print(f"ERROR: No se encuentra el dataset en {data_dir}")
        print("Ejecuta primero: python generate_dataset_chunked.py")
        sys.exit(1)

    # ---- Cargar dataset ----
    print(f"Cargando dataset desde {data_dir}...")
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val   = np.load(data_dir / "X_val.npy")
    y_val   = np.load(data_dir / "y_val.npy")
    n_classes = len(np.unique(y_train))

    print(f"  X_train: {X_train.shape}  n_classes: {n_classes}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  Rango features: [{X_train.min():.3f}, {X_train.max():.3f}]")

    dataset = DiskDataset(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        n_classes=n_classes,
        theory_names=THEORY_CLASSES[:n_classes],
    )

    # ---- Clasificador clásico de referencia ----
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    rf.fit(X_train, y_train)
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train, y_train)
    print(f"\n  Baselines clásicos:")
    print(f"  LogReg val:  {lr.score(X_val, y_val):.3f}")
    print(f"  RF     val:  {rf.score(X_val, y_val):.3f}")
    print(f"  Azar:        {1/n_classes:.3f}")

    # ---- Entrenar VQC ----
    print(f"\n{'='*60}")
    print(f"  Entrenando VQC ({args.mode.upper()})")
    print(f"{'='*60}")

    from src.infrastructure.qiskit_vqc_trainer import QiskitVQCTrainer
    trainer = QiskitVQCTrainer(
        mode=args.mode,
        ansatz_reps=2,
        patience=args.patience,
        learning_rate=args.learning_rate,
        readout_hidden_size=args.readout_hidden_size,
        readout_refit_every=args.readout_refit_every,
        training_batch_size=args.batch_size,
    )

    t0 = time.perf_counter()
    result = trainer.train_and_evaluate(
        dataset, n_qubits=12,
        shots=args.shots,
        max_iterations=args.max_iter,
    )
    elapsed = time.perf_counter() - t0

    acc  = getattr(result, 'accuracy_sim', getattr(result, 'accuracy', 0.0))
    lh   = getattr(result, 'loss_history', [])
    loss = lh[-1] if lh else float('nan')
    ep   = getattr(result, 'n_epochs', '?')

    print(f"\n{'='*60}")
    print(f"  RESULTADOS")
    print(f"{'='*60}")
    print(f"  VQC accuracy (val):  {acc:.3f}")
    print(f"  RF  accuracy (val):  {rf.score(X_val, y_val):.3f}")
    print(f"  Loss final:          {loss:.4f}  (ln(13)=2.5649)")
    print(f"  Épocas:              {ep}")
    print(f"  Tiempo:              {elapsed/3600:.2f}h")
    print(f"{'='*60}")

    # Guardar pesos
    if hasattr(trainer, '_theta'):
        np.save(data_dir / "vqc_weights.npy", trainer._theta)
        print(f"  Pesos guardados en {data_dir}/vqc_weights.npy")


if __name__ == "__main__":
    main()