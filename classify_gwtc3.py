"""
classify_gwtc3.py
=================
Script completo: entrena el VQC con datos sintéticos y clasifica
los 90 eventos reales de GWTC-3 descargados desde GWOSC.

Ejecutar desde la raíz del proyecto:
    python classify_gwtc3.py [--n-per-class N] [--max-events N] [--mode sim|ibm]

Tiempo estimado:
    - Generación sintética (200/clase): ~30 min
    - Entrenamiento VQC: ~2-3h
    - Descarga + clasificación 90 eventos: ~30-60 min
      (los strains se cachean en data/gwosc_cache/ para reruns)
"""

import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-class",  type=int, default=200,
                   help="Eventos sintéticos de entrenamiento por clase")
    p.add_argument("--max-events",   type=int, default=90,
                   help="Máximo de eventos GWTC-3 a clasificar")
    p.add_argument("--mode",         choices=["sim", "ibm"], default="sim")
    p.add_argument("--max-iter",     type=int, default=80)
    p.add_argument("--patience",     type=int, default=25)
    p.add_argument("--skip-train",   action="store_true",
                   help="Saltar entrenamiento (carga pesos de data/vqc_weights.npy)")
    args = p.parse_args()

    # ----------------------------------------------------------------
    # 1. Generar dataset sintético con LIGOPCAAdapter
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("  PASO 1: Generando dataset sintético (PyCBC + PCA O3)")
    print("="*60)
    from src.infrastructure.ligo_pca_adapter import LIGOPCAAdapter
    adapter = LIGOPCAAdapter(approximant="IMRPhenomD")
    dataset = adapter.generate_balanced_dataset(
        n_per_class=args.n_per_class,
        n_val_per_class=max(20, args.n_per_class // 4),
        seed=42, verbose=True,
    )
    print(f"  Techo clásico RF (estimado): ver clasificador clásico abajo")

    # Clasificador clásico de referencia
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=200, random_state=0)
    rf.fit(dataset.X_train, dataset.y_train)
    rf_acc = rf.score(dataset.X_val, dataset.y_val)
    print(f"  RF val accuracy: {rf_acc:.3f}  (azar={1/13:.3f})")

    # ----------------------------------------------------------------
    # 2. Entrenar VQC
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("  PASO 2: Entrenando VQC")
    print("="*60)
    from src.infrastructure.qiskit_vqc_trainer import QiskitVQCTrainer

    trainer = QiskitVQCTrainer(
        mode=args.mode,
        ansatz_reps=2,
        patience=args.patience,
        learning_rate=0.03,
        readout_hidden_size=32,
        readout_refit_every=10,
    )

    weights_path = Path("data/vqc_weights.npy")
    if args.skip_train and weights_path.exists():
        print("  Cargando pesos desde", weights_path)
        trainer._theta = np.load(weights_path)
    else:
        t0 = time.perf_counter()
        result = trainer.train_and_evaluate(
            dataset, n_qubits=12, shots=1024,
            max_iterations=args.max_iter,
        )
        elapsed = time.perf_counter() - t0
        acc = getattr(result, "accuracy_sim", getattr(result, "accuracy", 0.0))
        print(f"  VQC val accuracy: {acc:.3f}")
        lh = getattr(result, "loss_history", [])
        loss = lh[-1] if lh else float("nan")
        print(f"  Loss final:       {loss:.4f}")
        print(f"  Épocas:           {getattr(result, 'n_epochs', '?')}")
        print(f"  Tiempo:           {elapsed:.0f}s")

        # Guardar pesos para reruns
        weights_path.parent.mkdir(exist_ok=True)
        if hasattr(trainer, '_theta'):
            np.save(weights_path, trainer._theta)
            print(f"  Pesos guardados en {weights_path}")

    # ----------------------------------------------------------------
    # 3. Clasificar eventos reales de GWTC-3
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("  PASO 3: Clasificando eventos reales de GWTC-3")
    print("  (descargando strains desde GWOSC — se cachean localmente)")
    print("="*60)

    from gwosc_real_events import GWOSCClassifier
    clf = GWOSCClassifier(
        trainer=trainer,
        adapter=adapter,
        cache_dir="data/gwosc_cache",
    )

    results = clf.classify_gwtc3(max_events=args.max_events, verbose=True)

    # ----------------------------------------------------------------
    # 4. Tabla resumen
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("  RESULTADOS — Clasificación GWTC-3")
    print("="*60)
    print(clf.summary_table(results))

    # Guardar resultados
    out_path = Path("reports/gwtc3_classification.txt")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        f.write(clf.summary_table(results))
    print(f"\n  Tabla guardada en {out_path}")


if __name__ == "__main__":
    main()