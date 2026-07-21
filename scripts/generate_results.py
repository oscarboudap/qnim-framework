"""
generate_results.py  (v7 — version final con todos los fixes)
Sustituye scripts/generate_results.py
"""

import logging
import argparse, sys, time, os
from pathlib import Path
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.application.use_cases.generate_experiment_results_use_case import (
    GenerateExperimentResultsUseCase, ExperimentConfig,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="QNIM Framework — Resultados Experimentales",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["sim", "ibm"], default="sim")

    # VQC
    p.add_argument("--max-iter",            type=int,   default=60)
    p.add_argument("--batch-size",          type=int,   default=512,
                   help="Batch QNSPSA. Usar >=512 con datasets grandes.")
    p.add_argument("--patience",            type=int,   default=20)
    p.add_argument("--learning-rate",       type=float, default=0.03)
    p.add_argument("--ansatz-reps",         type=int,   default=2)
    p.add_argument("--shots",               type=int,   default=1024)
    p.add_argument("--reference-shots",     type=int,   default=4096)
    p.add_argument("--readout-hidden-size", type=int,   default=32)
    p.add_argument("--readout-refit-every", type=int,   default=10)
    p.add_argument("--use-zne",             action="store_true")
    p.add_argument("--n-feynman-params",    type=int,   default=4)

    # Generador
    p.add_argument("--ligo-pca",           action="store_true",
                   help="Pipeline del paper: blanqueo PSD O3 + PCA + ZZFeatureMap")
    p.add_argument("--physics-generator",  action="store_true",
                   help="PhysicsSSTGAdapter (PyCBC sin PCA)")
    p.add_argument("--approximant",        type=str,   default="IMRPhenomPv2",
                   help="Aproximante PyCBC. IMRPhenomPv2 es 300x mas rapido que IMRPhenomD.")
    p.add_argument("--n-per-class",        type=int,   default=4200,
                   help="Eventos train/clase. 4200 -> 54600 total (como el paper).")
    p.add_argument("--n-val-per-class",    type=int,   default=600)
    p.add_argument("--max-classes",        type=int,   default=None)

    # Dataset pregenerado en disco
    p.add_argument("--from-disk",          action="store_true",
                   help="Cargar dataset desde data/dataset_full/ (ya generado).")
    p.add_argument("--data-dir",           type=str,   default="data/dataset_full")

    return p.parse_args()


def main():
    args = parse_args()

    if args.ligo_pca and args.physics_generator:
        print("ERROR: --ligo-pca y --physics-generator son mutuamente excluyentes.")
        sys.exit(1)

    gen_str = ("LIGOPCAAdapter + ZZFeatureMap (paper)" if args.ligo_pca else
               "PhysicsSSTGAdapter (PyCBC)" if args.physics_generator else
               "SSTGAdapter (marcador artificial)")

    print("=" * 70)
    print("  QNIM Framework — Resultados Experimentales")
    print("=" * 70)
    print(f"  Modo:                   {args.mode.upper()}")
    print(f"  Generador:              {gen_str}")
    if args.ligo_pca or args.physics_generator:
        print(f"  Aproximante:            {args.approximant}")
        if args.from_disk:
            print(f"  Dataset:                desde disco ({args.data_dir})")
        else:
            print(f"  Eventos train/clase:    {args.n_per_class}")
            print(f"  Eventos val/clase:      {args.n_val_per_class}")
    print(f"  max_iter:               {args.max_iter}")
    print(f"  batch_size:             {args.batch_size}")
    print(f"  patience:               {args.patience}")
    print(f"  learning_rate:          {args.learning_rate}")
    print(f"  readout_hidden_size:    {args.readout_hidden_size}")
    print(f"  readout_refit_every:    {args.readout_refit_every}")
    print(f"  shots:                  {args.shots}")
    print(f"  use_zne:                {args.use_zne}")
    print(f"  max_classes:            {args.max_classes or 'todas (13)'}")
    print("=" * 70)

    # ---- Cargar dataset ----
    if args.from_disk:
        import numpy as np
        from dataclasses import dataclass
        from typing import List

        @dataclass
        class DiskDataset:
            X_train: np.ndarray
            y_train: np.ndarray
            X_val:   np.ndarray
            y_val:   np.ndarray
            n_classes: int
            theory_names: List[str]

        data_dir = Path(args.data_dir)
        print(f"Cargando dataset desde {data_dir}...")
        X_train = np.load(data_dir / "X_train.npy")
        y_train = np.load(data_dir / "y_train.npy")
        X_val   = np.load(data_dir / "X_val.npy")
        y_val   = np.load(data_dir / "y_val.npy")
        n_classes = len(np.unique(y_train))
        if args.max_classes:
            mask_tr = y_train < args.max_classes
            mask_va = y_val   < args.max_classes
            X_train, y_train = X_train[mask_tr], y_train[mask_tr]
            X_val,   y_val   = X_val[mask_va],   y_val[mask_va]
            n_classes = args.max_classes
        THEORY_CLASSES = [
            "GR","standard-siren","qnm-21","qnm-33","pn-deformation",
            "extra-dimensions","scalar-tensor","graviton-mass","chern-simons",
            "liv-alpha2","liv-alpha4","loop-quantum-gravity","gup",
        ]
        dataset = DiskDataset(X_train, y_train, X_val, y_val,
                              n_classes, THEORY_CLASSES[:n_classes])
        print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  n_classes: {n_classes}")

        # Baseline clasico rapido
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
        rf.fit(X_train, y_train)
        print(f"  RF val: {rf.score(X_val, y_val):.3f}  azar: {1/n_classes:.3f}")

        # Entrenar VQC directamente
        from src.infrastructure.qiskit_vqc_trainer import QiskitVQCTrainer
        trainer = QiskitVQCTrainer(
            mode=args.mode,
            ansatz_reps=args.ansatz_reps,
            patience=args.patience,
            learning_rate=args.learning_rate,
            readout_hidden_size=args.readout_hidden_size,
            readout_refit_every=args.readout_refit_every,
            training_batch_size=args.batch_size,
        )
        t0 = time.perf_counter()
        result = trainer.train_and_evaluate(
            dataset, n_qubits=12, shots=args.shots,
            max_iterations=args.max_iter,
            use_zne=args.use_zne,
            n_feynman_params=args.n_feynman_params,
        )
        elapsed = time.perf_counter() - t0

        acc  = getattr(result, 'accuracy_sim', getattr(result, 'accuracy', 0.0))
        lh   = getattr(result, 'loss_history', [])
        loss = lh[-1] if lh else float('nan')
        ep   = getattr(result, 'n_epochs', '?')

        # IBM si procede
        acc_ibm = None
        if args.mode == "ibm":
            token = os.environ.get("IBM_QUANTUM_TOKEN", "")
            if token:
                try:
                    final_w = getattr(result, 'final_weights', None)
                    if final_w is not None:
                        _, acc_ibm = trainer._validate_on_ibm(
                            weights=final_w,
                            dataset=dataset,
                            n_qubits=12,
                            use_zne=args.use_zne,
                        )
                    else:
                        print("  ⚠️  No hay pesos finales disponibles para validar en IBM.")
                except Exception as e:
                    print(f"  ⚠️  IBM falló: {e}")
            else:
                print("  ⚠️  IBM_QUANTUM_TOKEN no exportado.")
                print("       Ejecuta: export IBM_QUANTUM_TOKEN='tu_token'")

        print("\n" + "=" * 70)
        print("  RESULTADOS")
        print("=" * 70)
        print(f"  Accuracy simulador (val):  {acc:.3f}")
        if acc_ibm is not None:
            print(f"  Accuracy IBM (val):        {acc_ibm:.3f}")
        print(f"  RF baseline (val):         {rf.score(X_val, y_val):.3f}")
        print(f"  Loss final:                {loss:.4f}  (ln({n_classes})={__import__('math').log(n_classes):.4f})")
        print(f"  Épocas:                    {ep}")
        print(f"  Tiempo:                    {elapsed:.1f}s")
        print("=" * 70)

    else:
        # Flujo normal via ExperimentConfig
        config = ExperimentConfig(
            mode=args.mode,
            max_iterations=args.max_iter,
            batch_size=args.batch_size,
            patience=args.patience,
            learning_rate=args.learning_rate,
            ansatz_reps=args.ansatz_reps,
            shots=args.shots,
            reference_shots=args.reference_shots,
            readout_hidden_size=args.readout_hidden_size,
            readout_refit_every=args.readout_refit_every,
            max_classes=args.max_classes,
            n_feynman_params=args.n_feynman_params,
            use_zne=args.use_zne,
            use_physics_generator=args.physics_generator,
            use_ligo_pca=args.ligo_pca,
            physics_approximant=args.approximant,
            n_per_class=args.n_per_class,
            n_val_per_class=args.n_val_per_class,
        )

        t0     = time.perf_counter()
        result = GenerateExperimentResultsUseCase(config).execute()
        elapsed = time.perf_counter() - t0

        print("\n" + "=" * 70)
        print("  RESULTADOS")
        print("=" * 70)
        print(f"  Accuracy simulador (val):  {result.accuracy_sim:.3f}")
        if result.accuracy_ibm is not None:
            print(f"  Accuracy IBM (val):        {result.accuracy_ibm:.3f}")
        print(f"  Loss final:                {result.final_loss:.4f}")
        print(f"  Épocas entrenadas:         {result.n_epochs}")
        print(f"  Tiempo total:              {elapsed:.1f} s")
        print("=" * 70)


if __name__ == "__main__":
    main()