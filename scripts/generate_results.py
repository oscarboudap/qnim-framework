"""
generate_results.py  (v7 — version final con todos los fixes)
Sustituye scripts/generate_results.py
"""

import logging
import argparse, sys, time, os, json
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
    p.add_argument("--n-qubits",            type=int,   default=12,
                   help="Número de qubits del VQC; usar 27 para el perfil propuesto.")
    p.add_argument("--pca-components",      type=int,   default=12,
                   help="Número de componentes PCA que codifica el feature map.")
    p.add_argument("--n-pca-fit",           type=int,   default=64,
                   help="Número de componentes usadas para ajustar la PCA de origen.")
    p.add_argument("--feature-map-reps",    type=int,   default=2,
                   help="Repeticiones del feature map; usar 1 para el perfil 27q propuesto.")
    p.add_argument("--ansatz-reps",         type=int,   default=2)
    p.add_argument("--entanglement",        type=str,   default="linear",
                   choices=["linear", "circular", "full"])
    p.add_argument("--shots",               type=int,   default=1024)
    p.add_argument("--train-on-hardware",   action="store_true")
    p.add_argument("--hardware-shots",      type=int,   default=128)
    p.add_argument("--hardware-batch-size", type=int,   default=4)
    p.add_argument("--hardware-finetune-steps", type=int, default=20)
    p.add_argument("--backend",             type=str,   default="ibm_fez")
    p.add_argument("--reference-shots",     type=int,   default=4096)
    p.add_argument("--readout-hidden-size", type=int,   default=32)
    p.add_argument("--readout-refit-every", type=int,   default=10)
    p.add_argument("--use-zne",             action="store_true")
    p.add_argument("--n-feynman-params",    type=int,   default=4)
    p.add_argument("--profile",             type=str,   default=None,
                   help="Nombre de un perfil en config/experiment_profiles.json")

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
    p.add_argument("--n-test-per-class",   type=int,   default=100)
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

    profile_config = None
    if args.profile:
        profile_path = ROOT / "config" / "experiment_profiles.json"
        try:
            with profile_path.open("r", encoding="utf-8") as fh:
                profiles = json.load(fh)
            if args.profile in profiles:
                profile_config = profiles[args.profile]
                args.n_qubits = profile_config.get("n_qubits", args.n_qubits)
                args.pca_components = profile_config.get("pca_components", args.pca_components)
                args.n_pca_fit = profile_config.get("n_pca_fit", args.n_pca_fit)
                args.feature_map_reps = profile_config.get("feature_map_reps", args.feature_map_reps)
                args.ansatz_reps = profile_config.get("ansatz_reps", args.ansatz_reps)
                args.entanglement = profile_config.get("entanglement", args.entanglement)
                print(f"  Perfil cargado:         {args.profile} ({profile_config.get('description', '')})")
            else:
                print(f"  ⚠️  Perfil no encontrado: {args.profile}")
        except Exception as exc:
            print(f"  ⚠️  No se pudo cargar el perfil {args.profile}: {exc}")

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
    print(f"  n_qubits:               {args.n_qubits}")
    print(f"  feature_map_reps:       {args.feature_map_reps}")
    print(f"  ansatz_reps:            {args.ansatz_reps}")
    print(f"  entanglement:           {args.entanglement}")
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
            X_test: object = None
            y_test: object = None

        data_dir = Path(args.data_dir)
        print(f"Cargando dataset desde {data_dir}...")
        X_train = np.load(data_dir / "X_train.npy")
        y_train = np.load(data_dir / "y_train.npy")
        X_val   = np.load(data_dir / "X_val.npy")
        y_val   = np.load(data_dir / "y_val.npy")
        test_x_path = data_dir / "X_test.npy"
        test_y_path = data_dir / "y_test.npy"
        if test_x_path.exists() != test_y_path.exists():
            raise ValueError("El dataset debe incluir ambos archivos X_test.npy e y_test.npy")
        X_test = np.load(test_x_path) if test_x_path.exists() else None
        y_test = np.load(test_y_path) if test_y_path.exists() else None
        feature_arrays = [X_train, X_val] + ([X_test] if X_test is not None else [])
        if any(X.shape[1] < args.pca_components for X in feature_arrays):
            raise ValueError(
                f"El dataset debe tener al menos {args.pca_components} features"
            )
        X_train = X_train[:, :args.pca_components]
        X_val = X_val[:, :args.pca_components]
        if X_test is not None:
            X_test = X_test[:, :args.pca_components]
        n_classes = len(np.unique(y_train))
        if args.max_classes:
            mask_tr = y_train < args.max_classes
            mask_va = y_val   < args.max_classes
            X_train, y_train = X_train[mask_tr], y_train[mask_tr]
            X_val,   y_val   = X_val[mask_va],   y_val[mask_va]
            if X_test is not None:
                mask_test = y_test < args.max_classes
                X_test, y_test = X_test[mask_test], y_test[mask_test]
            n_classes = args.max_classes
        THEORY_CLASSES = [
            "GR","standard-siren","qnm-21","qnm-33","pn-deformation",
            "extra-dimensions","scalar-tensor","graviton-mass","chern-simons",
            "liv-alpha2","liv-alpha4","loop-quantum-gravity","gup",
        ]
        dataset = DiskDataset(
            X_train, y_train, X_val, y_val, n_classes, THEORY_CLASSES[:n_classes],
            X_test=X_test, y_test=y_test,
        )
        print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  n_classes: {n_classes}")

        # Baseline clasico rapido
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
        rf.fit(X_train, y_train)
        print(f"  RF val: {rf.score(X_val, y_val):.3f}  azar: {1/n_classes:.3f}")

        # Entrenar VQC directamente
        from src.infrastructure.qiskit_vqc_trainer import QiskitVQCTrainer
        from src.ibm_credentials import get_ibm_quantum_token

        token = get_ibm_quantum_token()
        if args.mode == "ibm" and not token:
            raise RuntimeError("IBM_QUANTUM_TOKEN no está configurado")
        trainer = QiskitVQCTrainer(
            mode=args.mode,
            token=token,
            backend_name=args.backend,
            train_on_hardware=args.train_on_hardware,
            hardware_shots=args.hardware_shots,
            hardware_batch_size=args.hardware_batch_size,
            hardware_finetune_steps=args.hardware_finetune_steps,
            feature_map_reps=args.feature_map_reps,
            ansatz_reps=args.ansatz_reps,
            entanglement=args.entanglement,
            patience=args.patience,
            learning_rate=args.learning_rate,
            readout_hidden_size=args.readout_hidden_size,
            readout_refit_every=args.readout_refit_every,
            training_batch_size=args.batch_size,
        )
        t0 = time.perf_counter()
        result = trainer.train_and_evaluate(
            dataset, n_qubits=args.n_qubits, shots=args.shots,
            n_features=args.pca_components,
            max_iterations=args.max_iter,
            use_real_hardware=args.mode == "ibm",
            hardware_finetune_steps=(
                args.hardware_finetune_steps if args.train_on_hardware else None
            ),
            use_zne=args.use_zne,
            n_feynman_params=args.n_feynman_params,
        )
        elapsed = time.perf_counter() - t0

        acc  = getattr(result, 'accuracy_sim', getattr(result, 'accuracy', 0.0))
        lh   = getattr(result, 'loss_history', [])
        loss = lh[-1] if lh else float('nan')
        ep   = getattr(result, 'n_epochs', '?')

        acc_ibm = None
        if args.mode == "ibm":
            acc_ibm = (
                result.accuracy_real_zne
                if args.use_zne
                else result.accuracy_real_no_zne
            )

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
            n_qubits=args.n_qubits,
            pca_components=args.pca_components,
            n_pca_fit=args.n_pca_fit,
            feature_map_reps=args.feature_map_reps,
            ansatz_reps=args.ansatz_reps,
            entanglement=args.entanglement,
            shots=args.shots,
            reference_shots=args.reference_shots,
            readout_hidden_size=args.readout_hidden_size,
            readout_refit_every=args.readout_refit_every,
            max_classes=args.max_classes,
            n_feynman_params=args.n_feynman_params,
            use_zne=args.use_zne,
            train_on_hardware=args.train_on_hardware,
            backend_name=args.backend,
            hardware_shots=args.hardware_shots,
            hardware_batch_size=args.hardware_batch_size,
            hardware_finetune_steps=args.hardware_finetune_steps,
            use_physics_generator=args.physics_generator,
            use_ligo_pca=args.ligo_pca,
            physics_approximant=args.approximant,
            n_per_class=args.n_per_class,
            n_val_per_class=args.n_val_per_class,
            n_test_per_class=args.n_test_per_class,
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