"""Run a controlled qubit/PCA scaling study on one fixed dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.application.scaling_metrics import (
    RANDOM_BASELINE,
    holm_bonferroni,
    paired_bootstrap_difference,
    permutation_test,
    stratified_bootstrap,
)

SCALING_MATRIX = (
    {"id": "n12_d12", "n_qubits": 12, "pca_components": 12},
    {"id": "n16_d12", "n_qubits": 16, "pca_components": 12},
    {"id": "n16_d16", "n_qubits": 16, "pca_components": 16},
    {"id": "n20_d12", "n_qubits": 20, "pca_components": 12},
    {"id": "n20_d20", "n_qubits": 20, "pca_components": 20},
    {"id": "n32_d12", "n_qubits": 32, "pca_components": 12},
    {"id": "n32_d32", "n_qubits": 32, "pca_components": 32},
    {"id": "n48_d48", "n_qubits": 48, "pca_components": 48},
    {"id": "n64_d64", "n_qubits": 64, "pca_components": 64},
    {"id": "n96_d96", "n_qubits": 96, "pca_components": 96},
    {"id": "n128_d128", "n_qubits": 128, "pca_components": 128},
)


@dataclass
class ScalingDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_classes: int


class RunScalingStudyUseCase:
    """Train each configuration on common train/validation data and test once."""

    def __init__(
        self,
        dataset_dir: str | Path,
        output_dir: str | Path = "results/scaling_study",
        max_iterations: int = 60,
        shots: int = 512,
        seed: int = 42,
        metrics_seed: int = 4242,
        bootstrap_iterations: int = 10000,
        permutation_iterations: int = 10000,
        experiments: tuple[dict, ...] = SCALING_MATRIX,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.max_iterations = max_iterations
        self.shots = shots
        self.seed = seed
        self.metrics_seed = metrics_seed
        self.bootstrap_iterations = bootstrap_iterations
        self.permutation_iterations = permutation_iterations
        self.experiments = experiments

    def _load_dataset(self) -> tuple[ScalingDataset, str]:
        arrays = {}
        for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
            path = self.dataset_dir / f"{name}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Falta el conjunto independiente requerido: {path}")
            arrays[name] = np.load(path, allow_pickle=False)

        for split in ("train", "val", "test"):
            X, y = arrays[f"X_{split}"], arrays[f"y_{split}"]
            if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
                raise ValueError(f"El split {split} debe tener X 2D y y 1D alineados")

        feature_counts = {arrays[f"X_{split}"].shape[1] for split in ("train", "val", "test")}
        if len(feature_counts) != 1:
            raise ValueError("Train, validation y test deben tener igual número de features")

        classes = np.unique(arrays["y_train"])
        if not np.array_equal(classes, np.arange(len(classes))):
            raise ValueError("Las etiquetas deben ser consecutivas desde cero")
        for split in ("val", "test"):
            if not np.array_equal(np.unique(arrays[f"y_{split}"]), classes):
                raise ValueError(f"El split {split} no contiene las mismas clases que train")

        test_counts = np.bincount(arrays["y_test"], minlength=len(classes))
        if len(np.unique(test_counts)) != 1:
            raise ValueError("El conjunto de test debe estar estratificado con igual conteo por clase")

        required_features = max(item["pca_components"] for item in self.experiments)
        if next(iter(feature_counts)) < required_features:
            raise ValueError(
                f"El dataset requiere al menos {required_features} features para la matriz de escalado"
            )

        digest = hashlib.sha256()
        for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
            digest.update(name.encode("ascii"))
            digest.update(np.ascontiguousarray(arrays[name]).tobytes())

        dataset = ScalingDataset(
            **arrays,
            n_classes=len(classes),
        )
        return dataset, digest.hexdigest()

    def _build_trainer(self, experiment: dict):
        from src.infrastructure.qiskit_vqc_trainer import QiskitVQCTrainer

        return QiskitVQCTrainer(
            mode="sim",
            feature_map_reps=2,
            ansatz_reps=2,
            entanglement="linear",
            patience=20,
            learning_rate=0.03,
            training_batch_size=64,
            random_seed=self.seed,
        )

    @staticmethod
    def _commit_hash() -> str | None:
        root = Path(__file__).resolve().parents[3]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    def execute(self) -> dict:
        dataset, dataset_hash = self._load_dataset()
        commit_hash = self._commit_hash()
        records = []

        for experiment in self.experiments:
            n_qubits = experiment["n_qubits"]
            n_features = experiment["pca_components"]
            if not 1 <= n_features <= n_qubits <= 128:
                raise ValueError(f"Dimensiones inválidas en experimento {experiment['id']}")

            current_dataset = SimpleNamespace(
                X_train=dataset.X_train[:, :n_features],
                y_train=dataset.y_train,
                X_val=dataset.X_val[:, :n_features],
                y_val=dataset.y_val,
                n_classes=dataset.n_classes,
            )
            trainer = self._build_trainer(experiment)
            training_result = trainer.train_and_evaluate(
                current_dataset,
                n_qubits=n_qubits,
                n_features=n_features,
                shots=self.shots,
                max_iterations=self.max_iterations,
                use_real_hardware=False,
            )
            accuracy, confusion_matrix, predictions = trainer._evaluate_real_predictions(
                weights=training_result.final_weights,
                X_train=current_dataset.X_train,
                X_val=dataset.X_test[:, :n_features],
                y_val=dataset.y_test,
                n_qubits=n_qubits,
                n_features=n_features,
                n_classes=dataset.n_classes,
                shots=self.shots,
            )
            predictions = np.asarray(predictions, dtype=int)
            bootstrap = stratified_bootstrap(
                dataset.y_test, predictions, self.bootstrap_iterations, self.metrics_seed
            )
            permutation = permutation_test(
                dataset.y_test, predictions, self.permutation_iterations, self.metrics_seed
            )
            records.append({
                **experiment,
                "seed": self.seed,
                "metrics_seed": self.metrics_seed,
                "mode": "sim",
                "backend": "local_simulator",
                "shots": self.shots,
                "job_ids": [],
                "dataset_sha256": dataset_hash,
                "git_commit": commit_hash,
                "accuracy_sim": float(training_result.accuracy_sim),
                "balanced_accuracy": float(accuracy),
                "bootstrap_ci": bootstrap,
                "permutation_test": permutation,
                "confusion_matrix": confusion_matrix,
                "y_true": dataset.y_test.astype(int).tolist(),
                "y_pred": predictions.tolist(),
                "paired_difference_vs_reference": None,
            })

        adjusted = holm_bonferroni([
            record["permutation_test"]["p_value"] for record in records
        ])
        for record, adjusted_p in zip(records, adjusted):
            record["permutation_test"]["p_value_holm"] = float(adjusted_p)

        reference = next(
            (record for record in records if record["n_qubits"] == 12 and record["pca_components"] == 12),
            None,
        )
        if reference is None:
            raise ValueError("La matriz debe incluir la referencia de 12 qubits / 12 PCA")
        random_baseline = (
            RANDOM_BASELINE if dataset.n_classes == 13 else 1.0 / dataset.n_classes
        )

        for record in records:
            if record is not reference:
                record["paired_difference_vs_reference"] = paired_bootstrap_difference(
                    record["y_true"],
                    reference["y_pred"],
                    record["y_pred"],
                    self.bootstrap_iterations,
                    self.metrics_seed,
                )
            difference = record["paired_difference_vs_reference"]
            record["criteria"] = {
                "lower_ci_above_random": (
                    record["bootstrap_ci"]["one_sided_lower_95"]
                    > random_baseline
                ),
                "holm_p_below_0_05": record["permutation_test"]["p_value_holm"] < 0.05,
                "paired_difference_lower_above_zero": (
                    difference["lower_95"] > 0 if difference is not None else None
                ),
            }
            run_id = f"{record['id']}_seed{self.seed}"
            self._write_json(
                self.output_dir / "experiments" / run_id / "result.json",
                record,
            )

        summary_dir = self.output_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "dataset_sha256": dataset_hash,
            "git_commit": commit_hash,
            "n_classes": dataset.n_classes,
            "random_baseline": random_baseline,
            "seed": self.seed,
            "metrics_seed": self.metrics_seed,
            "records": records,
        }
        self._write_json(summary_dir / "scaling_metrics.json", summary)
        csv_fields = [
            "id", "n_qubits", "pca_components", "seed", "balanced_accuracy",
            "lower_95", "upper_95", "one_sided_lower_95", "p_value_holm",
        ]
        with (summary_dir / "scaling_metrics.csv").open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
            writer.writeheader()
            for record in records:
                writer.writerow({
                    "id": record["id"],
                    "n_qubits": record["n_qubits"],
                    "pca_components": record["pca_components"],
                    "seed": record["seed"],
                    "balanced_accuracy": record["balanced_accuracy"],
                    "lower_95": record["bootstrap_ci"]["lower_95"],
                    "upper_95": record["bootstrap_ci"]["upper_95"],
                    "one_sided_lower_95": record["bootstrap_ci"]["one_sided_lower_95"],
                    "p_value_holm": record["permutation_test"]["p_value_holm"],
                })

        return summary