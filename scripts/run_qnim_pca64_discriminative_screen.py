"""Screen PCA prefixes with a fixed train-only StandardScaler/RBF SVC."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "results" / "scaling_study" / "qnim_q1_6_dataset_v2_pca64"
OUTPUT_JSON = ROOT / "results" / "scaling_study" / "qnim_pca64_discriminative_screen_v1.json"
OUTPUT_MD = ROOT / "results" / "scaling_study" / "qnim_pca64_discriminative_screen_v1.md"
DIMENSIONS = (16, 20, 24, 32)
SVC_RANDOM_STATE = 42
STABILITY_ANGLE_TOLERANCE = {"rtol": 1e-8, "atol": 1e-8}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_inputs() -> dict:
    manifest_path = DATASET_DIR / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "immutable_complete":
        raise ValueError("PCA64 source dataset is not marked immutable_complete")

    required_files = (
        "X_train_raw.npy",
        "X_train_pca64.npy",
        "X_val_pca64.npy",
        "y_train.npy",
        "y_val.npy",
        "pca_state.npz",
        "pca64_variance_analysis.json",
    )
    for filename in required_files:
        expected = manifest["file_sha256"].get(filename)
        if expected is None or _sha256_file(DATASET_DIR / filename) != expected:
            raise ValueError(f"PCA64 source file hash mismatch: {filename}")

    metadata = json.loads(
        (DATASET_DIR / "dataset_metadata.json").read_text(encoding="utf-8")
    )
    with np.load(DATASET_DIR / "pca_state.npz", allow_pickle=False) as state:
        pca_components = state["components"].copy()
        explained_variance = state["explained_variance"].copy()
        explained_variance_ratio = state["explained_variance_ratio"].copy()

    result = {
        "manifest": manifest,
        "metadata": metadata,
        "X_train_raw": np.load(
            DATASET_DIR / "X_train_raw.npy", allow_pickle=False
        ),
        "X_train": np.load(
            DATASET_DIR / "X_train_pca64.npy", allow_pickle=False
        ),
        "X_val": np.load(
            DATASET_DIR / "X_val_pca64.npy", allow_pickle=False
        ),
        "y_train": np.load(DATASET_DIR / "y_train.npy", allow_pickle=False),
        "y_val": np.load(DATASET_DIR / "y_val.npy", allow_pickle=False),
        "pca_components": pca_components,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "variance_analysis": json.loads(
            (DATASET_DIR / "pca64_variance_analysis.json").read_text(
                encoding="utf-8"
            )
        ),
    }

    classes = len(metadata["generation"]["class_names"])
    expected_train = metadata["generation"]["train_examples"]
    expected_validation = metadata["generation"]["validation_examples"]
    if result["X_train"].shape != (expected_train, 64):
        raise ValueError(f"Unexpected PCA train shape: {result['X_train'].shape}")
    if result["X_val"].shape != (expected_validation, 64):
        raise ValueError(f"Unexpected PCA validation shape: {result['X_val'].shape}")
    if result["X_train_raw"].shape[0] != expected_train:
        raise ValueError("Raw train rows do not match PCA train rows")
    if result["y_train"].shape != (expected_train,):
        raise ValueError("Unexpected train label shape")
    if result["y_val"].shape != (expected_validation,):
        raise ValueError("Unexpected validation label shape")
    labels = np.arange(classes)
    for split_name in ("train", "val"):
        y = result[f"y_{split_name}"]
        counts = np.bincount(y, minlength=classes)
        expected_per_class = metadata["generation"][
            "train_per_class" if split_name == "train" else "validation_per_class"
        ]
        if not np.array_equal(np.unique(y), labels) or not np.all(
            counts == expected_per_class
        ):
            raise ValueError(f"{split_name} labels are not balanced as specified")
    for key in ("X_train_raw", "X_train", "X_val"):
        if not np.isfinite(result[key]).all():
            raise ValueError(f"Non-finite values in {key}")
    if result["pca_components"].shape != (64, result["X_train_raw"].shape[1]):
        raise ValueError("Unexpected PCA component shape")
    return result


def _classification_metrics(
    y_true: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, names: list[str]
) -> dict:
    class_ids = np.arange(len(names))
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "global_accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                labels=class_ids,
                average="macro",
                zero_division=0,
            )
        ),
        "cross_entropy": float(
            log_loss(
                y_true,
                np.clip(probabilities, 1e-12, 1.0),
                labels=class_ids,
            )
        ),
        "per_class_recall": {
            name: float(value)
            for name, value in zip(
                names,
                recall_score(
                    y_true,
                    predictions,
                    labels=class_ids,
                    average=None,
                    zero_division=0,
                ),
            )
        },
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=class_ids
        ).tolist(),
    }


def _evaluate_dimension(
    dimension: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_names: list[str],
) -> dict:
    scaler = StandardScaler()
    model = SVC(
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        kernel="rbf",
        random_state=SVC_RANDOM_STATE,
    )
    started = time.perf_counter()
    X_train_scaled = scaler.fit_transform(X_train[:, :dimension])
    model.fit(X_train_scaled, y_train)
    fit_seconds = time.perf_counter() - started
    if scaler.n_samples_seen_ != len(y_train):
        raise RuntimeError("StandardScaler was not fitted on all train examples")
    if not np.array_equal(model.classes_, np.arange(len(class_names))):
        raise RuntimeError("SVC class probability order differs from class IDs")

    started = time.perf_counter()
    X_val_scaled = scaler.transform(X_val[:, :dimension])
    predictions = model.predict(X_val_scaled)
    probabilities = model.predict_proba(X_val_scaled)
    inference_seconds = time.perf_counter() - started
    return {
        "n_components": dimension,
        "fit_seconds": float(fit_seconds),
        "inference_seconds": float(inference_seconds),
        "scaler_fit_split": "train_only",
        "classifier_fit_split": "train_only",
        "metrics": _classification_metrics(
            y_val, predictions, probabilities, class_names
        ),
    }


def _angle_summary(angles_by_resample: list[list[float]]) -> dict:
    pooled = np.asarray(angles_by_resample, dtype=np.float64).ravel()
    maximum_by_resample = np.max(
        np.asarray(angles_by_resample, dtype=np.float64), axis=1
    )
    return {
        "pooled_principal_angle_percentiles_degrees": {
            "median": float(np.quantile(pooled, 0.50)),
            "p90": float(np.quantile(pooled, 0.90)),
            "p95": float(np.quantile(pooled, 0.95)),
            "p99": float(np.quantile(pooled, 0.99)),
            "maximum": float(np.max(pooled)),
        },
        "maximum_principal_angle_per_resample_degrees": [
            float(value) for value in maximum_by_resample
        ],
        "median_of_resample_maxima_degrees": float(
            np.median(maximum_by_resample)
        ),
    }


def _run_stability_followup(inputs: dict) -> dict:
    stability = inputs["variance_analysis"]["stability"]
    stored = stability["principal_angle_comparison"]["subspaces"]
    labels = inputs["y_train"]
    raw_train = inputs["X_train_raw"]
    reference_components = inputs["pca_components"]
    dimensions = (16, 20, 24, 32)
    by_dimension = {dimension: [] for dimension in dimensions}
    old_dimensions = (16, 32)
    reconstructed_matches = {str(dimension): True for dimension in old_dimensions}
    max_reconstruction_differences = {
        str(dimension): 0.0 for dimension in old_dimensions
    }

    if stability["method"] != (
        "class-stratified 80% subsampling without replacement, train only"
    ):
        raise ValueError("Unexpected saved PCA resampling method")
    if stability["resample_count"] != len(stability["subsamples"]):
        raise ValueError("Saved PCA resampling count does not match records")

    unique_labels = np.unique(labels)
    samples_per_class = stability["samples_per_class_per_resample"]
    for record in stability["subsamples"]:
        replicate = record["replicate"] - 1
        selection_rng = np.random.default_rng(record["selection_seed"])
        selected = np.concatenate(
            [
                selection_rng.choice(
                    np.flatnonzero(labels == label),
                    size=samples_per_class,
                    replace=False,
                )
                for label in unique_labels
            ]
        )
        selected.sort()
        if len(selected) != stability["total_samples_per_resample"]:
            raise ValueError("Reconstructed subsample has an unexpected size")
        for label in unique_labels:
            if np.count_nonzero(labels[selected] == label) != samples_per_class:
                raise ValueError("Reconstructed subsample is not class-stratified")

        model = PCA(
            n_components=64,
            svd_solver=inputs["metadata"]["pca"]["svd_solver"],
            random_state=record["pca_random_state"],
        )
        model.fit(raw_train[selected])
        for dimension in dimensions:
            angles = np.degrees(
                subspace_angles(
                    reference_components[:dimension].T,
                    model.components_[:dimension].T,
                )
            )
            by_dimension[dimension].append(angles.tolist())
            if dimension in old_dimensions:
                old = np.asarray(
                    stored[str(dimension)][
                        "principal_angles_degrees_by_resample"
                    ][replicate]
                )
                difference = float(np.max(np.abs(angles - old)))
                max_reconstruction_differences[str(dimension)] = max(
                    max_reconstruction_differences[str(dimension)], difference
                )
                reconstructed_matches[str(dimension)] &= bool(
                    np.allclose(
                        angles,
                        old,
                        **STABILITY_ANGLE_TOLERANCE,
                    )
                )

    if not all(reconstructed_matches.values()):
        raise ValueError(
            "Reconstructed PC16/PC32 resampling does not match the saved analysis: "
            f"{max_reconstruction_differences}"
        )

    component_records = inputs["variance_analysis"]["components"]
    eigenvalue_gaps = {}
    for dimension in dimensions:
        lower = float(inputs["explained_variance"][dimension - 1])
        upper = float(inputs["explained_variance"][dimension])
        eigenvalue_gaps[str(dimension)] = {
            "boundary": f"PC{dimension}/PC{dimension + 1}",
            "lambda_k": lower,
            "lambda_k_plus_1": upper,
            "absolute_gap": lower - upper,
            "relative_gap_over_lambda_k": (lower - upper) / lower,
            "boundary_component_variance_ratio": float(
                inputs["explained_variance_ratio"][dimension - 1]
            ),
            "next_component_variance_ratio": float(
                inputs["explained_variance_ratio"][dimension]
            ),
            "variance_ratio_in_components_1_to_k": float(
                component_records[dimension - 1][
                    "cumulative_explained_variance_ratio"
                ]
            ),
            "variance_ratio_in_components_k_plus_1_to_64": float(
                np.sum(inputs["explained_variance_ratio"][dimension:])
            ),
            "fraction_of_total_variance_after_boundary": float(
                np.sum(inputs["explained_variance_ratio"][dimension:])
            ),
        }

    return {
        "source_resampling_repetitions": int(stability["resample_count"]),
        "bootstrap_repetitions": 0,
        "method": stability["method"],
        "sampling_strategy": (
            "For each of 13 classes, sample 160 of 200 train events without "
            "replacement; concatenate and sort indices."
        ),
        "samples_per_class_per_resample": int(samples_per_class),
        "total_samples_per_resample": int(
            stability["total_samples_per_resample"]
        ),
        "selection_seed_base": int(stability["selection_seed_base"]),
        "pca_random_state_base": int(stability["pca_random_state_base"]),
        "quantile_method": "NumPy default linear interpolation over all angles pooled across resamples",
        "subspaces": {
            str(dimension): {
                **_angle_summary(by_dimension[dimension]),
                "principal_angles_degrees_by_resample": by_dimension[dimension],
            }
            for dimension in dimensions
        },
        "reconstruction_check_against_saved_dimensions": {
            "matches": reconstructed_matches,
            "maximum_absolute_difference_degrees": max_reconstruction_differences,
            "tolerances": STABILITY_ANGLE_TOLERANCE,
        },
        "eigenvalue_gaps": eigenvalue_gaps,
    }


def _paired_changes(results: dict) -> dict:
    reference = results["16"]["metrics"]
    scalar_metrics = (
        "balanced_accuracy",
        "global_accuracy",
        "macro_f1",
        "cross_entropy",
    )
    changes = {}
    for dimension, result in results.items():
        metrics = result["metrics"]
        changes[dimension] = {
            "paired_metric_differences_vs_pca16": {
                name: float(metrics[name] - reference[name])
                for name in scalar_metrics
            },
            "per_class_recall_difference_vs_pca16": {
                name: float(value - reference["per_class_recall"][name])
                for name, value in metrics["per_class_recall"].items()
            },
            "confusion_matrix_count_difference_vs_pca16": (
                np.asarray(metrics["confusion_matrix"], dtype=np.int64)
                - np.asarray(reference["confusion_matrix"], dtype=np.int64)
            ).tolist(),
        }
    return changes


def _markdown_report(payload: dict) -> str:
    names = payload["class_names"]
    rows = []
    for dimension, result in payload["results"].items():
        metrics = result["metrics"]
        delta = payload["paired_changes_vs_pca16"][dimension][
            "paired_metric_differences_vs_pca16"
        ]
        rows.append(
            "| {d} | {ba:.4f} | {acc:.4f} | {f1:.4f} | {ce:.4f} | "
            "{dba:+.4f} | {dacc:+.4f} | {df1:+.4f} | {dce:+.4f} | "
            "{fit:.3f} | {infer:.3f} |".format(
                d=dimension,
                ba=metrics["balanced_accuracy"],
                acc=metrics["global_accuracy"],
                f1=metrics["macro_f1"],
                ce=metrics["cross_entropy"],
                dba=delta["balanced_accuracy"],
                dacc=delta["global_accuracy"],
                df1=delta["macro_f1"],
                dce=delta["cross_entropy"],
                fit=result["fit_seconds"],
                infer=result["inference_seconds"],
            )
        )
    recall_rows = []
    for name in names:
        recall_rows.append(
            "| {name} | {values} |".format(
                name=name,
                values=" | ".join(
                    f"{payload['results'][str(d)]['metrics']['per_class_recall'][name]:.3f}"
                    for d in DIMENSIONS
                ),
            )
        )
    stability_rows = []
    for dimension in DIMENSIONS:
        summary = payload["pca_stability"]["subspaces"][str(dimension)][
            "pooled_principal_angle_percentiles_degrees"
        ]
        gap = payload["pca_stability"]["eigenvalue_gaps"][str(dimension)]
        stability_rows.append(
            f"| {dimension} | {summary['median']:.4f} | {summary['p90']:.4f} | "
            f"{summary['p95']:.4f} | {summary['p99']:.4f} | "
            f"{summary['maximum']:.4f} | {gap['relative_gap_over_lambda_k']:.6g} | "
            f"{gap['boundary_component_variance_ratio']:.3e} | "
            f"{gap['fraction_of_total_variance_after_boundary'] * 100:.3e}% |"
        )
    tail_at_32 = payload["pca_stability"]["eigenvalue_gaps"]["32"][
        "fraction_of_total_variance_after_boundary"
    ]

    best_dimension = max(
        DIMENSIONS,
        key=lambda d: payload["results"][str(d)]["metrics"]["balanced_accuracy"],
    )
    delta_best = payload["paired_changes_vs_pca16"][str(best_dimension)][
        "paired_metric_differences_vs_pca16"
    ]["balanced_accuracy"]
    return f"""# QNIM PCA64 discriminative scaling screen

## Scope and protocol

- Dataset: `qnim_q1_6_dataset_v2_pca64`; only train and validation arrays were loaded.
- No final test data, QNIM training, or IBM hardware was used. Source dataset and prior reports were not modified.
- The same 2,600 training and 650 validation examples and labels were used at every dimension.
- For each PCA prefix, `StandardScaler` was fit on train only, followed by RBF `SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=42)`. No hyperparameters were tuned on validation.
- `fit_seconds` includes scaler fitting and SVC fitting/calibration. `inference_seconds` includes validation scaling, `predict`, and `predict_proba`.

## Validation results

| PCA dims | Balanced accuracy | Global accuracy | Macro-F1 | Cross-entropy | Δ BA vs 16 | Δ accuracy | Δ macro-F1 | Δ cross-entropy | Fit s | Inference s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

### Per-class recall

| Class | PCA16 | PCA20 | PCA24 | PCA32 |
|---|---:|---:|---:|---:|
{chr(10).join(recall_rows)}

All four 13×13 confusion matrices and their count differences against PCA16 are in the JSON report; rows are true classes and columns are predicted classes, in this order: {", ".join(names)}.

The highest observed validation balanced accuracy is PCA{best_dimension} (difference vs PCA16: {delta_best:+.4f}). This is a descriptive comparison on one fixed validation set, not a significance test or an independent test result.

## PCA subspace stability and eigenvalue gaps

The source analysis used 20 **stratified subsamples**, not bootstrap samples: 160 of 200 training events per class (2,080 total), without replacement. For PC20/PC24, the sample indices were reconstructed from the saved seeds and strategy, then the randomized PCA fits were repeated on the existing raw training matrix; no waveforms were regenerated. Reconstructed PC16/PC32 angles match the saved values within the recorded tolerance.

Percentiles are over all principal angles pooled across the 20 resamples. The eigenvalue gap is `(lambda_k - lambda_(k+1))/lambda_k`; boundary variance ratios are from the full training-only PCA.

| Boundary k | Median angle ° | P90 ° | P95 ° | P99 ° | Maximum ° | Relative eigenvalue gap | Variance ratio at PCk | Variance after k through PC64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(stability_rows)}

## Conclusions and limitations

- This screen tests whether fixed SVC validation metrics change with PCA dimension; it does not establish generalization beyond this validation set.
- The JSON records paired metric differences, per-class recall changes, confusion matrices, timings, all angle distributions, and eigenvalue gaps.
- At PC32, components 33–64 account for {tail_at_32 * 100:.6g}% of total variance. Compare this low-variance tail with the PC32 angle distribution and eigenvalue gap: instability in these directions is consistent with weakly determined, low-variance PCA directions, not by itself evidence of class-discriminative information.
- Low-variance directions can have unstable bases even when cumulative explained variance is high; PCA subspace stability should be considered alongside SVC performance.
- Probability calibration is internal to SVC and uses training data. The validation set was used only for the requested final comparison.
"""


def run_screen() -> dict:
    if OUTPUT_JSON.exists() or OUTPUT_MD.exists():
        raise FileExistsError("Refusing to overwrite an existing screen report")
    started = time.perf_counter()
    inputs = _load_inputs()
    classifier_started = time.perf_counter()
    class_names = inputs["metadata"]["generation"]["class_names"]
    results = {}
    for dimension in DIMENSIONS:
        results[str(dimension)] = _evaluate_dimension(
            dimension,
            inputs["X_train"],
            inputs["y_train"],
            inputs["X_val"],
            inputs["y_val"],
            class_names,
        )
    classifier_seconds = time.perf_counter() - classifier_started

    stability_started = time.perf_counter()
    stability = _run_stability_followup(inputs)
    stability_seconds = time.perf_counter() - stability_started
    payload = {
        "schema_version": 1,
        "study_id": "QNIM_PCA64_DISCRIMINATIVE_SCREEN_V1",
        "status": "complete",
        "source_dataset": {
            "dataset_id": inputs["metadata"]["dataset_id"],
            "directory": "qnim_q1_6_dataset_v2_pca64",
            "dataset_manifest_sha256": _sha256_file(
                DATASET_DIR / "dataset_manifest.json"
            ),
            "verified_source_file_sha256": {
                filename: inputs["manifest"]["file_sha256"][filename]
                for filename in (
                    "X_train_raw.npy",
                    "X_train_pca64.npy",
                    "X_val_pca64.npy",
                    "y_train.npy",
                    "y_val.npy",
                    "pca_state.npz",
                    "pca64_variance_analysis.json",
                )
            },
            "loaded_splits": ["train", "validation"],
            "test_accessed": False,
        },
        "classifier": {
            "name": "StandardScaler + RBF SVC",
            "parameters": {
                "C": 1.0,
                "gamma": "scale",
                "class_weight": "balanced",
                "probability": True,
                "kernel": "rbf",
                "random_state": SVC_RANDOM_STATE,
            },
            "scaler": "StandardScaler fit on training rows only",
            "hyperparameter_tuning": False,
            "dimensions": list(DIMENSIONS),
            "same_train_validation_examples_and_labels": True,
            "validation_used_for_selection_or_tuning": False,
        },
        "class_names": class_names,
        "validation_examples": int(len(inputs["y_val"])),
        "results": results,
        "paired_changes_vs_pca16": _paired_changes(results),
        "pca_stability": stability,
        "timing": {
            "classifier_screen_seconds": float(classifier_seconds),
            "stability_followup_seconds": float(stability_seconds),
            "total_seconds": float(time.perf_counter() - started),
            "clock": "time.perf_counter",
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
        "limitations": [
            "One fixed train/validation split and one classifier random state; metric differences are descriptive, not inferential.",
            "No final test data were accessed.",
            "The PCA20/24 stability follow-up refits PCA on existing train-only resamples; it did not regenerate waveforms.",
        ],
    }
    report = _markdown_report(payload)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    OUTPUT_MD.write_text(report, encoding="utf-8")
    return payload


def main() -> int:
    payload = run_screen()
    print(
        json.dumps(
            {
                "status": payload["status"],
                "json": str(OUTPUT_JSON),
                "markdown": str(OUTPUT_MD),
                "results": {
                    dimension: {
                        "balanced_accuracy": result["metrics"][
                            "balanced_accuracy"
                        ],
                        "global_accuracy": result["metrics"][
                            "global_accuracy"
                        ],
                        "macro_f1": result["metrics"]["macro_f1"],
                        "cross_entropy": result["metrics"]["cross_entropy"],
                    }
                    for dimension, result in payload["results"].items()
                },
                "runtime": payload["timing"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
