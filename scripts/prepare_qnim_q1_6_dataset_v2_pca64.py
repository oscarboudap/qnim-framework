"""Build a train/validation-only Q1.6 dataset retaining 64 PCA components."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from scipy.linalg import subspace_angles, svdvals
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT_DIR = ROOT / "results" / "scaling_study" / "qnim_q1_6_dataset_v2_pca64"
SOURCE_DIR = ROOT / "results" / "scaling_study" / "qnim_q1_6_dataset_v1"
Q1_6_REPORT = ROOT / "results" / "scaling_study" / "qnim_q1_6_scaling_pilot_v1.json"

DATASET_SEED = 42
PCA_COMPONENTS = 64
PCA_RANDOM_STATE = 42
PCA_SOLVER = "randomized"
PREFIX_COMPONENTS = 16
PREFIX_RTOL = 1e-8
PREFIX_ATOL = 1e-10
RESAMPLE_COUNT = 20
RESAMPLE_FRACTION = 0.8
RESAMPLE_SEED_BASE = 4200
RESAMPLE_PCA_SEED_BASE = 5200
RANK_RELATIVE_TOLERANCES = (1e-12, 1e-10, 1e-8)


class IntegrityMismatch(RuntimeError):
    def __init__(self, details: dict):
        super().__init__("Historical 16-component prefix failed integrity checks")
        self.details = details


class PeakRSSMonitor:
    def __init__(self, interval_seconds: float = 0.1):
        self.process = psutil.Process()
        self.interval_seconds = interval_seconds
        self.initial_bytes = 0
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self.initial_bytes = self.process.memory_info().rss
        self.peak_bytes = self.initial_bytes
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _split_sha256(features: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (features, labels):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict:
    source_files = (
        "scripts/prepare_qnim_q1_6_dataset_v2_pca64.py",
        "scripts/run_qnim_q1_6_scaling_pilot.py",
        "src/infrastructure/ligo_pca_adapter.py",
        "src/infrastructure/stochastic_signal_generator.py",
        "src/domain/astrophysics/sstg/injectors/layer5_beyond_gr_complete.py",
        "src/domain/astrophysics/sstg/injectors/layer6_horizon_topology_complete.py",
    )
    return {
        filename: _file_sha256(ROOT / filename) for filename in source_files
    }


def _runtime_versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "pycbc": importlib.metadata.version("PyCBC"),
        "qiskit": importlib.metadata.version("qiskit"),
        "qiskit_aer": importlib.metadata.version("qiskit-aer"),
        "matplotlib": importlib.metadata.version("matplotlib"),
        "psutil": importlib.metadata.version("psutil"),
    }


def _read_frozen_configuration() -> tuple[dict, dict, dict]:
    from scripts.run_qnim_q1_6_scaling_pilot import _read_frozen_sources

    sources = _read_frozen_sources()
    report = json.loads(Q1_6_REPORT.read_text(encoding="utf-8"))
    old_metadata = json.loads(
        (SOURCE_DIR / "dataset_metadata.json").read_text(encoding="utf-8")
    )
    generation = old_metadata["generation"]
    config = sources["dataset_configuration"]
    if generation["dataset_seed"] != DATASET_SEED:
        raise ValueError("The frozen dataset seed is not 42")
    if generation["train_per_class"] != config["train_per_class"]:
        raise ValueError("Train count differs from the frozen Q1.6 protocol")
    if generation["validation_per_class"] != config["validation_per_class"]:
        raise ValueError("Validation count differs from the frozen Q1.6 protocol")
    if generation["n_pca_fit"] != PCA_COMPONENTS:
        raise ValueError("The historical PCA component budget is not 64")
    if generation["strict_physics"] is not True:
        raise ValueError("The frozen dataset was not generated in strict-physics mode")
    return sources, report, old_metadata


def _verify_versions(recorded: dict, current: dict) -> None:
    version_keys = (
        "python",
        "numpy",
        "scikit_learn",
        "pycbc",
        "qiskit",
        "qiskit_aer",
    )
    mismatches = {
        key: {"recorded": recorded.get(key), "current": current.get(key)}
        for key in version_keys
        if recorded.get(key) != current.get(key)
    }
    if mismatches:
        raise ValueError(f"Recorded dependency versions do not match: {mismatches}")


def _advance_rng_without_waveform(
    theory_class: int,
    count: int,
    adapter,
    rng: np.random.Generator,
) -> None:
    from src.infrastructure.stochastic_signal_generator import (
        _inject_qnm_subdominant,
    )

    dummy_spectrum = np.zeros(2, dtype=np.complex128)
    dummy_frequencies = np.array([0.0, 1.0])
    for _ in range(count):
        mass1 = float(rng.uniform(*adapter.m_range))
        rng.uniform(adapter.m_range[0], mass1)
        rng.uniform(*adapter.d_range)
        if theory_class == 1:
            rng.uniform(60.0, 80.0)
        elif theory_class == 2:
            _inject_qnm_subdominant(
                dummy_spectrum, dummy_frequencies, mode="21", rng=rng
            )
        elif theory_class == 3:
            _inject_qnm_subdominant(
                dummy_spectrum, dummy_frequencies, mode="33", rng=rng
            )
        elif theory_class == 4:
            rng.uniform(0.001, 0.01)
        elif theory_class == 6:
            rng.uniform(40000.0, 80000.0)
        elif theory_class == 8:
            rng.uniform(0.5, 1.2)
        elif theory_class in (9, 10, 12):
            rng.uniform(0.03, 0.10)


def _generate_train_validation(adapter, config: dict, generation: dict) -> tuple:
    train_per_class = config["train_per_class"]
    validation_per_class = config["validation_per_class"]
    discarded_draws = generation["discarded_post_validation_draws_per_class"]
    rng = np.random.default_rng(DATASET_SEED)
    train_raw = None
    validation_raw = None
    train_labels = np.empty(adapter.n_classes * train_per_class, dtype=int)
    validation_labels = np.empty(
        adapter.n_classes * validation_per_class, dtype=int
    )
    train_cursor = 0
    validation_cursor = 0
    expected_length = int(adapter.duration * adapter.sample_rate)

    for class_index in range(adapter.n_classes):
        for event_index in range(train_per_class + validation_per_class):
            waveform = adapter._generate_one(class_index, rng)
            if waveform.ndim != 1 or waveform.size != expected_length:
                raise ValueError(
                    f"Unexpected waveform shape for class {class_index}: "
                    f"{waveform.shape}; expected ({expected_length},)"
                )
            if train_raw is None:
                dtype = waveform.dtype
                train_raw = np.empty(
                    (len(train_labels), waveform.size), dtype=dtype
                )
                validation_raw = np.empty(
                    (len(validation_labels), waveform.size), dtype=dtype
                )
            elif waveform.dtype != train_raw.dtype:
                raise ValueError("Pre-PCA waveform dtype changed during generation")

            if event_index < train_per_class:
                train_raw[train_cursor] = waveform
                train_labels[train_cursor] = class_index
                train_cursor += 1
            else:
                validation_raw[validation_cursor] = waveform
                validation_labels[validation_cursor] = class_index
                validation_cursor += 1

        # Advance only the RNG parameters of discarded historical draws; no
        # post-validation waveform or split is constructed or persisted.
        _advance_rng_without_waveform(
            class_index, discarded_draws, adapter, rng
        )

    if train_cursor != len(train_labels) or validation_cursor != len(validation_labels):
        raise RuntimeError("Generated train/validation sample counts are incomplete")
    return train_raw, train_labels, validation_raw, validation_labels


def _align_component_signs(scores: np.ndarray, historical_train: np.ndarray) -> tuple:
    signs = np.ones(scores.shape[1], dtype=np.float64)
    correlations = []
    for component in range(PREFIX_COMPONENTS):
        new_values = scores[:, component]
        old_values = historical_train[:, component]
        dot = np.dot(
            new_values - new_values.mean(),
            old_values - old_values.mean(),
        )
        signs[component] = -1.0 if dot < 0 else 1.0
        correlation = np.corrcoef(new_values, old_values)[0, 1]
        correlations.append(float(correlation * signs[component]))
    return signs, correlations


def _load_and_verify_historical_cache(metadata: dict) -> dict:
    cache = {
        "X_train": np.load(SOURCE_DIR / "X_train.npy", allow_pickle=False),
        "y_train": np.load(SOURCE_DIR / "y_train.npy", allow_pickle=False),
        "X_val": np.load(SOURCE_DIR / "X_val.npy", allow_pickle=False),
        "y_val": np.load(SOURCE_DIR / "y_val.npy", allow_pickle=False),
    }
    expected_hashes = metadata["dataset_hashes"]
    hashes_match = (
        _split_sha256(cache["X_train"], cache["y_train"])
        == expected_hashes["train_sha256"]
        and _split_sha256(cache["X_val"], cache["y_val"])
        == expected_hashes["validation_sha256"]
    )
    if not hashes_match:
        raise ValueError("The existing 16-component cache hash is invalid")
    return cache


def _check_historical_prefix(
    train_scaled: np.ndarray,
    validation_scaled: np.ndarray,
    labels_train: np.ndarray,
    labels_validation: np.ndarray,
    historical: dict,
    historical_cache: dict,
) -> dict:
    old_train = historical_cache["X_train"]
    old_labels_train = historical_cache["y_train"]
    old_validation = historical_cache["X_val"]
    old_labels_validation = historical_cache["y_val"]
    if not np.array_equal(labels_train, old_labels_train):
        raise IntegrityMismatch({"split": "train", "labels_and_order_match": False})
    if not np.array_equal(labels_validation, old_labels_validation):
        raise IntegrityMismatch(
            {"split": "validation", "labels_and_order_match": False}
        )

    train_error = train_scaled[:, :PREFIX_COMPONENTS] - old_train
    validation_error = validation_scaled[:, :PREFIX_COMPONENTS] - old_validation
    train_ok = np.allclose(
        train_scaled[:, :PREFIX_COMPONENTS],
        old_train,
        rtol=PREFIX_RTOL,
        atol=PREFIX_ATOL,
    )
    validation_ok = np.allclose(
        validation_scaled[:, :PREFIX_COMPONENTS],
        old_validation,
        rtol=PREFIX_RTOL,
        atol=PREFIX_ATOL,
    )
    details = {
        "labels_and_order_match": True,
        "signs_aligned_before_scaling": True,
        "signs": historical["signs"],
        "correlations_after_sign_alignment": historical["correlations"],
        "tolerances": {"rtol": PREFIX_RTOL, "atol": PREFIX_ATOL},
        "train": {
            "matches": bool(train_ok),
            "max_absolute_error": float(np.max(np.abs(train_error))),
            "rmse": float(np.sqrt(np.mean(train_error**2))),
        },
        "validation": {
            "matches": bool(validation_ok),
            "max_absolute_error": float(np.max(np.abs(validation_error))),
            "rmse": float(np.sqrt(np.mean(validation_error**2))),
        },
    }
    details["historical_cache_hashes_verified"] = True
    if not train_ok or not validation_ok:
        raise IntegrityMismatch(details)
    return details


def _numerical_rank(X_train_raw: np.ndarray, pca: PCA) -> tuple:
    centered = X_train_raw - pca.mean_
    singular_values = svdvals(
        centered, overwrite_a=True, check_finite=False
    )
    largest = float(singular_values[0])
    machine_tolerance = max(X_train_raw.shape) * np.finfo(X_train_raw.dtype).eps * largest
    ranks = {
        "numpy_default_matrix_rank": int(
            np.count_nonzero(singular_values > machine_tolerance)
        ),
        "relative_tolerance_1e-12": int(
            np.count_nonzero(singular_values > largest * 1e-12)
        ),
        "relative_tolerance_1e-10": int(
            np.count_nonzero(singular_values > largest * 1e-10)
        ),
        "relative_tolerance_1e-8": int(
            np.count_nonzero(singular_values > largest * 1e-8)
        ),
    }
    return singular_values, {
        "method": "full singular-value decomposition of centered X_train_raw",
        "shape": list(X_train_raw.shape),
        "centered_rank_upper_bound": min(
            X_train_raw.shape[0] - 1, X_train_raw.shape[1]
        ),
        "machine_tolerance": machine_tolerance,
        "relative_tolerances": list(RANK_RELATIVE_TOLERANCES),
        "rank_by_tolerance": ranks,
        "largest_singular_value": largest,
        "smallest_returned_singular_value": float(singular_values[-1]),
    }


def _stability_analysis(X_train_raw: np.ndarray, y_train: np.ndarray, reference: PCA) -> dict:
    from sklearn.decomposition import PCA as ResamplePCA

    labels = np.unique(y_train)
    per_class_size = int(
        round(np.count_nonzero(y_train == labels[0]) * RESAMPLE_FRACTION)
    )
    ratio_runs = []
    angle_runs = {str(k): [] for k in (16, 32, 64)}
    selections = []
    for replicate in range(RESAMPLE_COUNT):
        selection_rng = np.random.default_rng(RESAMPLE_SEED_BASE + replicate)
        selected = np.concatenate(
            [
                selection_rng.choice(
                    np.flatnonzero(y_train == label),
                    size=per_class_size,
                    replace=False,
                )
                for label in labels
            ]
        )
        selected.sort()
        pca_seed = RESAMPLE_PCA_SEED_BASE + replicate
        resampled_pca = ResamplePCA(
            n_components=PCA_COMPONENTS,
            svd_solver=PCA_SOLVER,
            random_state=pca_seed,
        )
        resampled_pca.fit(X_train_raw[selected])
        ratio_runs.append(resampled_pca.explained_variance_ratio_)
        selections.append(
            {
                "replicate": replicate + 1,
                "selection_seed": RESAMPLE_SEED_BASE + replicate,
                "pca_random_state": pca_seed,
                "samples_per_class": per_class_size,
                "total_samples": int(len(selected)),
            }
        )
        for k in (16, 32, 64):
            angles = np.degrees(
                subspace_angles(
                    reference.components_[:k].T,
                    resampled_pca.components_[:k].T,
                )
            )
            angle_runs[str(k)].append(angles.tolist())

    ratios = np.asarray(ratio_runs)
    ratio_summary = [
        {
            "component": component + 1,
            "mean": float(np.mean(ratios[:, component])),
            "std": float(np.std(ratios[:, component], ddof=1)),
            "min": float(np.min(ratios[:, component])),
            "max": float(np.max(ratios[:, component])),
        }
        for component in range(PCA_COMPONENTS)
    ]
    angle_summary = {}
    for k, runs in angle_runs.items():
        array = np.asarray(runs)
        max_by_replicate = np.max(array, axis=1)
        mean_by_replicate = np.mean(array, axis=1)
        angle_summary[k] = {
            "principal_angles_degrees_by_resample": runs,
            "mean_of_all_angles_degrees": float(np.mean(array)),
            "std_of_all_angles_degrees": float(np.std(array, ddof=1)),
            "mean_max_angle_per_resample_degrees": float(
                np.mean(max_by_replicate)
            ),
            "maximum_angle_degrees": float(np.max(array)),
            "mean_angle_per_resample_degrees": [
                float(value) for value in mean_by_replicate
            ],
        }
    return {
        "method": "class-stratified 80% subsampling without replacement, train only",
        "resample_count": RESAMPLE_COUNT,
        "samples_per_class_per_resample": per_class_size,
        "total_samples_per_resample": int(per_class_size * len(labels)),
        "selection_seed_base": RESAMPLE_SEED_BASE,
        "pca_random_state_base": RESAMPLE_PCA_SEED_BASE,
        "subsamples": selections,
        "explained_variance_ratio_by_component": ratio_summary,
        "principal_angle_comparison": {
            "reference": (
                f"full {reference.n_samples_}-example train-only PCA64 basis"
            ),
            "subspaces": angle_summary,
        },
    }


def _save_npy(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)


def _make_plots(path: Path, ratios: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    components = np.arange(1, len(ratios) + 1)
    cumulative = np.cumsum(ratios) * 100
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.semilogy(components, ratios * 100, marker=".", linewidth=1.2)
    axis.set(title="PCA64 scree plot — train-only", xlabel="Component", ylabel="Explained variance (%)")
    axis.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path / "pca64_scree.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(components, cumulative, marker=".", linewidth=1.2)
    axis.set(
        title="PCA64 cumulative explained variance — train-only",
        xlabel="Components",
        ylabel="Cumulative explained variance (%)",
    )
    axis.set_ylim(0, 100.2)
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path / "pca64_cumulative_variance.png", dpi=160)
    plt.close(fig)


def _variance_report(
    pca: PCA, full_singular_values: np.ndarray, rank_report: dict, stability: dict
) -> dict:
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return {
        "pca_fit_split": "train_only",
        "fit_shape": rank_report["shape"],
        "solver": PCA_SOLVER,
        "random_state": PCA_RANDOM_STATE,
        "components": [
            {
                "component": i + 1,
                "explained_variance": float(pca.explained_variance_[i]),
                "explained_variance_ratio": float(
                    pca.explained_variance_ratio_[i]
                ),
                "cumulative_explained_variance_ratio": float(cumulative[i]),
                "pca_singular_value": float(pca.singular_values_[i]),
            }
            for i in range(PCA_COMPONENTS)
        ],
        "cumulative_ratio_at_16_32_64": {
            str(k): float(cumulative[k - 1]) for k in (16, 32, 64)
        },
        "full_centered_training_singular_values": [
            float(value) for value in full_singular_values
        ],
        "numerical_rank": rank_report,
        "stability": stability,
    }


def _markdown_report(metadata: dict, variance: dict) -> str:
    rank = variance["numerical_rank"]["rank_by_tolerance"]
    stability = variance["stability"]["principal_angle_comparison"]["subspaces"]
    prefix = metadata["historical_prefix_check"]
    angles = "\n".join(
        f"| {k} | {value['mean_of_all_angles_degrees']:.6g} | "
        f"{value['maximum_angle_degrees']:.6g} |"
        for k, value in stability.items()
    )
    return f"""# Q1.6 PCA64 train/validation dataset V2

## Status and scope

- Status: **complete**.
- Generated only {metadata['generation']['train_examples']} training and {metadata['generation']['validation_examples']} validation waveforms; no test waveform, split, or array was generated or loaded.
- No classifier or QNIM was trained.
- Historical datasets and reports were not modified.

## Configuration and provenance

- Classes: {metadata['generation']['class_count']}
- Train/validation: {metadata['generation']['train_per_class']} / {metadata['generation']['validation_per_class']} per class.
- Dataset seed: {metadata['generation']['dataset_seed']}.
- PCA: 64 components, randomized solver, random state 42; fit on `X_train_raw` only.
- Current versions: Python {metadata['runtime_versions']['python']}, NumPy {metadata['runtime_versions']['numpy']}, scikit-learn {metadata['runtime_versions']['scikit_learn']}, PyCBC {metadata['runtime_versions']['pycbc']}.
- Post-validation PRNG draws were advanced using parameter draws only to preserve the historical class ordering; no corresponding waveform or split was created.

## Historical prefix integrity

- Labels and class/sample order: **{prefix['labels_and_order_match']}**.
- Train and validation first-16 processed components match within `rtol={prefix['tolerances']['rtol']}`, `atol={prefix['tolerances']['atol']}`: **{prefix['train']['matches']} / {prefix['validation']['matches']}**.
- Signs aligned per component before scaling: `{prefix['signs'][:16]}`.
- Maximum absolute errors: train `{prefix['train']['max_absolute_error']:.4g}`, validation `{prefix['validation']['max_absolute_error']:.4g}`.

## PCA stability

20 train-only class-stratified subsamples were used, each retaining 80% (160 examples) per class without replacement. No validation rows were used for fitting or stability selection.

| Subspace dimension | Mean principal angle (degrees) | Maximum principal angle (degrees) |
|---:|---:|---:|
{angles}

Per-component explained-variance-ratio variation and every principal-angle vector are in `pca64_variance_analysis.json`.

## Numerical rank

Full singular values were computed for the centered training matrix. Rank counts:

- NumPy default tolerance: {rank['numpy_default_matrix_rank']}
- Relative tolerance 1e-12: {rank['relative_tolerance_1e-12']}
- Relative tolerance 1e-10: {rank['relative_tolerance_1e-10']}
- Relative tolerance 1e-8: {rank['relative_tolerance_1e-8']}

## Variance and artifacts

- Cumulative variance: PC16 {variance['cumulative_ratio_at_16_32_64']['16'] * 100:.6f}%, PC32 {variance['cumulative_ratio_at_16_32_64']['32'] * 100:.6f}%, PC64 {variance['cumulative_ratio_at_16_32_64']['64'] * 100:.6f}%.
- Runtime: {metadata['runtime']['total_seconds']:.2f} s.
- Peak RSS: {metadata['runtime']['peak_rss_bytes'] / 2**20:.1f} MiB.
- SHA-256 values for every array and artifact are recorded in `dataset_manifest.json`.

See `pca64_scree.png` and `pca64_cumulative_variance.png`.
"""


def build_dataset() -> dict:
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing dataset directory: {OUTPUT_DIR}"
        )
    current_versions = _runtime_versions()
    sources, historical_report, old_metadata = _read_frozen_configuration()
    _verify_versions(historical_report["software_versions"], current_versions)
    historical_cache = _load_and_verify_historical_cache(old_metadata)

    from src.infrastructure.ligo_pca_adapter import LIGOPCAAdapter

    generation = old_metadata["generation"]
    config = sources["dataset_configuration"]
    adapter = LIGOPCAAdapter(
        n_components=PCA_COMPONENTS,
        n_pca_fit=PCA_COMPONENTS,
        approximant=sources["approximant"],
        max_classes=len(sources["class_names"]),
        strict_physics=True,
    )
    started = time.perf_counter()
    stage_times = {}
    with PeakRSSMonitor() as memory:
        phase = time.perf_counter()
        X_train_raw, y_train, X_val_raw, y_val = _generate_train_validation(
            adapter, config, generation
        )
        stage_times["waveform_generation_seconds"] = time.perf_counter() - phase
        expected_length = int(adapter.duration * adapter.sample_rate)
        expected_train_shape = (
            adapter.n_classes * config["train_per_class"],
            expected_length,
        )
        expected_validation_shape = (
            adapter.n_classes * config["validation_per_class"],
            expected_length,
        )
        if X_train_raw.shape != expected_train_shape:
            raise ValueError(f"Unexpected X_train_raw shape: {X_train_raw.shape}")
        if X_val_raw.shape != expected_validation_shape:
            raise ValueError(f"Unexpected X_val_raw shape: {X_val_raw.shape}")

        phase = time.perf_counter()
        pca = PCA(
            n_components=PCA_COMPONENTS,
            svd_solver=PCA_SOLVER,
            random_state=PCA_RANDOM_STATE,
        )
        pca.fit(X_train_raw)
        if getattr(pca, "_fit_svd_solver", None) != PCA_SOLVER:
            raise RuntimeError("scikit-learn did not use the requested randomized solver")
        train_scores = pca.transform(X_train_raw)
        validation_scores = pca.transform(X_val_raw)
        historical_train = np.load(
            SOURCE_DIR / "X_train.npy", allow_pickle=False
        )
        signs, correlations = _align_component_signs(
            train_scores, historical_train
        )
        pca.components_ *= signs[:, None]
        train_scores *= signs
        validation_scores *= signs
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        train_scaled = np.clip(scaler.fit_transform(train_scores), -1.0, 1.0)
        validation_scaled = np.clip(
            scaler.transform(validation_scores), -1.0, 1.0
        )
        prefix_check = _check_historical_prefix(
            train_scaled,
            validation_scaled,
            y_train,
            y_val,
            {
                "signs": signs.tolist(),
                "correlations": correlations,
            },
            historical_cache,
        )
        stage_times["pca_and_prefix_check_seconds"] = time.perf_counter() - phase

        phase = time.perf_counter()
        full_singular_values, rank_report = _numerical_rank(X_train_raw, pca)
        stage_times["full_svd_rank_seconds"] = time.perf_counter() - phase

        phase = time.perf_counter()
        stability = _stability_analysis(X_train_raw, y_train, pca)
        stage_times["stability_analysis_seconds"] = time.perf_counter() - phase

        OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
        stage_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{OUTPUT_DIR.name}.building-",
                dir=OUTPUT_DIR.parent,
            )
        )
        phase = time.perf_counter()
        arrays = {
            "X_train_raw.npy": X_train_raw,
            "X_val_raw.npy": X_val_raw,
            "X_train_pca64.npy": train_scores,
            "X_val_pca64.npy": validation_scores,
            "X_train_scaled64.npy": train_scaled,
            "X_val_scaled64.npy": validation_scaled,
            "y_train.npy": y_train,
            "y_val.npy": y_val,
        }
        for filename, array in arrays.items():
            _save_npy(stage_dir / filename, array)

        np.savez_compressed(
            stage_dir / "pca_state.npz",
            mean=pca.mean_,
            components=pca.components_,
            explained_variance=pca.explained_variance_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            singular_values=pca.singular_values_,
            all_centered_training_singular_values=full_singular_values,
            n_samples_seen=np.asarray(X_train_raw.shape[0], dtype=np.int64),
            n_features_in=np.asarray(X_train_raw.shape[1], dtype=np.int64),
            sign_alignment=signs,
        )
        scaler_state = {
            "feature_range": list(scaler.feature_range),
            "clip": bool(scaler.clip),
            "n_features_in": int(scaler.n_features_in_),
            "n_samples_seen": int(scaler.n_samples_seen_),
            "min": scaler.min_.tolist(),
            "scale": scaler.scale_.tolist(),
            "data_min": scaler.data_min_.tolist(),
            "data_max": scaler.data_max_.tolist(),
            "data_range": scaler.data_range_.tolist(),
            "fit_split": "train_only",
        }
        _atomic_json(stage_dir / "scaler_state.json", scaler_state)

        variance_report = _variance_report(
            pca, full_singular_values, rank_report, stability
        )
        _atomic_json(
            stage_dir / "pca64_variance_analysis.json", variance_report
        )
        _make_plots(stage_dir, pca.explained_variance_ratio_)

        array_hashes = {
            filename: _array_sha256(array)
            for filename, array in arrays.items()
        }
        file_hashes = {
            filename: _file_sha256(stage_dir / filename)
            for filename in arrays
        }
        file_hashes["pca_state.npz"] = _file_sha256(
            stage_dir / "pca_state.npz"
        )
        file_hashes["scaler_state.json"] = _file_sha256(
            stage_dir / "scaler_state.json"
        )
        file_hashes["pca64_variance_analysis.json"] = _file_sha256(
            stage_dir / "pca64_variance_analysis.json"
        )
        file_hashes["pca64_scree.png"] = _file_sha256(
            stage_dir / "pca64_scree.png"
        )
        file_hashes["pca64_cumulative_variance.png"] = _file_sha256(
            stage_dir / "pca64_cumulative_variance.png"
        )

        runtime = {
            "stage_seconds": stage_times,
            "total_seconds": time.perf_counter() - started,
            "rss_start_bytes": memory.initial_bytes,
            "peak_rss_bytes": memory.peak_bytes,
            "peak_rss_delta_bytes": memory.peak_bytes - memory.initial_bytes,
            "sampling_interval_seconds": memory.interval_seconds,
            "clock": "time.perf_counter",
        }
        metadata = {
            "schema_version": 1,
            "dataset_id": "QNIM_Q1_6_DATASET_V2_PCA64",
            "status": "complete",
            "generation": {
                "dataset_seed": DATASET_SEED,
                "class_count": adapter.n_classes,
                "class_names": list(adapter.theory_names),
                "train_per_class": config["train_per_class"],
                "validation_per_class": config["validation_per_class"],
                "train_examples": int(len(y_train)),
                "validation_examples": int(len(y_val)),
                "strict_physics": True,
                "fallback_allowed": False,
                "waveform_source": "PyCBC",
                "approximant": sources["approximant"],
                "mass1_msun_range": list(adapter.m_range),
                "mass2_msun_range": [
                    adapter.m_range[0],
                    "mass1",
                ],
                "distance_mpc_range": list(adapter.d_range),
                "duration_seconds": adapter.duration,
                "sample_rate_hz": adapter.sample_rate,
                "f_lower_hz": adapter.f_lower,
                "f_low_white_hz": adapter.f_low_white,
                "f_high_white_hz": adapter.f_high_white,
                "post_validation_waveforms_generated": 0,
                "post_validation_rng_advance_only": generation[
                    "discarded_post_validation_draws_per_class"
                ],
                "post_validation_rng_advance_note": "Only seeded parameter RNG draws were consumed to preserve the historical inter-class stream; no post-validation waveform, label, matrix, or split was constructed.",
                "feature_shape_per_split": {
                    "X_train_raw": list(X_train_raw.shape),
                    "X_val_raw": list(X_val_raw.shape),
                },
                "pre_pca_feature_semantics": "single whitened time-domain hp waveform; one feature per time sample",
                "whitening": "analytic aLIGO O3 PSD, 20-1024 Hz bandpass, per-event unit L2 normalization",
                "detector_noise_included": False,
                "class_injectors": {
                    "GR": "none",
                    "standard-siren": "H0_true sampled from [60, 80]",
                    "qnm-21": "subdominant mode 21; amplitude ratio sampled from [0.05, 0.25]",
                    "qnm-33": "subdominant mode 33; amplitude ratio sampled from [0.10, 0.35]",
                    "pn-deformation": "delta_phi_hat sampled from [0.001, 0.01]",
                    "extra-dimensions": "Layer5BeyondGRInjector.inject_extra_dimensions_leakage with n_extra_dims=1",
                    "scalar-tensor": "omega_BD_eff sampled from [40000, 80000]",
                    "graviton-mass": "fixed mg_eV=1.27e-23",
                    "chern-simons": "kappa_gpc sampled from [0.5, 1.2]",
                    "liv-alpha2": "alpha=2.0, phase_ref_rad sampled from [0.03, 0.10]",
                    "liv-alpha4": "alpha=4.0, phase_ref_rad sampled from [0.03, 0.10]",
                    "loop-quantum-gravity": "Layer6HorizonTopologyInjector.inject_lqg_discrete_spectrum with area_quantum=0.1",
                    "gup": "beta_gup sampled from [0.03, 0.10]",
                },
            },
            "pca": {
                "n_components": PCA_COMPONENTS,
                "svd_solver": PCA_SOLVER,
                "random_state": PCA_RANDOM_STATE,
                "fit_split": "train_only",
                "fit_input": "X_train_raw",
                "fit_shape": list(X_train_raw.shape),
                "samples_per_class": config["train_per_class"],
                "feature_range": [-1.0, 1.0],
                "scaler_fit_split": "train_only",
                "components_sign_aligned_to_historical_prefix": signs.tolist(),
            },
            "runtime_versions": current_versions,
            "historical_prefix_check": prefix_check,
            "rank_diagnostics": rank_report,
            "array_hashes": array_hashes,
            "file_hashes": file_hashes,
            "historical_source_hashes": sources["source_hashes"],
            "generator_source_hashes": _source_hashes(),
            "historical_dataset_hashes": old_metadata["dataset_hashes"],
            "runtime": runtime,
            "final_test": {
                "generated": False,
                "loaded": False,
                "used_for_fit": False,
            },
        }
        report_text = _markdown_report(metadata, variance_report)
        (stage_dir / "pca64_audit_report.md").write_text(
            report_text, encoding="utf-8"
        )
        file_hashes["pca64_audit_report.md"] = _file_sha256(
            stage_dir / "pca64_audit_report.md"
        )
        metadata["file_hashes"] = file_hashes
        _atomic_json(stage_dir / "dataset_metadata.json", metadata)
        file_hashes["dataset_metadata.json"] = _file_sha256(
            stage_dir / "dataset_metadata.json"
        )
        manifest = {
            "schema_version": 1,
            "dataset_id": "QNIM_Q1_6_DATASET_V2_PCA64",
            "status": "immutable_complete",
            "protocol": {
                "historical_source_dataset": "qnim_q1_6_dataset_v1",
                "dataset_seed": DATASET_SEED,
                "pca_components": PCA_COMPONENTS,
                "pca_solver": PCA_SOLVER,
                "pca_random_state": PCA_RANDOM_STATE,
                "fit_split": "train_only",
                "validation_fit": False,
                "test_created_or_loaded": False,
            },
            "historical_hashes": old_metadata["dataset_hashes"],
            "prefix_integrity_check": prefix_check,
            "array_hashes": array_hashes,
            "file_sha256": file_hashes,
        }
        _atomic_json(stage_dir / "dataset_manifest.json", manifest)
        stage_times["persistence_and_reporting_seconds"] = (
            time.perf_counter() - phase
        )

    runtime["total_seconds"] = time.perf_counter() - started
    runtime["peak_rss_bytes"] = memory.peak_bytes
    runtime["peak_rss_delta_bytes"] = (
        memory.peak_bytes - memory.initial_bytes
    )
    metadata["runtime"] = runtime
    (stage_dir / "pca64_audit_report.md").write_text(
        _markdown_report(metadata, variance_report), encoding="utf-8"
    )
    file_hashes["pca64_audit_report.md"] = _file_sha256(
        stage_dir / "pca64_audit_report.md"
    )
    metadata["file_hashes"] = file_hashes
    _atomic_json(stage_dir / "dataset_metadata.json", metadata)
    file_hashes["dataset_metadata.json"] = _file_sha256(
        stage_dir / "dataset_metadata.json"
    )
    file_hashes["pca64_audit_report.md"] = _file_sha256(
        stage_dir / "pca64_audit_report.md"
    )
    manifest["file_sha256"] = file_hashes
    _atomic_json(stage_dir / "dataset_manifest.json", manifest)
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing dataset directory: {OUTPUT_DIR}"
        )
    stage_dir.rename(OUTPUT_DIR)
    return metadata


def main() -> int:
    try:
        metadata = build_dataset()
    except IntegrityMismatch as exc:
        print(
            json.dumps(
                {"status": "integrity_failed", "details": exc.details},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "output_dir": str(OUTPUT_DIR),
                "prefix_check": metadata["historical_prefix_check"],
                "runtime": metadata["runtime"],
                "rank": metadata["rank_diagnostics"]["rank_by_tolerance"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
