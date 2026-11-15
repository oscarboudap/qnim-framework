"""Run a controlled GR-vs-ppE preprocessing positive control."""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _colored_noise(psd_freqs, psd, sample_rate, n_samples, rng):
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    psd_bins = np.interp(frequencies, psd_freqs, psd)
    positive = psd_bins[(frequencies >= 20.0) & (frequencies <= 1024.0)]
    median_psd = float(np.median(positive))
    if not np.isfinite(median_psd) or median_psd <= 0:
        raise ValueError("La PSD no tiene densidad positiva en la banda útil")
    scale = np.sqrt(
        np.maximum(psd_bins * sample_rate / 2.0, np.finfo(float).tiny)
    )
    white = rng.standard_normal(n_samples)
    return np.fft.irfft(np.fft.rfft(white) * scale, n=n_samples)


def _phase_recovery_error(gr, beyond_gr, delta_phi_hat, sample_rate):
    frequencies = np.fft.rfftfreq(len(gr), d=1.0 / sample_rate)
    gr_spectrum = np.fft.rfft(gr)
    ppe_spectrum = np.fft.rfft(beyond_gr)
    valid = (
        (frequencies >= 20.0)
        & (frequencies <= 512.0)
        & (np.abs(gr_spectrum) > np.max(np.abs(gr_spectrum)) * 1e-6)
    )
    observed = np.angle(ppe_spectrum[valid] * np.conj(gr_spectrum[valid]))
    expected = delta_phi_hat * (frequencies[valid] / 100.0) ** (-7.0 / 3.0)
    residual = np.angle(np.exp(1j * (observed - expected)))
    return float(np.sqrt(np.mean(residual**2)))


def _group_statistics(y_true, prediction, groups, n_bootstrap, n_permutations, seed):
    unique_groups = np.unique(groups)
    group_indices = [
        np.flatnonzero(groups == group_id) for group_id in unique_groups
    ]
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_bootstrap)
    permutation = np.empty(n_permutations)

    for iteration in range(n_bootstrap):
        chosen = rng.integers(0, len(group_indices), size=len(group_indices))
        indices = np.concatenate([group_indices[index] for index in chosen])
        bootstrap[iteration] = balanced_accuracy_score(
            y_true[indices], prediction[indices]
        )

    for iteration in range(n_permutations):
        shuffled = y_true.copy()
        for indices in group_indices:
            if rng.integers(2):
                shuffled[indices] = shuffled[indices[::-1]]
        permutation[iteration] = balanced_accuracy_score(shuffled, prediction)

    observed = balanced_accuracy_score(y_true, prediction)
    p_value = (
        1 + np.count_nonzero(permutation >= observed)
    ) / (n_permutations + 1)
    return {
        "pair_bootstrap_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "pair_swap_permutation_p": float(p_value),
        "permutations": n_permutations,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check whether preprocessing preserves a known ppE(-1PN) injection"
    )
    parser.add_argument(
        "--noise-sample",
        default="data/o3b/samples/first_train_window_1256663960.npz",
    )
    parser.add_argument("--detector", choices=("H1", "L1"), default="H1")
    parser.add_argument("--train-pairs", type=int, default=200)
    parser.add_argument("--val-pairs", type=int, default=100)
    parser.add_argument("--n-components", type=int, default=64)
    parser.add_argument("--snr-min", type=float, default=8.0)
    parser.add_argument("--snr-max", type=float, default=30.0)
    parser.add_argument("--delta-phi-min", type=float, default=0.001)
    parser.add_argument("--delta-phi-max", type=float, default=0.01)
    parser.add_argument("--mass-min", type=float, default=10.0)
    parser.add_argument("--mass-max", type=float, default=80.0)
    parser.add_argument("--distance-min", type=float, default=100.0)
    parser.add_argument("--distance-max", type=float, default=1000.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--permutation-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument(
        "--output",
        default="results/scaling_study/ppe_pipeline_diagnostic.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.train_pairs, args.val_pairs, args.n_components) < 1:
        raise ValueError("Tamaños y componentes deben ser positivos")
    if not 0 < args.snr_min <= args.snr_max:
        raise ValueError("El intervalo SNR debe ser positivo y ordenado")
    if not 0 < args.delta_phi_min <= args.delta_phi_max:
        raise ValueError("El intervalo delta_phi debe ser positivo y ordenado")
    if not 0 < args.mass_min <= args.mass_max:
        raise ValueError("El intervalo de masas debe ser positivo y ordenado")
    if not 0 < args.distance_min <= args.distance_max:
        raise ValueError("El intervalo de distancias debe ser positivo y ordenado")

    from src.infrastructure.ligo_noise_injection_adapter import (
        LIGONoiseInjectionAdapter,
        NoiseSegment,
    )
    from src.infrastructure.physics_sstg_adapter import PhysicsSSTGAdapter
    from src.infrastructure.stochastic_signal_generator import _inject_pn_deformation

    with np.load(args.noise_sample, allow_pickle=False) as sample:
        sample_rate = int(sample["sampling_rate_hz"])
        empirical_noise = np.asarray(sample[args.detector], dtype=float)
    if empirical_noise.ndim != 1 or not np.isfinite(empirical_noise).all():
        raise ValueError("La muestra de strain debe ser un vector finito")

    duration = len(empirical_noise) / sample_rate
    psd_freqs, psd = welch(
        empirical_noise,
        fs=sample_rate,
        nperseg=min(len(empirical_noise), max(256, sample_rate)),
        detrend="constant",
        scaling="density",
    )
    frequency_band = (psd_freqs >= 20.0) & (psd_freqs <= min(1024.0, sample_rate / 2))
    if not frequency_band.any() or not np.isfinite(psd[frequency_band]).all():
        raise ValueError("No se pudo estimar una PSD válida en la banda del detector")

    placeholder_noise = NoiseSegment(
        segment_id="diagnostic-placeholder",
        data=empirical_noise,
        sample_rate=sample_rate,
        detector=args.detector,
    )
    adapter = LIGONoiseInjectionAdapter(
        noise_segments={
            "train": [placeholder_noise],
            "val": [
                NoiseSegment("diagnostic-val-placeholder", empirical_noise, sample_rate, args.detector)
            ],
            "test": [
                NoiseSegment("diagnostic-test-placeholder", empirical_noise, sample_rate, args.detector)
            ],
        },
        sample_rate=sample_rate,
        duration=duration,
        n_components=args.n_components,
        n_pca_fit=args.n_components,
        target_snr_range=(args.snr_min, args.snr_max),
        max_classes=5,
    )
    waveform_generator = PhysicsSSTGAdapter(
        waveform_duration=duration,
        sample_rate=sample_rate,
        max_classes=5,
        m_range=(args.mass_min, args.mass_max),
        d_range=(args.distance_min, args.distance_max),
    )

    raw_by_split = {"train": [], "val": []}
    labels_by_split = {"train": [], "val": []}
    groups_by_split = {"train": [], "val": []}
    target_snr_by_split = {"train": [], "val": []}
    beta_values = []
    phase_errors = []
    val_gr_templates = []
    val_ppe_templates = []
    split_ids = {"train": 0, "val": 1}
    rng = np.random.default_rng(args.seed)

    for split, pair_count in (
        ("train", args.train_pairs),
        ("val", args.val_pairs),
    ):
        for pair_index in range(pair_count):
            pair_seed = np.random.SeedSequence(
                [args.seed, split_ids[split], pair_index]
            )
            source_seed = np.random.SeedSequence(pair_seed.entropy, spawn_key=pair_seed.spawn_key)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                gr_waveform, _, _, gr_metadata = waveform_generator._generate_one(
                    0, np.random.default_rng(source_seed), (args.snr_min, args.snr_max)
                )
            if any("PyCBC falló" in str(item.message) for item in caught):
                raise RuntimeError("PyCBC falló; el control no permite el waveform fallback")

            beta = float(rng.uniform(args.delta_phi_min, args.delta_phi_max))
            frequencies = np.fft.rfftfreq(len(gr_waveform), d=1.0 / sample_rate)
            ppe_spectrum = _inject_pn_deformation(
                np.fft.rfft(gr_waveform),
                frequencies,
                M_total_msun=gr_metadata["m1"] + gr_metadata["m2"],
                delta_phi_hat=beta,
            )
            ppe_waveform = np.fft.irfft(ppe_spectrum, n=len(gr_waveform))
            phase_errors.append(
                _phase_recovery_error(
                    gr_waveform, ppe_waveform, beta, sample_rate
                )
            )
            beta_values.append(beta)
            noise_rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, split_ids[split], pair_index, 1])
            )
            colored_noise = _colored_noise(
                psd_freqs, psd, sample_rate, len(empirical_noise), noise_rng
            )
            segment = NoiseSegment(
                segment_id=f"{split}-{pair_index}",
                data=colored_noise,
                sample_rate=sample_rate,
                detector=args.detector,
            )
            target_snr = float(rng.uniform(args.snr_min, args.snr_max))
            gr_reference = np.asarray(
                gr_metadata.pop("_normalization_reference_waveform", gr_waveform)
            )
            gr_noisy, _, gr_template_white = adapter._inject_and_whiten(
                gr_waveform, segment, target_snr, gr_reference
            )
            ppe_noisy, _, _ = adapter._inject_and_whiten(
                ppe_waveform, segment, target_snr, gr_reference
            )
            if split == "val":
                noise_white = gr_noisy - gr_template_white
                val_gr_templates.append(gr_template_white)
                val_ppe_templates.append(ppe_noisy - noise_white)
            raw_by_split[split].extend((gr_noisy, ppe_noisy))
            labels_by_split[split].extend((0, 1))
            groups_by_split[split].extend((pair_index, pair_index))
            target_snr_by_split[split].extend((target_snr, target_snr))

    X_train_raw = np.stack(raw_by_split["train"])
    X_val_raw = np.stack(raw_by_split["val"])
    y_train = np.asarray(labels_by_split["train"], dtype=int)
    y_val = np.asarray(labels_by_split["val"], dtype=int)
    val_groups = np.asarray(groups_by_split["val"])
    n_components = min(
        args.n_components,
        len(X_train_raw) - 1,
        X_train_raw.shape[1],
    )
    if n_components != args.n_components:
        raise ValueError("No hay suficientes muestras de train para n_components")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=args.seed)
    pca.fit(X_train_raw)
    train_pca = pca.transform(X_train_raw)
    val_pca = pca.transform(X_val_raw)
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    X_train = np.clip(scaler.fit_transform(train_pca), -1.0, 1.0)
    X_val = np.clip(scaler.transform(val_pca), -1.0, 1.0)

    model_parameters = dict(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    predictions = {}
    for feature_set, train_features, val_features in (
        ("whitened_strain", X_train_raw, X_val_raw),
        ("pca_scaled", X_train, X_val),
    ):
        model = ExtraTreesClassifier(**model_parameters)
        model.fit(train_features, y_train)
        prediction = model.predict(val_features)
        predictions[feature_set] = prediction
    validation_metrics = {}
    for feature_set, prediction in predictions.items():
        validation_metrics[feature_set] = {
            "accuracy": float(accuracy_score(y_val, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y_val, prediction)),
            "class_counts": np.bincount(y_val, minlength=2).tolist(),
            "prediction_counts": np.bincount(prediction, minlength=2).tolist(),
            "validation_pairs": args.val_pairs,
            **_group_statistics(
                y_val,
                prediction,
                val_groups,
                args.bootstrap_iterations,
                args.permutation_iterations,
                args.seed + 1,
            ),
        }
    oracle_predictions = np.empty(len(y_val), dtype=int)
    for pair_index, (gr_template, ppe_template) in enumerate(
        zip(val_gr_templates, val_ppe_templates)
    ):
        for offset in range(2):
            sample_index = 2 * pair_index + offset
            query = X_val_raw[sample_index]
            log_likelihood_ratio = (
                2.0 * np.dot(query, ppe_template - gr_template)
                + np.dot(gr_template, gr_template)
                - np.dot(ppe_template, ppe_template)
            )
            oracle_predictions[sample_index] = int(log_likelihood_ratio > 0.0)
    oracle_metrics = {
        "accuracy": float(accuracy_score(y_val, oracle_predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_val, oracle_predictions)
        ),
        "prediction_counts": np.bincount(oracle_predictions, minlength=2).tolist(),
        "validation_pairs": args.val_pairs,
        "uses_exact_validation_source_parameters_and_delta_phi": True,
        "deployable": False,
        **_group_statistics(
            y_val,
            oracle_predictions,
            val_groups,
            args.bootstrap_iterations,
            args.permutation_iterations,
            args.seed + 1,
        ),
    }
    additional_classifiers = {}
    for feature_set, train_features, val_features in (
        ("whitened_strain", X_train_raw, X_val_raw),
        ("pca_scaled", X_train, X_val),
    ):
        models = {
            "logistic": make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=args.seed,
                ),
            ),
            "svc_rbf": make_pipeline(
                StandardScaler(),
                SVC(class_weight="balanced"),
            ),
        }
        for model_name, model in models.items():
            model.fit(train_features, y_train)
            prediction = model.predict(val_features)
            additional_classifiers[f"{feature_set}_{model_name}"] = {
                "accuracy": float(accuracy_score(y_val, prediction)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_val, prediction)
                ),
                "prediction_counts": np.bincount(prediction, minlength=2).tolist(),
                "validation_pairs": args.val_pairs,
                **_group_statistics(
                    y_val,
                    prediction,
                    val_groups,
                    args.bootstrap_iterations,
                    args.permutation_iterations,
                    args.seed + 1,
                ),
            }
    snr_array = np.asarray(target_snr_by_split["val"])
    for feature_set, prediction in predictions.items():
        validation_metrics[feature_set]["balanced_accuracy_by_target_snr"] = {}
        for lower, upper in (
            (args.snr_min, 12.0),
            (12.0, 20.0),
            (20.0, args.snr_max + 1e-9),
        ):
            mask = (snr_array >= lower) & (snr_array < upper)
            key = f"{lower:g}-{upper:g}"
            validation_metrics[feature_set]["balanced_accuracy_by_target_snr"][key] = (
                float(balanced_accuracy_score(y_val[mask], prediction[mask]))
                if mask.any() else None
            )

    train_pair_deltas = [
        np.linalg.norm(X_train_raw[index + 1] - X_train_raw[index])
        for index in range(0, len(X_train_raw), 2)
    ]
    train_pca_pair_deltas = [
        np.linalg.norm(train_pca[index + 1] - train_pca[index])
        for index in range(0, len(train_pca), 2)
    ]
    val_pair_deltas = [
        np.linalg.norm(X_val_raw[index + 1] - X_val_raw[index])
        for index in range(0, len(X_val_raw), 2)
    ]
    val_pca_pair_deltas = [
        np.linalg.norm(val_pca[index + 1] - val_pca[index])
        for index in range(0, len(val_pca), 2)
    ]
    result = {
        "task": "GR_vs_ppE_minus_1PN_positive_control",
        "noise_source": str(Path(args.noise_sample)),
        "noise_model": "independent Gaussian realizations shaped by the sample PSD",
        "real_noise_segments_used_for_PSD": 1,
        "test_data_read": False,
        "preprocessing_fit_split": "train_only",
        "paired_difference_used_as_classifier_input": False,
        "signature": {
            "name": "pn-deformation",
            "delta_phi_100hz_rad_configured_range": [
                args.delta_phi_min,
                args.delta_phi_max,
            ],
            "delta_phi_100hz_rad_range": [
                float(np.min(beta_values)),
                float(np.max(beta_values)),
            ],
            "median_phase_formula_error_rad": float(np.median(phase_errors)),
        },
        "sample_rate": sample_rate,
        "duration": duration,
        "snr_range": [args.snr_min, args.snr_max],
        "mass_range_msun": [args.mass_min, args.mass_max],
        "distance_range_mpc": [args.distance_min, args.distance_max],
        "train_pairs": args.train_pairs,
        "validation_pairs": args.val_pairs,
        "n_components": n_components,
        "paired_raw_whitened_delta_norm_median": {
            "train": float(np.median(train_pair_deltas)),
            "validation": float(np.median(val_pair_deltas)),
            "diagnostic_only_oracle": True,
        },
        "paired_pca_delta_norm_median": {
            "train": float(np.median(train_pca_pair_deltas)),
            "validation": float(np.median(val_pca_pair_deltas)),
            "diagnostic_only_oracle": True,
        },
        "validation": validation_metrics,
        "additional_classifiers": additional_classifiers,
        "known_parameter_template_likelihood_diagnostic": oracle_metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
