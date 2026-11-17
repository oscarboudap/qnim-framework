"""Run a source-marginalized ppE phase study on O3b off-source strain."""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.signal import welch
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.prepare_noise_injection_dataset import load_o3b_segments
from src.application.population_ppe_inference import (
    infer_shared_beta,
    marginalize_source_parameter_log_likelihoods,
    phase_residual_log_likelihoods,
)
from src.infrastructure.ligo_noise_injection_adapter import _optimal_snr
from src.infrastructure.physics_sstg_adapter import _generate_base_waveform
from src.infrastructure.stochastic_signal_generator import _inject_pn_deformation


SAMPLE_RATE = 4096
DURATION = 16.0
F_LOW_HZ = 20.0
F_HIGH_HZ = 512.0
BAND_EDGES_HZ = np.array([20.0, 30.0, 40.0, 55.0, 75.0, 105.0, 150.0])
SOURCE_BANK = (
    (20.0, 10.0),
    (30.0, 10.0),
    (30.0, 20.0),
    (40.0, 10.0),
    (40.0, 20.0),
    (40.0, 30.0),
    (60.0, 20.0),
    (60.0, 40.0),
)
INJECTION_BETAS = (0.0, 0.001, 0.005)


def _wilson_interval(successes: int, trials: int) -> list[float]:
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    half_width = z * np.sqrt(
        proportion * (1.0 - proportion) / trials
        + z**2 / (4.0 * trials**2)
    ) / denominator
    return [float(center - half_width), float(center + half_width)]


def _selected_gps_windows(
    gps_ranges: dict,
    frame_inventory: dict,
    calibration_count: int,
    events_per_hypothesis: int,
) -> tuple[list[list[float]], list[list[float]]]:
    hdf_frames = [
        item for item in frame_inventory["strain"]
        if item.get("detector") == "H1" and item.get("format") == "hdf5"
    ]

    def clipped_intervals(split: str) -> list[tuple[float, float]]:
        intersections = []
        for gps_start, gps_end in gps_ranges[split]:
            for frame in hdf_frames:
                frame_start = float(frame["GPSstart"])
                frame_end = frame_start + float(frame["duration"])
                start = max(float(gps_start), frame_start)
                end = min(float(gps_end), frame_end)
                if end - start >= DURATION:
                    intersections.append((start, end))
        return sorted(set(intersections))

    def candidate_windows(
        intervals: list[tuple[float, float]], stride: float
    ) -> list[list[float]]:
        candidates = []
        for start, end in intervals:
            count = int(np.floor((end - start - DURATION) / stride)) + 1
            candidates.extend([
                [start + index * stride, start + index * stride + DURATION]
                for index in range(count)
            ])
        return candidates

    def sample_windows(
        candidates: list[list[float]], count: int, split: str
    ) -> list[list[float]]:
        if count > len(candidates):
            raise ValueError(
                f"Solo hay {len(candidates)} ventanas H1 no solapadas en {split}"
            )
        if count == 0:
            return []
        positions = np.rint(
            np.linspace(0, len(candidates) - 1, count)
        ).astype(int)
        if len(np.unique(positions)) != count:
            raise ValueError(f"No se pudieron seleccionar ventanas únicas de {split}")
        return [candidates[index] for index in positions]

    train_candidates = candidate_windows(
        clipped_intervals("train"), stride=100.0
    )
    train_windows = sample_windows(
        train_candidates, calibration_count, "train"
    )

    total_validation = events_per_hypothesis * len(INJECTION_BETAS)
    validation_candidates = candidate_windows(
        clipped_intervals("validation"), stride=DURATION
    )
    val_windows = sample_windows(
        validation_candidates, total_validation, "validation"
    )
    return train_windows, val_windows


def _build_source_bank() -> list[np.ndarray]:
    waveform_bank = []
    expected_samples = int(SAMPLE_RATE * DURATION)
    for mass1, mass2 in SOURCE_BANK:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            waveform, delta_t = _generate_base_waveform(
                mass1,
                mass2,
                distance_mpc=300.0,
                duration=DURATION,
                sample_rate=SAMPLE_RATE,
                f_lower=F_LOW_HZ,
                approximant="IMRPhenomD",
            )
        if any("PyCBC falló" in str(item.message) for item in caught):
            raise RuntimeError("Se requiere PyCBC; no se acepta el waveform fallback")
        if delta_t != 1.0 / SAMPLE_RATE or waveform.shape != (expected_samples,):
            raise RuntimeError("El generador devolvió una forma de onda incompatible")
        waveform_bank.append(np.asarray(waveform, dtype=float))
    return waveform_bank


def _phase_residual_by_band(
    observed: np.ndarray,
    template: np.ndarray,
    psd_bins: np.ndarray,
    usable: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observed_fft = np.fft.rfft(observed)
    template_fft = np.fft.rfft(template)
    phase_residuals = []
    effective_centers = []

    for low, high in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:]):
        mask = usable & (frequencies >= low) & (frequencies < high)
        if np.count_nonzero(mask) < 3:
            raise ValueError(f"No hay suficientes bins en la banda {low}-{high} Hz")
        template_power = np.abs(template_fft[mask]) ** 2 / psd_bins[mask]
        total_power = float(np.sum(template_power))
        if not np.isfinite(total_power) or total_power <= 0.0:
            raise ValueError(f"La plantilla no aporta información en {low}-{high} Hz")
        cross_power = (
            observed_fft[mask] * np.conj(template_fft[mask]) / psd_bins[mask]
        )
        phase_residuals.append(float(np.angle(np.sum(cross_power))))
        mean_phase_basis = float(
            np.sum(
                template_power
                * (frequencies[mask] / 100.0) ** (-7.0 / 3.0)
            )
            / total_power
        )
        effective_centers.append(
            100.0 * mean_phase_basis ** (-3.0 / 7.0)
        )

    return np.unwrap(phase_residuals), np.asarray(effective_centers)


def _inject_into_noise(
    noise: np.ndarray,
    waveform: np.ndarray,
    target_snr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = int(SAMPLE_RATE * DURATION)
    if noise.shape != (n_samples,) or waveform.shape != (n_samples,):
        raise ValueError(
            f"Noise y waveform deben tener {DURATION:g} s a {SAMPLE_RATE} Hz"
        )
    psd_frequencies, psd = welch(
        noise,
        fs=SAMPLE_RATE,
        nperseg=SAMPLE_RATE,
        detrend="constant",
        scaling="density",
    )
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / SAMPLE_RATE)
    psd_bins = np.interp(frequencies, psd_frequencies, psd)
    usable = (
        (frequencies >= F_LOW_HZ)
        & (frequencies <= F_HIGH_HZ)
        & np.isfinite(psd_bins)
        & (psd_bins > 0.0)
    )
    if not usable.any():
        raise ValueError("No hay bins PSD utilizables en la banda del detector")

    delta_f = SAMPLE_RATE / n_samples
    delta_t = 1.0 / SAMPLE_RATE
    reference_fft = np.fft.rfft(waveform)
    reference_snr = _optimal_snr(
        reference_fft, psd_bins, usable, delta_f, delta_t
    )
    if not np.isfinite(reference_snr) or reference_snr <= 0.0:
        raise ValueError("No se pudo calcular un SNR válido para la inyección")
    scale = target_snr / reference_snr
    observed = noise + scale * waveform
    return observed, psd_bins, usable, frequencies


def _bootstrap_detection_fraction(
    beta_grid: np.ndarray,
    event_log_likelihoods: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> dict:
    detections = 0
    for _ in range(replicates):
        indices = rng.integers(
            0, len(event_log_likelihoods), size=len(event_log_likelihoods)
        )
        result = infer_shared_beta(beta_grid, event_log_likelihoods[indices])
        low, high = result.credible_interval
        detections += int(
            result.log_bayes_factor_10 > np.log(10.0)
            and (low > 0.0 or high < 0.0)
        )
    return {
        "bootstrap_replicates": replicates,
        "fraction_with_log_bf10_gt_log10_and_interval_excluding_zero": (
            detections / replicates
        ),
        "wilson_95": _wilson_interval(detections, replicates),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer a shared ppE beta from injections into O3b off-source strain"
    )
    parser.add_argument("--calibration-events", type=int, default=32)
    parser.add_argument("--events-per-hypothesis", type=int, default=50)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument(
        "--output",
        default="results/scaling_study/strain_population_ppe_h1_16s_20261115.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.calibration_events < 8:
        raise ValueError("Se requieren al menos 8 ventanas de calibración")
    if args.events_per_hypothesis < 1 or args.bootstrap_replicates < 1:
        raise ValueError("Los tamaños de validación/bootstrap deben ser positivos")

    gps_ranges = json.loads(
        Path("data/o3b/windows/gps_ranges.json").read_text(encoding="utf-8")
    )
    frame_inventory = json.loads(
        Path("data/o3b/events/O3b_4KHZ_R1_H1.json").read_text(encoding="utf-8")
    )
    train_windows, validation_windows = _selected_gps_windows(
        gps_ranges,
        frame_inventory,
        args.calibration_events,
        args.events_per_hypothesis,
    )
    segments = load_o3b_segments(
        gps_ranges={
            "train": train_windows,
            "validation": validation_windows,
        },
        detector="H1",
        inventory_path="data/o3b/events/O3b_4KHZ_R1_H1.json",
        frame_cache="data/o3b/events",
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
    )
    if len(segments["train"]) != args.calibration_events:
        raise RuntimeError("No se cargaron todas las ventanas de calibración")
    if len(segments["val"]) != len(validation_windows):
        raise RuntimeError("No se cargaron todas las ventanas de validation")
    if segments["test"]:
        raise RuntimeError("El flujo no debe cargar ventanas de test")

    rng = np.random.default_rng(args.seed)
    waveform_bank = _build_source_bank()
    n_samples = int(SAMPLE_RATE * DURATION)
    delta_f = SAMPLE_RATE / n_samples
    delta_t = 1.0 / SAMPLE_RATE
    calibration_phase_errors = np.empty(
        (len(waveform_bank), args.calibration_events, len(BAND_EDGES_HZ) - 1)
    )
    for source_index, waveform in enumerate(waveform_bank):
        for event_index, segment in enumerate(segments["train"]):
            observed, psd_bins, usable, frequencies = _inject_into_noise(
                np.asarray(segment.data, dtype=float), waveform, target_snr=20.0
            )
            phase, _ = _phase_residual_by_band(
                observed, waveform, psd_bins, usable, frequencies
            )
            calibration_phase_errors[source_index, event_index] = phase
    phase_covariances = np.stack([
        LedoitWolf().fit(source_errors).covariance_
        for source_errors in calibration_phase_errors
    ])
    np.linalg.cholesky(phase_covariances)

    beta_grid = np.linspace(-0.01, 0.01, 401)
    beta_assignments = np.repeat(INJECTION_BETAS, args.events_per_hypothesis)
    beta_assignments = beta_assignments[rng.permutation(len(beta_assignments))]
    event_log_likelihoods = {beta: [] for beta in INJECTION_BETAS}
    event_metadata = {beta: [] for beta in INJECTION_BETAS}

    for event_index, (segment, beta_true) in enumerate(
        zip(segments["val"], beta_assignments)
    ):
        source_index = int(rng.integers(len(waveform_bank)))
        gr_waveform = waveform_bank[source_index]
        mass1, mass2 = SOURCE_BANK[source_index]
        frequencies = np.fft.rfftfreq(n_samples, d=delta_t)
        ppe_fft = _inject_pn_deformation(
            np.fft.rfft(gr_waveform),
            frequencies,
            M_total_msun=mass1 + mass2,
            delta_phi_hat=float(beta_true),
        )
        ppe_waveform = np.fft.irfft(ppe_fft, n=n_samples)
        target_snr = float(rng.uniform(8.0, 30.0))
        observed, psd_bins, usable, frequencies = _inject_into_noise(
            np.asarray(segment.data, dtype=float),
            ppe_waveform,
            target_snr=target_snr,
        )

        candidate_residuals = []
        candidate_frequencies = []
        for candidate_waveform in waveform_bank:
            phase, effective_centers = _phase_residual_by_band(
                observed,
                candidate_waveform,
                psd_bins,
                usable,
                frequencies,
            )
            candidate_residuals.append(phase)
            candidate_frequencies.append(effective_centers)
        event_covariances = phase_covariances * (20.0 / target_snr) ** 2
        source_log_likelihoods = phase_residual_log_likelihoods(
            beta_grid,
            np.asarray(candidate_residuals),
            np.asarray(candidate_frequencies),
            event_covariances,
        )
        marginalized_log_likelihood = marginalize_source_parameter_log_likelihoods(
            source_log_likelihoods[np.newaxis, :, :]
        )[0]
        event_log_likelihoods[float(beta_true)].append(marginalized_log_likelihood)
        event_metadata[float(beta_true)].append({
            "event_index": event_index,
            "gps_start": float(segment.gps_start),
            "target_snr": target_snr,
            "injected_source_bank_index": source_index,
        })

    results = {}
    for beta_true in INJECTION_BETAS:
        likelihoods = np.asarray(event_log_likelihoods[beta_true])
        inference = infer_shared_beta(beta_grid, likelihoods)
        results[str(beta_true)] = {
            "n_events": len(likelihoods),
            "injected_beta": beta_true,
            "posterior_mean": inference.posterior_mean,
            "credible_interval_95": list(inference.credible_interval),
            "log_bayes_factor_10": inference.log_bayes_factor_10,
            "bootstrap_detection_stability": _bootstrap_detection_fraction(
                beta_grid,
                likelihoods,
                args.bootstrap_replicates,
                rng,
            ),
            "events": event_metadata[beta_true],
        }

    output = {
        "study": "H1 O3b off-source strain injections with phase-only source marginalization",
        "noise_data": "local GWOSC O3b H1 strain; train calibration and disjoint validation windows only",
        "test_windows_loaded": False,
        "waveform_approximant": "IMRPhenomD, nonspinning",
        "sample_rate_hz": SAMPLE_RATE,
        "duration_seconds": DURATION,
        "source_prior": {
            "type": "uniform discrete prior over the following (m1,m2) bank in solar masses",
            "bank": [list(pair) for pair in SOURCE_BANK],
            "distance_mpc_for_bank_waveforms": 300.0,
            "spins_included": False,
            "amplitude_phase_time_treatment": "phase-only data likelihood; constant and linear-frequency phase terms profiled",
        },
        "phase_bands_hz": [
            [float(low), float(high)]
            for low, high in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:])
        ],
        "phase_covariance_estimation": {
            "method": "Per-source Ledoit-Wolf shrinkage covariance of train-only GR injection phase residuals",
            "calibration_events": args.calibration_events,
            "calibration_target_snr": 20.0,
            "validation_covariance_scaling": "(20/target_snr)^2",
            "matrix_rad2_by_source_bank": phase_covariances.tolist(),
        },
        "beta_prior_range": [float(beta_grid[0]), float(beta_grid[-1])],
        "seed": args.seed,
        "validation_events_per_hypothesis": args.events_per_hypothesis,
        "bootstrap_replicates": args.bootstrap_replicates,
        "results_by_injected_beta": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Resultados de strain guardados en {output_path.resolve()}")
    for beta_true, result in results.items():
        print(
            f"beta={beta_true}: n={result['n_events']}, "
            f"posterior mean={result['posterior_mean']:.6g}, "
            f"CI95={result['credible_interval_95']}, "
            f"logBF10={result['log_bayes_factor_10']:.4f}, "
            f"bootstrap detection="
            f"{result['bootstrap_detection_stability']['fraction_with_log_bf10_gt_log10_and_interval_excluding_zero']:.3f}"
        )


if __name__ == "__main__":
    main()
