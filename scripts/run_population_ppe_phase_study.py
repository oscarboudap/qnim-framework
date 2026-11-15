"""Run a reproducible synthetic scaling study for shared ppE phase inference."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.application.population_ppe_inference import (
    infer_shared_beta,
    phase_residual_log_likelihoods,
)


FREQUENCY_CENTERS_HZ = np.array(
    [25.0, 40.0, 60.0, 90.0, 140.0, 220.0, 350.0, 500.0]
)
EVENT_COUNTS = (1, 4, 16, 64, 128, 256)
BAND_COUNTS = (3, 5, 8)
INJECTED_BETAS = (0.0, 0.001, 0.005)
SNR_RANGE = (8.0, 30.0)
DETECTION_LOG_BF_THRESHOLD = math.log(10.0)


def _wilson_interval(successes: int, trials: int) -> list[float]:
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z**2 / (4.0 * trials**2)
    ) / denominator
    return [center - half_width, center + half_width]


def _run_cell(
    rng: np.random.Generator,
    n_events: int,
    frequencies: np.ndarray,
    beta_true: float,
    beta_grid: np.ndarray,
    replicates: int,
) -> dict:
    basis = (frequencies / 100.0) ** (-7.0 / 3.0)
    frequency_coordinate = frequencies / 100.0
    covariance_base = np.eye(len(frequencies))
    detections = 0
    covered = 0
    interval_widths = np.empty(replicates)
    log_bayes_factors = np.empty(replicates)

    for replicate in range(replicates):
        snr = rng.uniform(*SNR_RANGE, size=n_events)
        phase_sigma = 1.0 / snr
        residuals = (
            beta_true * basis
            + rng.uniform(-0.5, 0.5, size=(n_events, 1))
            + rng.uniform(-0.1, 0.1, size=(n_events, 1))
            * frequency_coordinate
            + rng.normal(
                0.0, phase_sigma[:, None], size=(n_events, len(frequencies))
            )
        )
        covariances = (
            phase_sigma[:, None, None] ** 2 * covariance_base[None, :, :]
        )
        event_log_likelihoods = phase_residual_log_likelihoods(
            beta_grid, residuals, frequencies, covariances
        )
        inference = infer_shared_beta(beta_grid, event_log_likelihoods)
        low, high = inference.credible_interval
        covered += int(low <= beta_true <= high)
        interval_widths[replicate] = high - low
        log_bayes_factors[replicate] = inference.log_bayes_factor_10
        detections += int(
            inference.log_bayes_factor_10 > DETECTION_LOG_BF_THRESHOLD
            and (low > 0.0 or high < 0.0)
        )

    return {
        "n_events": n_events,
        "n_bands": len(frequencies),
        "frequency_centers_hz": frequencies.tolist(),
        "beta_true": beta_true,
        "replicates": replicates,
        "detection_rule": "log_bayes_factor_10 > log(10) and 95% interval excludes zero",
        "detection_fraction": detections / replicates,
        "detection_fraction_wilson_95": _wilson_interval(detections, replicates),
        "credible_interval_95_coverage": covered / replicates,
        "median_credible_interval_width": float(np.median(interval_widths)),
        "median_log_bayes_factor_10": float(np.median(log_bayes_factors)),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Monte Carlo scaling study for a shared ppE(-1PN) coefficient"
    )
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument(
        "--output",
        default="results/scaling_study/population_ppe_phase_scaling_20261115.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.replicates < 1:
        raise ValueError("--replicates debe ser positivo")

    rng = np.random.default_rng(args.seed)
    beta_grid = np.linspace(-0.01, 0.01, 401)
    results = []
    for beta_true in INJECTED_BETAS:
        for n_events in EVENT_COUNTS:
            for n_bands in BAND_COUNTS:
                results.append(
                    _run_cell(
                        rng,
                        n_events,
                        FREQUENCY_CENTERS_HZ[:n_bands],
                        beta_true,
                        beta_grid,
                        args.replicates,
                    )
                )

    output = {
        "study": "shared_ppE_minus_1PN_phase_inference_synthetic_scaling",
        "data_kind": "synthetic phase residuals; not detector strain or an observational result",
        "seed": args.seed,
        "replicates_per_cell": args.replicates,
        "snr_range": list(SNR_RANGE),
        "phase_noise_model": "independent Gaussian phase errors with sigma=1/SNR per band",
        "event_phase_offset": "uniform(-0.5, 0.5) radians",
        "event_linear_frequency_offset": "uniform(-0.1, 0.1) * (f/100 Hz)",
        "beta_prior_range": [float(beta_grid[0]), float(beta_grid[-1])],
        "event_counts": list(EVENT_COUNTS),
        "band_counts": list(BAND_COUNTS),
        "all_frequency_centers_hz": FREQUENCY_CENTERS_HZ.tolist(),
        "injected_beta_values": list(INJECTED_BETAS),
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Resultados guardados en {output_path.resolve()}")
    print("beta_true  n_events  n_bands  detection_fraction  coverage  median_logBF10")
    for result in results:
        print(
            f"{result['beta_true']:>9.3g}  "
            f"{result['n_events']:>8}  "
            f"{result['n_bands']:>7}  "
            f"{result['detection_fraction']:>18.3f}  "
            f"{result['credible_interval_95_coverage']:>8.3f}  "
            f"{result['median_log_bayes_factor_10']:>14.3f}"
        )


if __name__ == "__main__":
    main()
