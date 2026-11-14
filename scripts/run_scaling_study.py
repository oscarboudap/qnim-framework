"""Run the simulator-only QNIM qubit/PCA scaling matrix."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.application.use_cases.run_scaling_study_use_case import RunScalingStudyUseCase
from src.application.scaling_metrics import holm_bonferroni


def parse_args():
    parser = argparse.ArgumentParser(description="QNIM controlled scaling study (local simulator only)")
    parser.add_argument("--dataset-dir", required=True, help="Directory with train/validation .npy arrays; test is not read")
    parser.add_argument("--output-dir", default="results/scaling_study")
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Semillas de entrenamiento separadas por coma; comparten el mismo validation",
    )
    parser.add_argument("--metrics-seed", type=int, default=4242)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--permutation-iterations", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",")] if args.seeds else [args.seed]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("La lista de semillas debe ser no vacía y sin duplicados")

    output_dir = Path(args.output_dir)
    summaries = []
    for seed in seeds:
        run_output = output_dir if len(seeds) == 1 else output_dir / "runs" / f"seed_{seed}"
        summaries.append(RunScalingStudyUseCase(
            dataset_dir=args.dataset_dir,
            output_dir=run_output,
            max_iterations=args.max_iter,
            shots=args.shots,
            seed=seed,
            metrics_seed=args.metrics_seed,
            bootstrap_iterations=args.bootstrap_iterations,
            permutation_iterations=args.permutation_iterations,
        ).execute())

    dataset_hashes = {summary["dataset_sha256"] for summary in summaries}
    if len(dataset_hashes) != 1:
        raise RuntimeError("Las repeticiones no usaron el mismo dataset")

    records = [record for summary in summaries for record in summary["records"]]
    adjusted_p = holm_bonferroni([
        record["permutation_test"]["p_value"] for record in records
    ])
    for record, p_value in zip(records, adjusted_p):
        record["permutation_test"]["p_value_holm_campaign"] = float(p_value)
        record["criteria"]["holm_campaign_p_below_0_05"] = float(p_value) < 0.05

    by_configuration = {}
    for record in records:
        by_configuration.setdefault(record["id"], []).append(record)
    reproducibility = {}
    for experiment_id, experiment_records in by_configuration.items():
        scores = [record["balanced_accuracy"] for record in experiment_records]
        paired = [
            record["paired_difference_vs_reference"]["mean_difference"]
            for record in experiment_records
            if record["paired_difference_vs_reference"] is not None
        ]
        reproducibility[experiment_id] = {
            "n_seeds": len(scores),
            "balanced_accuracy_mean": float(sum(scores) / len(scores)),
            "balanced_accuracy_std": float(__import__("numpy").std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "paired_difference_mean": float(sum(paired) / len(paired)) if paired else None,
            "paired_difference_positive_seeds": sum(value > 0 for value in paired),
            "seeds": [record["seed"] for record in experiment_records],
        }

    summary = {
        "dataset_sha256": next(iter(dataset_hashes)),
        "git_commit": summaries[0]["git_commit"],
        "n_classes": summaries[0]["n_classes"],
        "random_baseline": summaries[0]["random_baseline"],
        "seeds": seeds,
        "metrics_seed": args.metrics_seed,
        "records": records,
        "reproducibility": reproducibility,
    }
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "multi_seed_results.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    csv_path = summary_dir / "multi_seed_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fields = [
            "id", "seed", "n_qubits", "pca_components", "balanced_accuracy",
            "p_value", "p_value_holm_campaign", "paired_difference_mean",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            diff = record["paired_difference_vs_reference"]
            writer.writerow({
                "id": record["id"],
                "seed": record["seed"],
                "n_qubits": record["n_qubits"],
                "pca_components": record["pca_components"],
                "balanced_accuracy": record["balanced_accuracy"],
                "p_value": record["permutation_test"]["p_value"],
                "p_value_holm_campaign": record["permutation_test"]["p_value_holm_campaign"],
                "paired_difference_mean": diff["mean_difference"] if diff else None,
            })

    print(f"Estudio guardado en {output_dir.resolve()}")
    print(f"Configuraciones x semillas completadas: {len(records)}")


if __name__ == "__main__":
    main()