import json
import sys

from scripts import run_scaling_study


class FakeStudy:
    seen = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.seen.append(kwargs)

    def execute(self):
        seed = self.kwargs["seed"]
        return {
            "dataset_sha256": "same-dataset-hash",
            "git_commit": None,
            "n_classes": 13,
            "random_baseline": 1 / 13,
            "records": [
                {
                    "id": "n12_d12",
                    "seed": seed,
                    "n_qubits": 12,
                    "pca_components": 12,
                    "balanced_accuracy": 0.2 + seed / 1000,
                    "permutation_test": {"p_value": 0.03},
                    "paired_difference_vs_reference": None,
                    "criteria": {},
                },
                {
                    "id": "n16_d12",
                    "seed": seed,
                    "n_qubits": 16,
                    "pca_components": 12,
                    "balanced_accuracy": 0.3 + seed / 1000,
                    "permutation_test": {"p_value": 0.04},
                    "paired_difference_vs_reference": {"mean_difference": 0.1},
                    "criteria": {},
                },
            ],
        }


def test_cli_runs_requested_seeds_and_writes_aggregate(tmp_path, monkeypatch):
    FakeStudy.seen = []
    monkeypatch.setattr(run_scaling_study, "RunScalingStudyUseCase", FakeStudy)
    monkeypatch.setattr(sys, "argv", [
        "run_scaling_study.py",
        "--dataset-dir", str(tmp_path / "dataset"),
        "--output-dir", str(tmp_path / "results"),
        "--seeds", "11,22",
        "--metrics-seed", "4242",
    ])

    run_scaling_study.main()

    assert [run["seed"] for run in FakeStudy.seen] == [11, 22]
    assert all(run["metrics_seed"] == 4242 for run in FakeStudy.seen)
    assert all("seed_" in str(run["output_dir"]) for run in FakeStudy.seen)
    result_path = tmp_path / "results/summary/multi_seed_results.json"
    summary = json.loads(result_path.read_text(encoding="utf-8"))
    assert summary["seeds"] == [11, 22]
    assert len(summary["records"]) == 4
    assert summary["reproducibility"]["n16_d12"]["n_seeds"] == 2
    assert summary["records"][0]["permutation_test"]["p_value_holm_campaign"] == 0.12
