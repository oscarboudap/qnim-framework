import pytest

from scripts.analyze_qnim_q1_6_partial import _build_analysis


def _run(model_family, arm, n_qubits, seed, balanced_accuracy):
    return {
        "status": "completed",
        "model_family": model_family,
        "arm": arm,
        "n_qubits": n_qubits,
        "n_features": n_qubits if arm == "full_system" else 4,
        "initialization_seed": seed,
        "metrics": {
            "validation": {
                "balanced_accuracy": balanced_accuracy,
                "accuracy": balanced_accuracy,
                "macro_f1": balanced_accuracy,
                "cross_entropy": 1.0,
                "class_recall": {"A": balanced_accuracy, "B": balanced_accuracy},
                "confusion_matrix": [[1, 0], [0, 1]],
            }
        },
        "timing": {"fit_and_evaluation_seconds": 2.0},
        "memory": {"rss_peak_bytes": 100},
    }


def _source_report():
    return {
        "experiment_id": "QNIM_Q1_6_SCALING_PILOT_V1",
        "status": "prepared",
        "configuration": {
            "qubit_counts": [4, 8],
            "arms": {
                "fixed_input_dimension": {"n_features": 4},
            },
            "training": {"initialization_seeds": [42, 43]},
        },
        "dataset": {
            "dataset_seed": 42,
            "class_names": ["A", "B"],
            "dataset_hashes": {"train_sha256": "train", "validation_sha256": "val"},
            "first_12_component_hashes_match_q1": True,
            "test_access": {
                "files_loaded": False,
                "projected": False,
                "evaluated": False,
                "persisted": False,
            },
        },
        "final_test": {
            "opened": False,
            "evaluated": False,
            "status": "locked_not_opened",
        },
        "runs": [
            _run("qnim", "full_system", 4, 42, 0.1),
            _run("pca_svm_rbf", "full_system", 4, 42, 0.2),
            _run("qnim", "full_system", 8, 42, 0.3),
            _run("qnim", "fixed_input_dimension", 8, 42, 0.15),
            _run("pca_svm_rbf", "full_system", 8, 42, 0.4),
            {
                **_run("qnim", "full_system", 8, 43, 0.99),
                "status": "failed",
            },
        ],
        "errors": [],
    }


def test_partial_analysis_uses_only_completed_runs_and_lists_pending():
    analysis = _build_analysis(_source_report(), "abc", "source.json")

    assert analysis["scope"]["completed_run_count"] == 5
    assert analysis["scope"]["expected_run_count"] == 10
    assert analysis["scope"]["pending_run_count"] == 5
    q8 = analysis["aggregates"]["by_model_and_arm"]["qnim:full_system"]["8"]
    assert q8["metrics"]["balanced_accuracy"]["mean"] == 0.3
    assert q8["metrics"]["balanced_accuracy"]["sample_std"] is None
    assert q8["initialization_seeds"] == [42]
    assert analysis["dataset"]["test_access"]["evaluated"] is False
    assert analysis["methodology"]["training_performed"] is False


def test_partial_analysis_pairs_svm_by_seed_and_marks_q4_alias():
    analysis = _build_analysis(_source_report(), "abc", "source.json")
    comparisons = analysis["aggregates"]["paired_balanced_accuracy_differences"]

    q4_svm = comparisons["qnim_full_minus_svm"][0]
    assert q4_svm["paired_differences"] == [{
        "seed": 42,
        "difference": pytest.approx(-0.1),
        "shared_fit_alias": False,
    }]
    q4_fixed = comparisons["qnim_full_minus_fixed"][0]
    assert q4_fixed["paired_differences"] == [{
        "seed": 42,
        "difference": 0.0,
        "shared_fit_alias": True,
    }]


def test_partial_analysis_rejects_source_with_test_access():
    report = _source_report()
    report["dataset"]["test_access"]["files_loaded"] = True

    with pytest.raises(ValueError, match="final-test data access"):
        _build_analysis(report, "abc", "source.json")
