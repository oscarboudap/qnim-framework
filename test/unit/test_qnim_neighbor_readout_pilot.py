import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import run_qnim_neighbor_readout_pilot as pilot
from src.infrastructure.qiskit_vqc_trainer import (
    _qubit_readout_features,
    _readout_feature_count,
)


def _small_dataset():
    rng = np.random.default_rng(2026)
    X_train = rng.normal(size=(65, 16))
    y_train = np.repeat(np.arange(13), 5)
    X_val = rng.normal(loc=5.0, size=(13, 16))
    y_val = np.arange(13)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }


def _dataset_info():
    return {
        "full_split_hashes": {"train_sha256": "a" * 64, "validation_sha256": "b" * 64},
        "first_12_component_hashes": {"train_sha256": "c" * 64, "validation_sha256": "d" * 64},
        "dataset_manifest_sha256": pilot.APPROVED_DATASET_MANIFEST_SHA256,
        "dataset_manifest": "qnim_neighbor_readout_pilot_v1_dataset_manifest.json",
        "class_names": json.loads(
            pilot.DATASET_MANIFEST_PATH.read_text(encoding="utf-8")
        )["class_names"],
        "shapes": {"X_train": [65, 16], "y_train": [65], "X_val": [13, 16], "y_val": [13]},
    }


def _install_fake_inputs(monkeypatch, tmp_path):
    protocol_path = tmp_path / "approved_protocol.json"
    protocol_path.write_bytes(pilot.PROTOCOL_PATH.read_bytes())
    dataset_manifest_path = tmp_path / "approved_dataset_manifest.json"
    dataset_manifest_path.write_bytes(pilot.DATASET_MANIFEST_PATH.read_bytes())
    dataset = _small_dataset()
    monkeypatch.setattr(
        pilot, "load_cached_dataset",
        lambda *_: (dataset, _dataset_info()),
    )
    return protocol_path, dataset_manifest_path, dataset


def _fake_fit_calls(monkeypatch):
    calls = []

    def fake_fit(dataset, config):
        calls.append(pilot.run_key(config))
        return {
            "status": "completed",
            "timing": {"wall_clock_elapsed_seconds": 0.01},
            "memory": {"rss_peak_bytes": 100},
            "metrics": {"validation": {"balanced_accuracy": 0.5}},
        }

    monkeypatch.setattr(pilot, "_fit_one", fake_fit)
    return calls


def test_execution_matrix_is_exact_and_deduplicates_q4_fixed_input():
    assert [
        (row["arm"], row["n_qubits"], row["pca_components"])
        for row in pilot.QNIM_MATRIX
    ] == [
        ("full_system", 4, 4),
        ("full_system", 8, 8),
        ("full_system", 12, 12),
        ("fixed_input_dimension", 8, 4),
        ("fixed_input_dimension", 12, 4),
    ]
    plan = pilot.build_execution_plan()
    assert len(plan) == 24
    assert sum(row["model_family"] == "qnim" for row in plan) == 15
    assert sum(row["model_family"] == "pca_svm_rbf" for row in plan) == 9
    assert len({pilot.run_key(row) for row in plan}) == 24
    assert sum(
        row["model_family"] == "qnim"
        and row["arm"] == "fixed_input_dimension"
        and row["n_qubits"] == 4
        for row in plan
    ) == 0


def test_single_qnim_filter_selects_exactly_one_supported_run():
    plan = pilot.build_execution_plan(
        arm="full_system", n_qubits=4, seed=43
    )
    assert len(plan) == 1
    assert plan[0]["model_family"] == "qnim"
    assert plan[0]["arm"] == "full_system"
    assert plan[0]["n_qubits"] == 4
    assert plan[0]["initialization_seed"] == 43
    with pytest.raises(ValueError, match="supplied together"):
        pilot.build_execution_plan(arm="full_system")
    with pytest.raises(ValueError, match="same fit"):
        pilot.build_execution_plan(
            arm="fixed_input_dimension", n_qubits=4, seed=42
        )


def test_immutable_dataset_manifest_is_hash_pinned_and_linked_to_protocol():
    manifest, _, manifest_hash = pilot._read_dataset_manifest(
        pilot.DATASET_MANIFEST_PATH, pilot.APPROVED_PROTOCOL_SHA256
    )
    assert manifest_hash == pilot.APPROVED_DATASET_MANIFEST_SHA256
    assert manifest["train_validation_hashes"] == {
        "train_sha256": "08c1d2d69925aa08b3265e661d405c75ecafba16af16c5b8bf974b7f11c8e275",
        "validation_sha256": "1de635e0042ddf55c593ec26e27b2bb458b389dea14b32df054c62b4560b24c1",
    }


def test_changed_dataset_manifest_is_rejected(tmp_path):
    changed = tmp_path / "manifest.json"
    content = pilot.DATASET_MANIFEST_PATH.read_text(encoding="utf-8").replace(
        "approved_immutable", "altered"
    )
    changed.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        pilot._read_dataset_manifest(changed, pilot.APPROVED_PROTOCOL_SHA256)


def test_neighbor_readout_dimension_and_order_follow_qiskit_little_endian():
    assert [_readout_feature_count(qubits, pilot.READOUT_MODE) for qubits in (4, 8, 12)] == [
        7, 15, 23
    ]
    features = _qubit_readout_features(
        {"000": 32, "101": 32}, 3, "marginals_zz_neighbors"
    )
    np.testing.assert_allclose(features, [0.5, 1.0, 0.5, 0.0, 0.0])


def test_existing_readout_modes_keep_their_dimensions():
    assert [_readout_feature_count(4, mode) for mode in ("marginals", "marginals_zz")] == [
        4, 10
    ]


def test_batch_size_is_passed_to_the_training_implementation():
    trainer = pilot._build_trainer()
    assert trainer.training_batch_size == 32
    assert trainer.random_seed == 42
    assert trainer.readout_mode == "marginals_zz_neighbors"


def _write_integrity_dataset(path, arrays):
    path.mkdir(parents=True)
    for name, array in arrays.items():
        np.save(path / f"{name}.npy", array, allow_pickle=False)
    hashes = {
        "train_sha256": pilot.split_sha256(arrays["X_train"], arrays["y_train"]),
        "validation_sha256": pilot.split_sha256(arrays["X_val"], arrays["y_val"]),
    }
    prefix_hashes = {
        "train_sha256": pilot.split_sha256(arrays["X_train"][:, :12], arrays["y_train"]),
        "validation_sha256": pilot.split_sha256(arrays["X_val"][:, :12], arrays["y_val"]),
    }
    class_names = json.loads(
        pilot.DATASET_MANIFEST_PATH.read_text(encoding="utf-8")
    )["class_names"]
    dataset_manifest = {
        "expected_shapes": {
            name: list(array.shape) for name, array in arrays.items()
        },
        "train_validation_hashes": hashes,
        "historical_q1_first_12_component_hashes": prefix_hashes,
        "pca_prefix_check": {"components": 12},
        "class_names": class_names,
    }
    (path / "dataset_metadata.json").write_text(json.dumps({
        "dataset_hashes": hashes,
        "class_names": class_names,
    }), encoding="utf-8")
    return dataset_manifest


def test_dataset_hash_validation_checks_full_and_historical_12_component_prefix(
    tmp_path, monkeypatch
):
    arrays = _small_dataset()
    dataset_manifest = _write_integrity_dataset(tmp_path / "dataset", arrays)
    loaded, info = pilot.load_cached_dataset(
        tmp_path / "dataset", dataset_manifest, "f" * 64
    )
    assert info["full_split_hashes"] == dataset_manifest["train_validation_hashes"]
    assert info["first_12_component_hashes"] == dataset_manifest["historical_q1_first_12_component_hashes"]
    np.testing.assert_array_equal(loaded["X_train"], arrays["X_train"])


def test_dataset_hash_mismatch_rejects_cached_data_before_training(tmp_path, monkeypatch):
    arrays = _small_dataset()
    dataset_manifest = _write_integrity_dataset(tmp_path / "dataset", arrays)
    arrays["X_val"][0, 0] += 1
    np.save(tmp_path / "dataset" / "X_val.npy", arrays["X_val"], allow_pickle=False)
    with pytest.raises(ValueError, match="hashes"):
        pilot.load_cached_dataset(
            tmp_path / "dataset", dataset_manifest, "f" * 64
        )


def test_dataset_loader_never_requests_final_test_files(tmp_path, monkeypatch):
    arrays = _small_dataset()
    dataset_manifest = _write_integrity_dataset(tmp_path / "dataset", arrays)
    real_load = np.load
    loaded_names = []

    def guarded_load(file, *args, **kwargs):
        name = Path(file).name
        loaded_names.append(name)
        assert "test" not in name.lower()
        return real_load(file, *args, **kwargs)

    monkeypatch.setattr(pilot.np, "load", guarded_load)
    pilot.load_cached_dataset(
        tmp_path / "dataset", dataset_manifest, "f" * 64
    )
    assert set(loaded_names) == {"X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"}


def test_svm_never_predicts_on_train_or_fits_on_validation(monkeypatch):
    import sklearn.pipeline
    import sklearn.preprocessing
    import sklearn.svm

    dataset = _small_dataset()
    fitted = {}
    predicted_inputs = []

    class FakeEstimator:
        classes_ = np.arange(13)
        n_support_ = np.ones(13, dtype=int)
        support_vectors_ = np.ones((13, 4))
        dual_coef_ = np.ones((12, 13))
        intercept_ = np.ones(12)

    class FakePipeline:
        def __init__(self):
            self.steps = [type("Scaler", (), {"mean_": None})(), FakeEstimator()]

        def fit(self, X, y):
            fitted["X"] = X.copy()
            fitted["y"] = y.copy()
            self.steps[0].mean_ = X.mean(axis=0)
            return self

        def predict_proba(self, X):
            predicted_inputs.append(X.copy())
            return np.full((len(X), 13), 1 / 13)

        def __getitem__(self, index):
            return self.steps[index]

    monkeypatch.setattr(sklearn.pipeline, "make_pipeline", lambda *args: FakePipeline())
    monkeypatch.setattr(sklearn.preprocessing, "StandardScaler", lambda: object())
    monkeypatch.setattr(sklearn.svm, "SVC", lambda **kwargs: object())
    config = {
        "model_family": "pca_svm_rbf",
        "arm": "full_system",
        "n_qubits": 4,
        "pca_components": 4,
        "initialization_seed": 42,
    }

    result = pilot._fit_svm(dataset, config)
    np.testing.assert_array_equal(fitted["X"], dataset["X_train"][:, :4])
    np.testing.assert_array_equal(fitted["y"], dataset["y_train"])
    assert len(predicted_inputs) == 1
    np.testing.assert_array_equal(predicted_inputs[0], dataset["X_val"][:, :4])
    assert result["metrics"]["training"] is None
    assert result["classifier"]["random_state"] == 42
    assert result["classifier"]["scaler_fit_sample_count"] == len(dataset["y_train"])


def test_atomic_checkpoint_keeps_previous_file_if_replace_fails(tmp_path, monkeypatch):
    checkpoint = tmp_path / "results.json"
    checkpoint.write_text('{"previous": true}\n', encoding="utf-8")

    def fail_replace(source, target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(pilot.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        pilot._atomic_json(checkpoint, {"new": True})
    assert checkpoint.read_text(encoding="utf-8") == '{"previous": true}\n'
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_initial_checkpoint_is_prepared_and_references_both_immutable_manifests(
    tmp_path
):
    protocol_content = pilot.PROTOCOL_PATH.read_bytes()
    protocol_hash = hashlib.sha256(protocol_content).hexdigest()
    manifest_content = pilot.DATASET_MANIFEST_PATH.read_bytes()
    dataset_info = _dataset_info()
    results, manifest = pilot._init_state(
        tmp_path / "prepared",
        pilot.build_execution_plan(),
        protocol_content,
        protocol_hash,
        manifest_content,
        pilot.APPROVED_DATASET_MANIFEST_SHA256,
        dataset_info,
        None,
    )
    assert results["status"] == manifest["status"] == "prepared"
    assert manifest["dataset_manifest_sha256"] == pilot.APPROVED_DATASET_MANIFEST_SHA256
    assert (tmp_path / "prepared" / "protocol.json").read_bytes() == protocol_content
    assert (tmp_path / "prepared" / "dataset_manifest.json").read_bytes() == manifest_content


def test_single_filter_checkpoints_only_the_selected_qnim_fit(tmp_path, monkeypatch):
    protocol_path, dataset_manifest_path, _ = _install_fake_inputs(monkeypatch, tmp_path)
    calls = _fake_fit_calls(monkeypatch)
    output_dir = tmp_path / "single"
    assert pilot.run_pilot(
        arm="fixed_input_dimension",
        n_qubits=8,
        seed=44,
        protocol_path=protocol_path,
        dataset_manifest_path=dataset_manifest_path,
        output_dir=output_dir,
    ) == 0
    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert len(calls) == len(results["runs"]) == 1
    assert results["runs"][0]["model_family"] == "qnim"
    assert results["runs"][0]["arm"] == "fixed_input_dimension"
    assert results["runs"][0]["n_qubits"] == 8
    assert results["runs"][0]["initialization_seed"] == 44
    completed_hash = results["runs"][0]["record_sha256"]
    assert pilot.run_pilot(
        resume=True,
        arm="fixed_input_dimension",
        n_qubits=8,
        seed=44,
        protocol_path=protocol_path,
        dataset_manifest_path=dataset_manifest_path,
        output_dir=output_dir,
    ) == 0
    after_resume = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert calls == [results["runs"][0]["run_key"]]
    assert after_resume["runs"][0]["record_sha256"] == completed_hash


def test_checkpoint_persists_each_run_and_resume_skips_completed(
    tmp_path, monkeypatch
):
    protocol_path, dataset_manifest_path, _ = _install_fake_inputs(monkeypatch, tmp_path)
    calls = _fake_fit_calls(monkeypatch)
    output_dir = tmp_path / "output"
    assert pilot.run_pilot(
        max_runs=1, protocol_path=protocol_path,
        dataset_manifest_path=dataset_manifest_path, output_dir=output_dir
    ) == 0
    first = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert first["runs"][0]["status"] == "completed"
    assert (output_dir / "protocol.json").read_bytes() == protocol_path.read_bytes()
    assert (output_dir / "dataset_manifest.json").read_bytes() == dataset_manifest_path.read_bytes()
    manifest = json.loads((output_dir / "execution_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "partial"
    assert first["status"] == "partial"
    assert manifest["dataset_manifest_sha256"] == pilot.APPROVED_DATASET_MANIFEST_SHA256
    assert manifest["results_file_sha256"] == hashlib.sha256(
        (output_dir / "results.json").read_bytes()
    ).hexdigest()

    first_result_hash = first["runs"][0]["record_sha256"]
    assert pilot.run_pilot(
        resume=True, max_runs=1, protocol_path=protocol_path,
        dataset_manifest_path=dataset_manifest_path, output_dir=output_dir
    ) == 0
    second = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert calls == [first["runs"][0]["run_key"], second["runs"][1]["run_key"]]
    assert len(second["runs"][0]["attempts"]) == 1
    assert second["runs"][0]["record_sha256"] == first_result_hash
    assert second["runs"][1]["status"] == "completed"


def test_resume_rejects_tampered_result_hash(tmp_path, monkeypatch):
    protocol_path, dataset_manifest_path, _ = _install_fake_inputs(monkeypatch, tmp_path)
    _fake_fit_calls(monkeypatch)
    output_dir = tmp_path / "output"
    pilot.run_pilot(
        max_runs=1, protocol_path=protocol_path,
        dataset_manifest_path=dataset_manifest_path, output_dir=output_dir
    )
    results_path = output_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results["runs"][0]["metrics"]["tampered"] = True
    results_path.write_text(json.dumps(results), encoding="utf-8")
    with pytest.raises(ValueError, match="result hash"):
        pilot.run_pilot(
            resume=True, protocol_path=protocol_path,
            dataset_manifest_path=dataset_manifest_path, output_dir=output_dir
        )


def test_interrupted_execution_preserves_completed_run_and_records_active_run(
    tmp_path, monkeypatch
):
    protocol_path, dataset_manifest_path, _ = _install_fake_inputs(monkeypatch, tmp_path)
    calls = []

    def complete_then_interrupt(dataset, config):
        calls.append(pilot.run_key(config))
        status = "completed" if len(calls) == 1 else "interrupted"
        return {
            "status": status,
            "timing": {"wall_clock_elapsed_seconds": 0.01},
            "memory": {"rss_peak_bytes": 100},
            "error": {"type": "KeyboardInterrupt", "message": "stopped"} if status == "interrupted" else None,
        }

    monkeypatch.setattr(pilot, "_fit_one", complete_then_interrupt)
    output_dir = tmp_path / "output"
    assert pilot.run_pilot(
        protocol_path=protocol_path,
        dataset_manifest_path=dataset_manifest_path,
        output_dir=output_dir,
    ) == 130
    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "execution_manifest.json").read_text(encoding="utf-8"))
    assert results["runs"][0]["status"] == "completed"
    assert results["runs"][1]["status"] == "interrupted"
    assert results["runs"][1]["attempts"][0]["status"] == "interrupted"
    assert manifest["status"] == "partial"
    assert manifest["active_run"] is None


def test_fit_one_records_keyboard_interrupt_without_losing_timing(monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(pilot, "_fit_qnim", interrupt)
    result = pilot._fit_one(_small_dataset(), pilot.build_execution_plan()[0])
    assert result["status"] == "interrupted"
    assert result["timing"]["wall_clock_elapsed_seconds"] >= 0
    assert result["memory"]["rss_peak_bytes"] >= result["memory"]["rss_start_bytes"]


def test_dry_run_prints_plan_without_creating_output_or_fitting(
    tmp_path, monkeypatch, capsys
):
    protocol_path, dataset_manifest_path, _ = _install_fake_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        pilot,
        "_fit_one",
        lambda *args: pytest.fail("dry-run must not train"),
    )
    output_dir = tmp_path / "not-created"
    assert pilot.run_pilot(
        dry_run=True,
        max_runs=2,
        protocol_path=protocol_path,
        dataset_manifest_path=dataset_manifest_path,
        output_dir=output_dir,
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["expected_fits"]["selected"] == {
        "qnim": 15, "pca_svm_rbf": 9, "total": 24
    }
    assert payload["resource_configuration"]["batch_size"] == 32
    assert payload["training_started"] is False
    assert not output_dir.exists()


def test_cli_exposes_resume_and_max_runs_without_starting_training(monkeypatch):
    received = {}

    def fake_run_pilot(**kwargs):
        received.update(kwargs)
        return 0

    monkeypatch.setattr(pilot, "run_pilot", fake_run_pilot)
    assert pilot.main(["--resume", "--max-runs", "2"]) == 0
    assert received == {
        "dry_run": False,
        "max_runs": 2,
        "resume": True,
        "arm": None,
        "n_qubits": None,
        "seed": None,
        "output_dir": None,
    }
    assert pilot.main([
        "--dry-run", "--arm", "full_system", "--qubits", "4", "--seed", "42",
    ]) == 0
    assert received["dry_run"] is True
    assert received["arm"] == "full_system"
    assert received["n_qubits"] == 4
    assert received["seed"] == 42


def test_dry_run_does_not_modify_approved_protocol_or_historical_q1_6_files(tmp_path):
    protected = (
        pilot.PROTOCOL_PATH,
        pilot.DATASET_MANIFEST_PATH,
        pilot.ROOT / "scripts" / "run_qnim_q1_6_scaling_pilot.py",
        pilot.ROOT / "results" / "scaling_study" / "qnim_q1_6_scaling_pilot_v1.json",
        pilot.ROOT / "results" / "scaling_study" / "qnim_q1_6_scaling_pilot_v1_partial_20261120.json",
        pilot.ROOT / "results" / "scaling_study" / "qnim_q1_6_scaling_pilot_v1_partial_analysis.json",
        pilot.DATASET_DIR / "X_train.npy",
        pilot.DATASET_DIR / "y_train.npy",
        pilot.DATASET_DIR / "X_val.npy",
        pilot.DATASET_DIR / "y_val.npy",
        pilot.DATASET_DIR / "dataset_metadata.json",
    )
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    output_dir = tmp_path / "dry-run-output"
    assert pilot.run_pilot(dry_run=True, output_dir=output_dir) == 0
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    assert after == before
    assert not output_dir.exists()
