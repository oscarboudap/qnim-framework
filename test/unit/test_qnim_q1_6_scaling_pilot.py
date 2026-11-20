import json
from pathlib import Path

import numpy as np

from scripts import run_qnim_q1_6_scaling_pilot as pilot
from src.infrastructure.qiskit_vqc_trainer import (
    _n_readout_params,
    _prepare_feature_matrix,
    _readout_feature_count,
)


class FakeAdapter:
    strict_physics = True
    n_classes = 3
    theory_names = ["A", "B", "C"]

    def _generate_one(self, class_index, rng):
        return rng.normal(size=16) + class_index * np.linspace(0.0, 1.0, 16)


def test_train_validation_generation_is_reproducible_and_discards_tail():
    kwargs = {
        "adapter": FakeAdapter(),
        "seed": 42,
        "train_per_class": 5,
        "validation_per_class": 2,
        "discarded_draws_per_class": 1,
        "n_components": 4,
        "n_pca_fit": 6,
    }

    first = pilot._generate_train_validation(**kwargs)
    second = pilot._generate_train_validation(**kwargs)

    assert first["X_train"].shape == (15, 4)
    assert first["X_val"].shape == (6, 4)
    assert not any(name.startswith("X_test") for name in first)
    np.testing.assert_array_equal(first["X_train"], second["X_train"])
    np.testing.assert_array_equal(first["X_val"], second["X_val"])
    np.testing.assert_array_equal(first["y_train"], second["y_train"])
    np.testing.assert_array_equal(first["y_val"], second["y_val"])


def test_fixed_input_architecture_encodes_four_inputs_and_uses_extra_qubits():
    from src.infrastructure.qiskit_vqc_trainer import (
        _build_feature_map_and_ansatz,
    )

    for n_qubits in pilot.QUIBIT_COUNTS:
        circuit, feature_parameters, ansatz_parameters = (
            _build_feature_map_and_ansatz(
                n_qubits,
                n_features=pilot.FIXED_INPUT_DIMENSION,
                feature_map_reps=pilot.FEATURE_MAP_REPS,
                ansatz_reps=pilot.ANSATZ_REPS,
                entanglement=pilot.ENTANGLEMENT,
            )
        )
        assert circuit.num_qubits == n_qubits
        assert len(feature_parameters) == pilot.FIXED_INPUT_DIMENSION
        assert len(ansatz_parameters) > 0
        if n_qubits > pilot.FIXED_INPUT_DIMENSION:
            decomposed = circuit.decompose(reps=10)
            cross_boundary_cx = [
                instruction
                for instruction in decomposed.data
                if instruction.operation.name == "cx"
                and any(
                    decomposed.find_bit(qubit).index >= pilot.FIXED_INPUT_DIMENSION
                    for qubit in instruction.qubits
                )
            ]
            assert cross_boundary_cx


def test_scaling_arms_match_input_circuit_and_readout_dimensions():
    from src.infrastructure.qiskit_vqc_trainer import (
        _build_feature_map_and_ansatz,
    )

    for n_qubits in pilot.QUIBIT_COUNTS:
        feature_counts = (
            (n_qubits,)
            if n_qubits == pilot.FIXED_INPUT_DIMENSION
            else (n_qubits, pilot.FIXED_INPUT_DIMENSION)
        )
        for n_features in feature_counts:
            circuit, feature_parameters, ansatz_parameters = (
                _build_feature_map_and_ansatz(
                    n_qubits,
                    n_features=n_features,
                    feature_map_reps=pilot.FEATURE_MAP_REPS,
                    ansatz_reps=pilot.ANSATZ_REPS,
                    entanglement=pilot.ENTANGLEMENT,
                )
            )
            n_observables = n_qubits * (n_qubits + 1) // 2
            assert _prepare_feature_matrix(
                np.zeros((2, n_features)), n_features
            ).shape == (2, n_features)
            assert circuit.num_qubits == n_qubits
            assert len(feature_parameters) == n_features
            assert len(ansatz_parameters) == 6 * n_qubits
            assert _readout_feature_count(
                n_qubits, pilot.READOUT_MODE
            ) == n_observables
            assert _readout_feature_count(
                n_qubits, "marginals_zz_neighbors"
            ) == 2 * n_qubits - 1
            assert _n_readout_params(
                n_observables, pilot.N_CLASSES, pilot.READOUT_HIDDEN_SIZE
            ) == (
                n_observables * pilot.READOUT_HIDDEN_SIZE
                + pilot.READOUT_HIDDEN_SIZE
                + pilot.READOUT_HIDDEN_SIZE * pilot.N_CLASSES
                + pilot.N_CLASSES
            )


def test_q1_6_dataset_loader_never_opens_test_files(tmp_path, monkeypatch):
    class_names = [f"class-{index}" for index in range(pilot.N_CLASSES)]
    X_train = np.zeros((2600, 16), dtype=float)
    y_train = np.repeat(np.arange(pilot.N_CLASSES), 200)
    X_val = np.zeros((650, 16), dtype=float)
    y_val = np.repeat(np.arange(pilot.N_CLASSES), 50)
    arrays = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }
    for name, array in arrays.items():
        np.save(tmp_path / f"{name}.npy", array)
    metadata = {
        "class_names": class_names,
        "dataset_hashes": {
            "train_sha256": pilot._split_sha256(X_train, y_train),
            "validation_sha256": pilot._split_sha256(X_val, y_val),
        },
        "final_test": {"evaluated": False},
    }
    (tmp_path / "dataset_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    loaded = []
    original_load = np.load

    def recording_load(path, *args, **kwargs):
        loaded.append(Path(path).name)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(pilot.np, "load", recording_load)

    result = pilot._load_dataset(tmp_path, class_names)

    assert set(loaded) == {
        "X_train.npy",
        "y_train.npy",
        "X_val.npy",
        "y_val.npy",
    }
    assert "X_test" not in result and "y_test" not in result
