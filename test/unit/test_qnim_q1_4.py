import numpy as np
from qiskit import QuantumCircuit

from scripts.run_qnim_q1_4 import _build_representations
from src.infrastructure.qiskit_vqc_trainer import (
    _flatten_readout_mlp,
    _fit_readout_mlp_classically,
    _make_vqc_loss_fn,
    _measure_qubit_observables_exact,
    _measure_qubit_marginals_exact,
    _qubit_readout_features,
    _readout_feature_count,
    QiskitVQCTrainer,
)


def _measure(circuit):
    return _measure_qubit_observables_exact(
        circuit,
        x_params=[],
        ansatz_params=[],
        theta=np.array([]),
        X_cheb=np.empty((1, 0)),
        n_qubits=circuit.num_qubits,
    )


def test_exact_observables_preserve_bell_state_correlation():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    observables = _measure(circuit)

    np.testing.assert_allclose(observables["p0"], [[0.5, 0.5]], atol=1e-12)
    np.testing.assert_allclose(observables["zz"], [[1.0]], atol=1e-12)
    np.testing.assert_allclose(observables["xyz"], [[0.0] * 6], atol=1e-12)


def test_exact_local_y_expectation_uses_qiskit_qubit_order_and_sign():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.s(0)

    observables = _measure(circuit)

    np.testing.assert_allclose(observables["xyz"], [[0.0, 1.0, 0.0]], atol=1e-12)


def test_q1_4_variants_have_registered_dimensions_and_pair_order():
    observables = {
        "p0": np.zeros((2, 12)),
        "zz": np.tile(np.arange(66, dtype=float), (2, 1)),
        "xyz": np.zeros((2, 36)),
    }

    features = _build_representations(observables)

    assert {name: values.shape[1] for name, values in features.items()} == {
        "A_marginals": 12,
        "B_adjacent_zz": 23,
        "C_all_zz": 78,
        "D_local_xyz": 36,
    }
    np.testing.assert_array_equal(
        features["B_adjacent_zz"][:, 12:],
        observables["zz"][:, [0, 11, 21, 30, 38, 45, 51, 56, 60, 63, 65]],
    )
    np.testing.assert_array_equal(features["C_all_zz"][:, 12:], observables["zz"])


def test_scaling_readout_dimensions_follow_qubit_count():
    assert [
        _readout_feature_count(qubits, "marginals_zz")
        for qubits in (4, 8, 12, 16, 20)
    ] == [10, 36, 78, 136, 210]


def test_neighbor_zz_readout_has_linear_dimensions():
    assert [
        _readout_feature_count(qubits, "marginals_zz_neighbors")
        for qubits in (1, 4, 8, 12, 16)
    ] == [1, 7, 15, 23, 31]


def test_marginal_zz_readout_preserves_pair_correlations_in_counts_and_statevector():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    measured = _qubit_readout_features(
        {"00": 32, "11": 32}, 2, readout_mode="marginals_zz"
    )
    exact = _measure_qubit_marginals_exact(
        circuit,
        x_params=[],
        ansatz_params=[],
        theta=np.array([]),
        X_cheb=np.empty((1, 0)),
        n_qubits=2,
        readout_mode="marginals_zz",
    )

    np.testing.assert_allclose(measured, [0.5, 0.5, 1.0], atol=1e-12)
    np.testing.assert_allclose(exact, [[0.5, 0.5, 1.0]], atol=1e-12)


def test_neighbor_zz_readout_selects_only_adjacent_pairs_in_counts_and_statevector():
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 2)

    measured = _qubit_readout_features(
        {"000": 32, "101": 32},
        3,
        readout_mode="marginals_zz_neighbors",
    )
    exact = _measure_qubit_marginals_exact(
        circuit,
        x_params=[],
        ansatz_params=[],
        theta=np.array([]),
        X_cheb=np.empty((1, 0)),
        n_qubits=3,
        readout_mode="marginals_zz_neighbors",
    )

    np.testing.assert_allclose(measured, [0.5, 1.0, 0.5, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(exact, [measured], atol=1e-12)


def test_readout_mlp_accepts_marginal_and_zz_features():
    features = np.array([
        [0.5, 0.5, 1.0],
        [0.5, 0.5, -1.0],
    ])
    weights = _fit_readout_mlp_classically(
        features,
        np.array([0, 1]),
        n_classes=2,
        hidden_size=4,
        n_iters=2,
    )

    assert weights[0].shape == (4, 3)


def test_qnim_loss_trains_and_scores_the_marginal_zz_readout():
    X_train = np.array([
        [0.1, 0.2],
        [-0.1, -0.2],
        [0.3, 0.4],
        [-0.3, -0.4],
    ])
    y_train = np.array([0, 1, 0, 1])
    loss_fn = _make_vqc_loss_fn(
        X_train=X_train,
        y_train=y_train,
        n_qubits=2,
        n_features=2,
        shots=16,
        batch_size_local=4,
        feature_map_reps=1,
        ansatz_reps=1,
        readout_hidden_size=4,
        readout_mode="marginals_zz",
        random_seed=7,
    )
    ansatz = np.zeros(loss_fn.n_ansatz_params)
    W1, b1, W2, b2 = loss_fn.warm_start_readout(ansatz, max_samples=4)
    theta = np.concatenate((ansatz, _flatten_readout_mlp(W1, b1, W2, b2)))

    measured = loss_fn.measure_quantum_features_exact(X_train[:2], ansatz)
    loss = loss_fn.evaluate_batch(theta, np.arange(4), shots_override=16)
    trainer = QiskitVQCTrainer(
        mode="sim",
        readout_hidden_size=4,
        readout_mode="marginals_zz",
        random_seed=7,
    )
    _, confusion, predictions = trainer._evaluate_real_predictions(
        weights=theta,
        X_train=X_train,
        X_val=X_train,
        y_val=y_train,
        n_qubits=2,
        n_features=2,
        n_classes=2,
        shots=16,
    )

    assert measured.shape == (2, 3)
    assert W1.shape == (4, 3)
    assert np.isfinite(loss)
    assert len(predictions) == 4
    assert len(confusion) == 2
