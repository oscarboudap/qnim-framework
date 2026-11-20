import numpy as np
import pytest

from qiskit import QuantumCircuit
from src.infrastructure.qiskit_vqc_trainer import _measure_qubit_marginals_exact
from src.infrastructure.qnspsa_eml_feynman import (
    QNSPSAConfig,
    QNSPSAEMLFeynman,
)


def test_optimizer_active_mask_freezes_parameters_and_records_updates():
    initial = np.array([0.75, -0.5, 0.8])
    target = np.array([0.0, 0.2, -0.1])
    mask = np.array([False, True, True])
    optimizer = QNSPSAEMLFeynman(QNSPSAConfig(
        maxiter=4,
        lr=0.01,
        perturbation=0.05,
        lambda_eml=0.0,
        n_feynman_params=0,
        patience=10,
        seed=42,
    ))

    result = optimizer.minimize(
        lambda params: float(np.sum((params - target) ** 2)),
        initial,
        active_mask=mask,
    )

    assert result.n_iter == 4
    assert len(result.optimizer_step_history) == 4
    assert all(
        not np.isclose(step["step_norm"], 0.0)
        for step in result.optimizer_step_history
    )
    assert all(
        step["proposed_step_norm"] >= step["step_norm"]
        for step in result.optimizer_step_history
    )
    assert all("blocking_loss_delta" in step for step in result.optimizer_step_history)
    assert result.optimal_params[0] == initial[0]
    assert not np.array_equal(result.optimal_params[1:], initial[1:])


def test_optimizer_active_mask_validates_shape_and_nonempty_selection():
    optimizer = QNSPSAEMLFeynman(QNSPSAConfig(maxiter=1, n_feynman_params=0))
    loss = lambda params: float(np.sum(params**2))

    with pytest.raises(ValueError, match="misma forma"):
        optimizer.minimize(loss, np.ones(2), active_mask=np.ones(1, dtype=bool))

    with pytest.raises(ValueError, match="al menos un parámetro"):
        optimizer.minimize(loss, np.ones(2), active_mask=np.zeros(2, dtype=bool))


def test_exact_marginals_match_a_hadamard_state():
    circuit = QuantumCircuit(1)
    circuit.h(0)

    marginals = _measure_qubit_marginals_exact(
        circuit,
        x_params=[],
        ansatz_params=[],
        theta=np.array([]),
        X_cheb=np.empty((1, 0)),
        n_qubits=1,
    )

    np.testing.assert_allclose(marginals, [[0.5]], atol=1e-12)
