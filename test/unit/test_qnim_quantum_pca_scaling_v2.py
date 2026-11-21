import numpy as np
import pytest
from types import SimpleNamespace

from src.infrastructure.qnim_v2_support import (
    atomic_save_npz,
    atomic_write_json,
    bitstring_features,
    candidate_statistics,
    compare_candidates,
    indices_sha256,
    mlp_loss_and_feature_gradient,
    measurement_artifact_arrays,
    measurement_moments,
    parameters_sha256,
    probability_artifact_arrays,
    probability_diagnostics,
    stratified_disjoint_roles,
    validate_disjoint_roles,
)
from src.infrastructure.qnspsa_eml_feynman import (
    QNSPSAConfig,
    QNSPSAEMLFeynman,
)
from src.infrastructure.qiskit_vqc_trainer import _make_vqc_loss_fn
from scripts.run_qnim_quantum_pca_scaling_v2 import _parameter_hashes


def _labels():
    return np.repeat(np.arange(13, dtype=np.int64), 200)


def _small_head(class_zero_bias):
    parameters = np.zeros(15, dtype=np.float64)
    parameters[13] = class_zero_bias
    return parameters


def test_roles_are_stratified_disjoint_complete_and_deterministic():
    labels = _labels()
    roles = stratified_disjoint_roles(labels, seed=42)
    repeated = stratified_disjoint_roles(labels, seed=42)

    assert {name: len(indices) for name, indices in roles.items()} == {
        "gradient": 2275,
        "readout": 195,
        "reference": 130,
    }
    for name in roles:
        np.testing.assert_array_equal(roles[name], repeated[name])
    validate_disjoint_roles(labels, roles)


def test_roles_reject_overlap_and_non_stratified_sizes():
    labels = _labels()
    roles = stratified_disjoint_roles(labels)
    roles["readout"] = np.concatenate((roles["readout"][:-1], roles["gradient"][:1]))

    with pytest.raises(ValueError, match="pairwise disjoint"):
        validate_disjoint_roles(labels, roles)
    with pytest.raises(ValueError, match="each class"):
        stratified_disjoint_roles(labels, per_class=(174, 15, 10))


def test_measurement_moments_preserve_joint_bitstring_covariance():
    mean, standard_errors, covariance, shots = measurement_moments(
        {"00": 3, "11": 1},
        n_qubits=2,
    )

    assert shots == 4
    np.testing.assert_allclose(mean, [0.75, 0.75, 1.0])
    assert standard_errors[0] == pytest.approx(0.25)
    assert standard_errors[1] == pytest.approx(0.25)
    assert covariance[0, 1] > 0.0
    assert standard_errors[2] == pytest.approx(0.0)


def test_bitstring_feature_order_matches_little_endian_neighbor_zz_readout():
    np.testing.assert_array_equal(
        bitstring_features("01", n_qubits=2),
        np.array([0.0, 1.0, -1.0]),
    )


def test_candidate_uncertainty_and_paired_tie_rule():
    labels = np.zeros(10, dtype=np.int64)
    indices = np.array([2, 7], dtype=np.int64)
    counts = [{"00": 512}, {"00": 512}]
    better = candidate_statistics(
        counts,
        labels,
        indices,
        _small_head(2.0),
        n_qubits=2,
        n_ansatz_parameters=1,
        n_classes=2,
        hidden_size=2,
    )
    worse = candidate_statistics(
        counts,
        labels,
        indices,
        _small_head(-2.0),
        n_qubits=2,
        n_ansatz_parameters=1,
        n_classes=2,
        hidden_size=2,
    )
    assert better["objective"] < worse["objective"]
    assert np.all(better["per_example_loss_variances"] == 0.0)

    decision = compare_candidates(
        better,
        worse,
        comparison_number=1,
    )
    assert decision["replaces_incumbent"] is True
    assert decision["one_sided_upper_bound"] < -0.0005

    tied = compare_candidates(
        worse,
        worse,
        comparison_number=2,
    )
    assert tied["paired_objective_delta"] == pytest.approx(0.0)
    assert tied["replaces_incumbent"] is False


def test_candidate_comparison_rejects_mismatched_reference_indices():
    labels = np.zeros(10, dtype=np.int64)
    counts = [{"00": 512}, {"00": 512}]
    candidate = candidate_statistics(
        counts,
        labels,
        np.array([2, 7]),
        _small_head(1.0),
        n_qubits=2,
        n_ansatz_parameters=1,
        n_classes=2,
        hidden_size=2,
    )
    incumbent = candidate_statistics(
        counts,
        labels,
        np.array([2, 8]),
        _small_head(1.0),
        n_qubits=2,
        n_ansatz_parameters=1,
        n_classes=2,
        hidden_size=2,
    )

    with pytest.raises(ValueError, match="identical reference"):
        compare_candidates(candidate, incumbent, comparison_number=1)


def test_delta_method_tracks_monte_carlo_for_a_smooth_finite_shot_case():
    rng = np.random.default_rng(2026)
    outcome_strings = np.array(["00", "01", "10", "11"])
    outcome_probabilities = np.array([0.3, 0.2, 0.2, 0.3])
    outcome_features = np.stack(
        [bitstring_features(value, n_qubits=2) for value in outcome_strings]
    )
    expected_features = outcome_probabilities @ outcome_features
    parameters = np.concatenate((
        np.array([0.0]),
        np.array([[0.2, 0.2, 0.1], [-0.1, 0.2, -0.2]]).ravel(),
        np.array([0.7, 0.5]),
        np.array([[0.4, -0.3], [-0.2, 0.3]]).ravel(),
        np.array([0.1, 0.0]),
    ))
    _, true_loss, _ = mlp_loss_and_feature_gradient(
        expected_features,
        parameters,
        0,
        n_ansatz_parameters=1,
        n_features=3,
        n_classes=2,
        hidden_size=2,
    )

    losses = []
    estimated_variances = []
    covered = []
    labels = np.array([0])
    indices = np.array([0])
    for _ in range(600):
        counts_by_outcome = np.bincount(
            rng.choice(4, size=512, p=outcome_probabilities), minlength=4
        )
        counts = {
            outcome: int(count)
            for outcome, count in zip(outcome_strings, counts_by_outcome)
            if count
        }
        stats = candidate_statistics(
            [counts],
            labels,
            indices,
            parameters,
            n_qubits=2,
            n_ansatz_parameters=1,
            n_classes=2,
            hidden_size=2,
        )
        loss = stats["objective"]
        variance = stats["per_example_loss_variances"][0]
        losses.append(loss)
        estimated_variances.append(variance)
        covered.append(abs(loss - true_loss) <= 1.96 * np.sqrt(variance))

    variance_ratio = np.mean(estimated_variances) / np.var(losses, ddof=1)
    coverage = np.mean(covered)
    assert 0.8 < variance_ratio < 1.25
    assert 0.90 < coverage < 0.98


def test_delta_method_low_probability_and_uniform_entropy_cases_are_finite():
    low_probability_parameters = _small_head(-1000.0)
    probabilities, loss, feature_gradient = mlp_loss_and_feature_gradient(
        np.zeros(3),
        low_probability_parameters,
        0,
        n_ansatz_parameters=1,
        n_features=3,
        n_classes=2,
        hidden_size=2,
    )
    assert np.isfinite(probabilities).all()
    assert loss == pytest.approx(-np.log(1e-10))
    np.testing.assert_array_equal(feature_gradient, np.zeros(3))

    uniform_parameters = np.zeros(48)
    probabilities, loss, feature_gradient = mlp_loss_and_feature_gradient(
        np.zeros(3),
        uniform_parameters,
        0,
        n_ansatz_parameters=1,
        n_features=3,
        n_classes=13,
        hidden_size=2,
    )
    np.testing.assert_allclose(probabilities, np.full(13, 1 / 13))
    assert loss == pytest.approx(np.log(13))
    assert np.isfinite(feature_gradient).all()


def test_nearly_tied_candidates_do_not_replace_on_uncertain_point_estimate():
    indices = np.arange(130)
    base = {
        "reference_indices": indices,
        "reference_indices_sha256": indices_sha256(indices),
        "per_example_loss_variances": np.full(130, 0.01),
    }
    incumbent = {**base, "per_example_losses": np.zeros(130)}
    candidate = {
        **base,
        "per_example_losses": np.full(130, -0.0001),
    }

    decision = compare_candidates(candidate, incumbent, comparison_number=1)

    assert decision["paired_objective_delta"] == pytest.approx(-0.0001)
    assert decision["replaces_incumbent"] is False


def test_repeated_null_comparisons_have_low_familywise_replacement_rate():
    rng = np.random.default_rng(42)
    indices = np.arange(130)
    index_hash = indices_sha256(indices)
    per_example_variance = 0.01
    false_replacement_families = 0
    family_count = 1000

    for _ in range(family_count):
        fixed_example_losses = rng.normal(size=len(indices))

        def noisy_candidate():
            return {
                "reference_indices": indices,
                "reference_indices_sha256": index_hash,
                "per_example_losses": fixed_example_losses
                + rng.normal(0.0, np.sqrt(per_example_variance), len(indices)),
                "per_example_loss_variances": np.full(
                    len(indices), per_example_variance
                ),
            }

        incumbent = noisy_candidate()
        replaced = False
        for comparison_number in range(1, 23):
            candidate = noisy_candidate()
            decision = compare_candidates(
                candidate,
                incumbent,
                comparison_number=comparison_number,
            )
            if decision["replaces_incumbent"]:
                incumbent = candidate
                replaced = True
        false_replacement_families += int(replaced)

    assert false_replacement_families / family_count < 0.08


def test_parameter_and_index_hashes_are_stable():
    values = np.array([0.25, -0.5], dtype=np.float32)
    assert parameters_sha256(values) == parameters_sha256(values.astype(np.float64))
    assert indices_sha256(np.array([1, 3])) == indices_sha256(
        np.array([1, 3], dtype=np.int32)
    )
    assert parameters_sha256(values) != parameters_sha256(values[::-1])


def test_probability_diagnostics_report_per_example_outputs_and_calibration():
    probabilities = np.array([[0.8, 0.2], [0.4, 0.6]], dtype=np.float64)
    result = probability_diagnostics(
        probabilities,
        np.array([0, 0]),
        n_classes=2,
        calibration_bins=2,
    )

    np.testing.assert_array_equal(result["predicted_labels"], [0, 1])
    assert len(result["negative_log_likelihood"]) == 2
    assert len(result["top_two_margin"]) == 2
    assert len(result["predictive_entropy"]) == 2
    assert result["multiclass_brier_score"] == pytest.approx(0.4)
    assert len(result["calibration_bins"]) == 2
    clipped = probability_diagnostics(
        np.array([[1.0, 0.0]]),
        np.array([1]),
        n_classes=2,
    )
    assert clipped["negative_log_likelihood"][0] == pytest.approx(-np.log(1e-10))


def test_measurement_and_probability_artifacts_round_trip_atomically(tmp_path):
    features, errors, _, shots = measurement_moments(
        {"0" * 16: 256, "1" * 16: 256}, n_qubits=16
    )
    measurement_rows = [{
        "example_id": 7,
        "split": "train",
        "class_id": 2,
        "iteration": 0,
        "phase": "initial",
        "quantum_parameter_sha256": "a" * 64,
        "classical_parameter_sha256": "b" * 64,
        "complete_parameter_sha256": "c" * 64,
        "actual_shots": shots,
        "features": features,
        "standard_errors": errors,
        "counts": {"0" * 16: 256, "1" * 16: 256},
    }]
    measurement_arrays = measurement_artifact_arrays(measurement_rows)

    measurement_path = tmp_path / "measurements.npz"
    atomic_save_npz(measurement_path, measurement_arrays)
    with np.load(measurement_path) as saved:
        np.testing.assert_array_equal(saved["features"][0], features)
        assert saved["marginals"].shape == (1, 16)
        assert saved["adjacent_zz_correlations"].shape == (1, 15)
        np.testing.assert_array_equal(saved["outcome_counts"], [256, 256])
        assert saved["actual_shots"][0] == 512

    probability_rows = [{
        "example_id": 7,
        "split": "validation",
        "true_label": 2,
        "predicted_label": 2,
        "iteration": 0,
        "phase": "selected",
        "quantum_parameter_sha256": "a" * 64,
        "classical_parameter_sha256": "b" * 64,
        "complete_parameter_sha256": "c" * 64,
        "probabilities": np.full(13, 1 / 13),
        "negative_log_likelihood": np.log(13),
        "top_two_margin": 0.0,
        "predictive_entropy": np.log(13),
    }]
    probability_arrays = probability_artifact_arrays(probability_rows)
    probability_path = tmp_path / "probabilities.npz"
    atomic_save_npz(probability_path, probability_arrays)
    with np.load(probability_path) as saved:
        np.testing.assert_allclose(saved["probabilities"].sum(axis=1), [1.0])

    metadata_path = tmp_path / "metadata.json"
    atomic_write_json(metadata_path, {"backend": "synthetic", "shots": 512})
    assert '"backend": "synthetic"' in metadata_path.read_text(encoding="utf-8")


def test_optimizer_scores_initial_pre_refit_and_post_refit_complete_vectors():
    scored = []
    accepted_vectors = []
    incumbent = {"objective": float("inf")}

    def candidate_callback(iteration, phase, parameters, context):
        objective = float(np.sum(parameters**2))
        if phase == "initial" or objective < incumbent["objective"]:
            incumbent["objective"] = objective
            replaced = True
        else:
            replaced = False
        scored.append((iteration, phase, parameters.copy()))
        if replaced:
            accepted_vectors.append(parameters.copy())
        return {
            "objective": objective,
            "replaces_incumbent": replaced,
            "objective_function_calls": 1,
        }

    def refit_callback(iteration, parameters, loss):
        parameters[1] = 0.0
        return True

    optimizer = QNSPSAEMLFeynman(
        QNSPSAConfig(
            maxiter=1,
            n_feynman_params=1,
            patience=5,
            seed=42,
        )
    )
    result = optimizer.minimize(
        lambda values: float(np.sum(values**2)),
        np.array([0.5, 1.0]),
        callback=refit_callback,
        checkpoint_candidate_callback=candidate_callback,
        parameter_block_sizes=(1, 1),
    )

    assert [(iteration, phase) for iteration, phase, _ in scored] == [
        (0, "initial"),
        (1, "pre_refit"),
        (1, "post_refit"),
    ]
    assert not np.array_equal(scored[1][2], scored[2][2])
    np.testing.assert_array_equal(result.optimal_params, accepted_vectors[-1])
    assert result.n_evals >= 3
    assert "block_telemetry" in result.optimizer_step_history[0]
    assert "quantum" in result.optimizer_step_history[0]["block_telemetry"][
        "spsa_gradient_norm"
    ]


def test_atomic_checkpoint_survives_interrupted_later_attempt(tmp_path):
    checkpoint_path = tmp_path / "selected_checkpoint.npz"
    first = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    latest = np.array([0.4, 0.5, 0.6], dtype=np.float64)
    atomic_save_npz(checkpoint_path, {"parameters": first})

    with pytest.raises(RuntimeError, match="simulated interruption"):
        atomic_save_npz(checkpoint_path, {"parameters": latest})
        raise RuntimeError("simulated interruption after checkpoint replacement")

    with np.load(checkpoint_path) as saved:
        np.testing.assert_array_equal(saved["parameters"], latest)
        assert parameters_sha256(latest) == parameters_sha256(saved["parameters"])


class _FakeBitArray:
    def __init__(self, counts, *, num_bits=16, num_shots=512):
        self._counts = counts
        self.num_bits = num_bits
        self.num_shots = num_shots

    def get_counts(self):
        return self._counts


class _FakeJob:
    def __init__(self, results):
        self._results = results

    def result(self):
        return self._results


class _FakeSampler:
    def __init__(self):
        self.calls = []
        self.num_bits = 16
        self.num_shots = None
        self.invalid_counts = False
        self.short_bitstrings = False

    def run(self, pubs, *, shots):
        patterns = ["0" * 16, "0" * 15 + "1", "1" + "0" * 15, "1" * 16]
        self.calls.append({"pubs": pubs, "shots": shots})
        results = []
        for index, pub in enumerate(pubs):
            circuit = pub[0] if isinstance(pub, tuple) else pub
            assert circuit.num_qubits == 16
            assert circuit.num_clbits == 16
            assert isinstance(pub, tuple) and len(pub) == 2
            assert pub[1] == []
            bitstring = patterns[index % len(patterns)]
            if self.short_bitstrings:
                bitstring = bitstring[1:]
            counts = {"invalid": shots} if self.invalid_counts else {bitstring: shots}
            meas = _FakeBitArray(
                counts,
                num_bits=self.num_bits,
                num_shots=shots if self.num_shots is None else self.num_shots,
            )
            results.append(SimpleNamespace(data=SimpleNamespace(meas=meas)))
        return _FakeJob(results)


def _fake_q16_loss(sampler):
    labels = np.arange(13, dtype=np.int64)
    X_train = np.linspace(0.0, 1.0, 13 * 16).reshape(13, 16)
    roles = {
        "gradient": np.arange(7, dtype=np.int64),
        "readout": np.arange(7, 10, dtype=np.int64),
        "reference": np.arange(10, 13, dtype=np.int64),
    }
    loss_fn = _make_vqc_loss_fn(
        X_train=X_train,
        y_train=labels,
        n_qubits=16,
        n_features=16,
        shots=512,
        backend_sampler=sampler,
        batch_size_local=4,
        feature_map_reps=2,
        ansatz_reps=2,
        entanglement="linear",
        readout_hidden_size=16,
        random_seed=42,
        readout_mode="marginals_zz_neighbors",
        input_preprocessing="identity",
        gradient_indices=roles["gradient"],
        readout_indices=roles["readout"],
        checkpoint_reference_indices=roles["reference"],
        readout_fit_seed=42,
    )
    return loss_fn, X_train, labels, roles


def test_fake_sampler_contract_roles_features_and_pub_association():
    sampler = _FakeSampler()
    loss_fn, X_train, labels, roles = _fake_q16_loss(sampler)
    assert loss_fn.n_ansatz_params == 96
    assert loss_fn.readout_feature_count == 31

    theta = np.zeros(829, dtype=np.float64)
    state_hashes = _parameter_hashes(theta, loss_fn.n_ansatz_params)
    assert state_hashes["quantum_parameter_sha256"] == parameters_sha256(theta[:96])
    assert state_hashes["classical_parameter_sha256"] == parameters_sha256(theta[96:])
    assert state_hashes["complete_parameter_sha256"] == parameters_sha256(theta)
    loss_fn.refresh_batch()
    assert set(loss_fn.current_batch_indices()).issubset(set(roles["gradient"]))
    assert np.isfinite(loss_fn(theta))
    assert sampler.calls[-1]["shots"] == 512
    assert len(sampler.calls[-1]["pubs"]) == 4

    loss_fn.warm_start_readout(theta[: loss_fn.n_ansatz_params])
    np.testing.assert_array_equal(loss_fn.last_readout_indices, roles["readout"])
    assert len(sampler.calls[-1]["pubs"]) == len(roles["readout"])
    assert sampler.calls[-1]["shots"] == 512

    count_rows = loss_fn.measure_counts_by_indices(theta, roles["reference"], 512)
    assert len(count_rows) == len(roles["reference"])
    np.testing.assert_array_equal(
        bitstring_features(next(iter(count_rows[0])), 16),
        np.ones(31),
    )
    np.testing.assert_array_equal(
        bitstring_features(next(iter(count_rows[1])), 16)[:16],
        np.array([0.0] + [1.0] * 15),
    )
    np.testing.assert_array_equal(
        bitstring_features(next(iter(count_rows[2])), 16)[:16],
        np.array([1.0] * 15 + [0.0]),
    )

    stats = candidate_statistics(
        count_rows,
        labels,
        roles["reference"],
        theta,
        n_qubits=16,
        n_ansatz_parameters=96,
        n_classes=13,
        hidden_size=16,
    )
    assert stats["features"].shape == (3, 31)
    assert stats["probabilities"].shape == (3, 13)
    assert loss_fn.readout_feature_count == 16 + 15

    with pytest.raises(ValueError, match="fixed reference indices"):
        loss_fn.measure_counts_by_indices(theta, roles["readout"], 512)

    validation = loss_fn.measure_counts_external(theta, X_train[:2], 512)
    assert len(validation) == 2
    assert loss_fn.execution_telemetry["n_validation_examples_evaluated"] == 2
    assert loss_fn.execution_telemetry["n_sampler_run_calls"] == len(sampler.calls)
    assert loss_fn.execution_telemetry["n_sampler_jobs_submitted"] == len(sampler.calls)
    assert loss_fn.execution_telemetry["n_sampler_jobs_completed"] == len(sampler.calls)


@pytest.mark.parametrize(
    ("num_bits", "num_shots", "invalid_counts", "message"),
    [
        (15, 512, False, "has 15 bits"),
        (16, 511, False, "has 511 shots"),
        (16, 512, True, "invalid counts"),
        (None, 512, False, "has a 15-bit count"),
    ],
)
def test_fake_sampler_rejects_malformed_measurement_results(
    num_bits, num_shots, invalid_counts, message
):
    sampler = _FakeSampler()
    sampler.num_bits = num_bits
    sampler.num_shots = num_shots
    sampler.invalid_counts = invalid_counts
    sampler.short_bitstrings = message == "has a 15-bit count"
    loss_fn, X_train, _, _ = _fake_q16_loss(sampler)

    with pytest.raises(RuntimeError, match=message):
        loss_fn.measure_counts_external(np.zeros(829), X_train[:1], 512)
