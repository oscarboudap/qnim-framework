from src.application.use_cases.generate_experiment_results_use_case import ExperimentConfig


def test_experiment_config_supports_27q_hardware_profile():
    cfg = ExperimentConfig(
        n_qubits=27,
        feature_map_reps=1,
        ansatz_reps=2,
        entanglement="linear",
    )

    assert cfg.n_qubits == 27
    assert cfg.feature_map_reps == 1
    assert cfg.ansatz_reps == 2
    assert cfg.entanglement == "linear"
