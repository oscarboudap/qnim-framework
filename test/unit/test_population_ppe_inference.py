import numpy as np
import pytest

from src.application.population_ppe_inference import (
    infer_shared_beta,
    phase_residual_log_likelihoods,
)


def test_shared_beta_evidence_accumulates_and_posterior_narrows():
    beta_grid = np.linspace(-0.02, 0.02, 161)
    beta_true = 0.008
    sigma = 0.006
    event_log_likelihood = -0.5 * ((beta_grid - beta_true) / sigma) ** 2

    one_event = infer_shared_beta(beta_grid, event_log_likelihood[None, :])
    population = infer_shared_beta(
        beta_grid, np.tile(event_log_likelihood, (16, 1))
    )

    assert population.n_events == 16
    assert np.sum(population.posterior_mass) == pytest.approx(1.0)
    assert population.credible_interval[1] - population.credible_interval[0] < (
        one_event.credible_interval[1] - one_event.credible_interval[0]
    )
    assert population.log_bayes_factor_10 > one_event.log_bayes_factor_10
    assert population.credible_interval[0] < beta_true < population.credible_interval[1]


def test_shared_beta_null_prefers_gr_for_zero_centered_likelihoods():
    beta_grid = np.linspace(-0.02, 0.02, 161)
    event_log_likelihood = -0.5 * (beta_grid / 0.002) ** 2

    result = infer_shared_beta(
        beta_grid, np.tile(event_log_likelihood, (16, 1))
    )

    assert abs(result.posterior_mean) < 1e-6
    assert result.log_bayes_factor_10 < 0.0


def test_shared_beta_rejects_malformed_likelihood_inputs():
    beta_grid = np.linspace(-0.02, 0.02, 5)

    with pytest.raises(ValueError, match="beta=0"):
        infer_shared_beta(beta_grid + 0.001, np.zeros((2, 5)))
    with pytest.raises(ValueError, match="forma"):
        infer_shared_beta(beta_grid, np.zeros((2, 4)))
    with pytest.raises(ValueError, match="finitos"):
        infer_shared_beta(beta_grid, np.full((2, 5), np.nan))


def test_phase_likelihood_profiles_out_event_phase_and_time_offsets():
    beta_grid = np.linspace(-0.004, 0.004, 161)
    frequencies = np.array([25.0, 40.0, 60.0, 90.0, 140.0, 220.0, 350.0, 500.0])
    basis = (frequencies / 100.0) ** (-7.0 / 3.0)
    rng = np.random.default_rng(17)
    beta_true = 0.0015
    phase_residuals = np.stack([
        beta_true * basis
        + rng.uniform(-0.2, 0.2)
        + rng.uniform(-0.1, 0.1) * frequencies / 100.0
        + rng.normal(0.0, 0.01, len(frequencies))
        for _ in range(24)
    ])
    covariance = np.eye(len(frequencies)) * 0.01**2

    log_likelihoods = phase_residual_log_likelihoods(
        beta_grid, phase_residuals, frequencies, covariance
    )
    shifted_log_likelihoods = phase_residual_log_likelihoods(
        beta_grid,
        phase_residuals
        + rng.uniform(-0.3, 0.3, (24, 1))
        + rng.uniform(-0.1, 0.1, (24, 1)) * frequencies / 100.0,
        frequencies,
        covariance,
    )
    population = infer_shared_beta(beta_grid, log_likelihoods)

    np.testing.assert_allclose(shifted_log_likelihoods, log_likelihoods, atol=1e-8)
    assert population.credible_interval[0] < beta_true < population.credible_interval[1]
    assert population.log_bayes_factor_10 > 0.0


def test_phase_likelihood_rejects_degenerate_frequencies():
    with pytest.raises(ValueError, match="dos frecuencias"):
        phase_residual_log_likelihoods(
            np.array([-0.1, 0.0, 0.1]),
            np.zeros((1, 3)),
            np.array([100.0, 100.0, 100.0]),
            np.eye(3),
        )
