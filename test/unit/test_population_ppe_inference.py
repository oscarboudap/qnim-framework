import numpy as np
import pytest

from src.application.population_ppe_inference import infer_shared_beta


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
