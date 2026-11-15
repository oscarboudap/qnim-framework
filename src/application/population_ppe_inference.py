"""Combine event-level likelihoods for a shared ppE deformation parameter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class SharedBetaInference:
    beta_grid: np.ndarray
    posterior_mass: np.ndarray
    posterior_mean: float
    credible_interval: tuple[float, float]
    log_bayes_factor_10: float
    n_events: int


def infer_shared_beta(
    beta_grid: np.ndarray,
    event_log_likelihoods: np.ndarray,
    credible_level: float = 0.95,
) -> SharedBetaInference:
    """Infer one population-wide beta from per-event, nuisance-marginalized likelihoods.

    ``event_log_likelihoods[i, j]`` must be the log likelihood for event ``i``
    at ``beta_grid[j]``, after marginalizing that event's source parameters.
    The alternative uses a uniform continuous prior over the supplied beta
    interval; the null is the point hypothesis beta=0.
    """
    beta = np.asarray(beta_grid, dtype=float)
    log_likelihoods = np.asarray(event_log_likelihoods, dtype=float)
    if beta.ndim != 1 or beta.size < 3 or not np.isfinite(beta).all():
        raise ValueError("beta_grid debe ser un vector finito de al menos 3 puntos")
    if np.any(np.diff(beta) <= 0):
        raise ValueError("beta_grid debe ser estrictamente creciente")
    zero_indices = np.flatnonzero(beta == 0.0)
    if zero_indices.size != 1:
        raise ValueError("beta_grid debe contener exactamente el punto beta=0")
    if log_likelihoods.ndim != 2 or log_likelihoods.shape[1] != beta.size:
        raise ValueError("event_log_likelihoods debe tener forma (eventos, len(beta_grid))")
    if log_likelihoods.shape[0] < 1 or not np.isfinite(log_likelihoods).all():
        raise ValueError("Se requiere al menos un evento con log-likelihoods finitos")
    if not 0.0 < credible_level < 1.0:
        raise ValueError("credible_level debe estar entre 0 y 1")

    widths = np.empty_like(beta)
    widths[0] = (beta[1] - beta[0]) / 2.0
    widths[-1] = (beta[-1] - beta[-2]) / 2.0
    widths[1:-1] = (beta[2:] - beta[:-2]) / 2.0
    prior_density = 1.0 / (beta[-1] - beta[0])

    centered_log_likelihoods = log_likelihoods - np.max(
        log_likelihoods, axis=1, keepdims=True
    )
    population_log_likelihood = np.sum(centered_log_likelihoods, axis=0)
    log_integrand = (
        population_log_likelihood + np.log(widths) + np.log(prior_density)
    )
    log_evidence_1 = float(logsumexp(log_integrand))
    posterior_mass = np.exp(log_integrand - log_evidence_1)
    posterior_mean = float(np.sum(beta * posterior_mass))

    tail_probability = (1.0 - credible_level) / 2.0
    cumulative_mass = np.cumsum(posterior_mass)
    credible_interval = tuple(
        float(value)
        for value in np.interp(
            [tail_probability, 1.0 - tail_probability],
            cumulative_mass,
            beta,
        )
    )
    log_likelihood_0 = float(population_log_likelihood[zero_indices[0]])

    return SharedBetaInference(
        beta_grid=beta.copy(),
        posterior_mass=posterior_mass,
        posterior_mean=posterior_mean,
        credible_interval=credible_interval,
        log_bayes_factor_10=log_evidence_1 - log_likelihood_0,
        n_events=log_likelihoods.shape[0],
    )
