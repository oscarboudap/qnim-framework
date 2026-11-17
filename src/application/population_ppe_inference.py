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


def phase_residual_log_likelihoods(
    beta_grid: np.ndarray,
    phase_residuals: np.ndarray,
    frequency_centers_hz: np.ndarray,
    phase_covariances: np.ndarray,
) -> np.ndarray:
    """Build event likelihoods for the ppE -1PN phase law.

    The residual in each frequency band is modeled as
    ``beta * (f / 100 Hz)**(-7/3)`` plus event-specific constant and linear
    frequency terms. Those two nuisance terms represent arbitrary phase and
    coalescence-time offsets and are profiled out with generalized least
    squares. Covariance matrices may be shared or event-specific.
    """
    beta = np.asarray(beta_grid, dtype=float)
    residuals = np.asarray(phase_residuals, dtype=float)
    frequencies = np.asarray(frequency_centers_hz, dtype=float)
    covariances = np.asarray(phase_covariances, dtype=float)
    if beta.ndim != 1 or beta.size < 1 or not np.isfinite(beta).all():
        raise ValueError("beta_grid debe ser un vector finito no vacío")
    if residuals.ndim != 2 or residuals.shape[0] < 1 or residuals.shape[1] < 3:
        raise ValueError("phase_residuals debe tener forma (eventos, al menos 3 bandas)")
    if not np.isfinite(residuals).all():
        raise ValueError("phase_residuals debe contener solo valores finitos")

    n_events, n_bands = residuals.shape
    if frequencies.ndim == 1 and frequencies.shape == (n_bands,):
        frequencies = np.broadcast_to(frequencies, (n_events, n_bands))
    if frequencies.shape != (n_events, n_bands):
        raise ValueError("frequency_centers_hz debe ser bandas o eventos x bandas")
    if not np.isfinite(frequencies).all() or np.any(frequencies <= 0):
        raise ValueError("Las frecuencias centrales deben ser positivas y finitas")

    if covariances.ndim == 2 and covariances.shape == (n_bands, n_bands):
        covariances = np.broadcast_to(
            covariances, (n_events, n_bands, n_bands)
        )
    if covariances.shape != (n_events, n_bands, n_bands):
        raise ValueError("phase_covariances debe ser bandas x bandas o eventos x bandas x bandas")
    if not np.isfinite(covariances).all():
        raise ValueError("phase_covariances debe contener solo valores finitos")

    if not np.allclose(
        covariances,
        np.swapaxes(covariances, 1, 2),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Cada matriz de covarianza debe ser simétrica")
    try:
        np.linalg.cholesky(covariances)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Cada matriz de covarianza debe ser definida positiva") from exc

    nuisance_design = np.stack(
        (np.ones_like(frequencies), frequencies / 100.0), axis=-1
    )
    if np.any(np.linalg.matrix_rank(nuisance_design) < 2):
        raise ValueError(
            "Cada evento debe tener al menos dos frecuencias centrales distintas"
        )
    basis = (frequencies / 100.0) ** (-7.0 / 3.0)
    weighted_nuisance = np.linalg.solve(covariances, nuisance_design)
    nuisance_information = np.einsum(
        "nbi,nbj->nij", nuisance_design, weighted_nuisance
    )
    weighted_residual = np.linalg.solve(
        covariances, residuals[..., np.newaxis]
    )[..., 0]
    weighted_basis = np.linalg.solve(
        covariances, basis[..., np.newaxis]
    )[..., 0]
    residual_score = np.einsum(
        "nbi,nb->ni", nuisance_design, weighted_residual
    )
    basis_score = np.einsum("nbi,nb->ni", nuisance_design, weighted_basis)
    profiled_residual_score = np.linalg.solve(
        nuisance_information, residual_score[..., np.newaxis]
    )[..., 0]
    profiled_basis_score = np.linalg.solve(
        nuisance_information, basis_score[..., np.newaxis]
    )[..., 0]
    residual_chi_squared = (
        np.einsum("nb,nb->n", residuals, weighted_residual)
        - np.einsum("ni,ni->n", residual_score, profiled_residual_score)
    )
    cross_term = (
        np.einsum("nb,nb->n", residuals, weighted_basis)
        - np.einsum("ni,ni->n", residual_score, profiled_basis_score)
    )
    basis_chi_squared = (
        np.einsum("nb,nb->n", basis, weighted_basis)
        - np.einsum("ni,ni->n", basis_score, profiled_basis_score)
    )
    chi_squared = (
        residual_chi_squared[:, np.newaxis]
        - 2.0 * cross_term[:, np.newaxis] * beta[np.newaxis, :]
        + basis_chi_squared[:, np.newaxis] * beta[np.newaxis, :] ** 2
    )
    if np.any(chi_squared < -1e-8):
        raise RuntimeError("La proyección de nuisance produjo chi-cuadrado negativo")
    return -0.5 * np.maximum(chi_squared, 0.0)


def marginalize_source_parameter_log_likelihoods(
    source_log_likelihoods: np.ndarray,
    source_log_prior_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Marginalize discrete source hypotheses for each event and beta value.

    ``source_log_likelihoods`` has shape ``(events, source hypotheses, beta)``.
    Prior weights may be shared across events or specified per event; they are
    normalized internally. The result has shape ``(events, beta)``.
    """
    log_likelihoods = np.asarray(source_log_likelihoods, dtype=float)
    if (
        log_likelihoods.ndim != 3
        or min(log_likelihoods.shape) < 1
        or not np.isfinite(log_likelihoods).all()
    ):
        raise ValueError(
            "source_log_likelihoods debe ser un array finito "
            "(eventos, hipótesis de fuente, beta)"
        )

    n_events, n_sources, _ = log_likelihoods.shape
    if source_log_prior_weights is None:
        normalized_log_prior = np.full(
            (n_events, n_sources), -np.log(n_sources), dtype=float
        )
    else:
        log_prior = np.asarray(source_log_prior_weights, dtype=float)
        if log_prior.ndim == 1 and log_prior.shape == (n_sources,):
            log_prior = np.broadcast_to(log_prior, (n_events, n_sources))
        if log_prior.shape != (n_events, n_sources):
            raise ValueError(
                "source_log_prior_weights debe ser hipótesis o eventos x hipótesis"
            )
        if np.isnan(log_prior).any() or np.isposinf(log_prior).any():
            raise ValueError("Los pesos log-prior deben ser finitos o -inf")
        normalization = logsumexp(log_prior, axis=1, keepdims=True)
        if not np.isfinite(normalization).all():
            raise ValueError("Cada evento debe tener al menos una hipótesis con prior positivo")
        normalized_log_prior = log_prior - normalization

    return logsumexp(
        log_likelihoods + normalized_log_prior[..., np.newaxis], axis=1
    )


def infer_shared_beta(
    beta_grid: np.ndarray,
    event_log_likelihoods: np.ndarray,
    credible_level: float = 0.95,
) -> SharedBetaInference:
    """Infer one population-wide beta from supplied per-event likelihoods.

    ``event_log_likelihoods[i, j]`` must be the log likelihood for event ``i``
    at ``beta_grid[j]``. This function does not profile or marginalize event
    parameters; callers must account for source parameters before interpreting
    the result physically. The alternative uses a uniform continuous prior
    over the supplied beta interval; the null is the point hypothesis beta=0.
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
