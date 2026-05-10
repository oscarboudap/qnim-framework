"""
src/infrastructure/qubo_match_ligo.py
CORREGIDO: templates proyectados al mismo espacio que observed_features.
"""
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger("qnim.infrastructure.qubo_match_ligo")


def ligo_o3_psd(freqs: np.ndarray) -> np.ndarray:
    f0, S0 = 150.0, 4.8e-46
    psd = np.zeros_like(freqs, dtype=float)
    for i, f in enumerate(freqs):
        if f < 20.0:     psd[i] = 1e30
        elif f < 30.0:   psd[i] = S0 * (f0/f)**20
        elif f < 80.0:   psd[i] = S0 * 4.0 * (f0/f)**4
        elif f < 200.0:  psd[i] = S0 * (1.0 + (f0/f)**2)
        else:            psd[i] = S0 * (1.0 + (f/f0)**2)
    return np.maximum(psd, 1e-50)


def _pn_features_from_params(m1: float, m2: float, chi_eff: float,
                               n_features: int = 12) -> np.ndarray:
    """
    Proyecta parametros fisicos al MISMO espacio que usa el SSTG adapter.
    El SSTG toma |FFT|[:12] normalizado de una senal con amplitud f^{-7/6}.
    Replicamos esa fisica aqui para que el match function opere en el
    mismo espacio de Hilbert que X_train.
    """
    M = m1 + m2
    M_c = (m1 * m2)**0.6 / M**0.2
    freqs = np.arange(1, n_features + 1, dtype=float) * 20.0  # 20-240 Hz
    psd_w = ligo_o3_psd(freqs)
    amp = (M_c / 30.0)**(5.0/6.0) * freqs**(-7.0/6.0)
    amp_whitened = amp / np.sqrt(psd_w)
    phase_mod = 1.0 + 0.15 * chi_eff * (M_c/30.0)**(1.0/3.0) / freqs**(2.0/3.0)
    phase_mod = np.clip(phase_mod, 0.5, 2.0)
    f_isco = 4397.0 / M
    isco_window = np.exp(-0.5 * (freqs / f_isco)**2)
    features = amp_whitened * phase_mod * isco_window
    norm = np.max(np.abs(features))
    if norm > 1e-30:
        features = features / norm
    return features.astype(float)


@dataclass
class LIGOMatchQUBOResult:
    m1_msun: float
    m2_msun: float
    chi_eff: float
    best_match: float
    best_template_idx: int
    snr_optimal: float
    qubo_linear: dict
    qubo_quadratic: dict
    is_gr_consistent: bool


def build_ligo_match_qubo(observed_features: np.ndarray, n_templates: int = 64,
                           sample_rate: float = 4096.0, penalty_weight_factor: float = 10.0,
                           seed: int = 42) -> LIGOMatchQUBOResult:
    rng = np.random.default_rng(seed)
    n_feat = len(observed_features)
    side = max(2, int(np.ceil(n_templates**(1.0/3.0))))
    m1_grid = np.linspace(20.0, 50.0, side)
    m2_grid = np.linspace(15.0, 45.0, side)
    chi_grid = np.linspace(-0.5, 0.5, side)
    templates_params = []
    for m1 in m1_grid:
        for m2 in m2_grid:
            for chi in chi_grid:
                if len(templates_params) >= n_templates: break
                if m2 <= m1: templates_params.append((m1, m2, chi))
            if len(templates_params) >= n_templates: break
        if len(templates_params) >= n_templates: break
    templates_params = templates_params[:n_templates]
    n_t = len(templates_params)
    logger.info(f"Construyendo cuadricula de {n_t} templates 3.5PN...")
    template_features = np.zeros((n_t, n_feat))
    for idx, (m1, m2, chi) in enumerate(templates_params):
        template_features[idx] = _pn_features_from_params(m1, m2, chi, n_feat)
    obs = observed_features.copy().astype(float)
    obs_norm = np.linalg.norm(obs)
    obs_normalized = obs / obs_norm if obs_norm > 1e-30 else obs
    matches = template_features @ obs_normalized
    matches = np.clip(matches, -1.0, 1.0)
    one_minus_m2 = 1.0 - matches**2
    P = penalty_weight_factor * float(np.max(one_minus_m2)) + 0.01
    qubo_linear = {i: float(one_minus_m2[i]) for i in range(n_t)}
    qubo_quadratic = {(i, j): float(P) for i in range(n_t) for j in range(i+1, n_t)}
    best_idx = int(np.argmax(matches))
    best_m1, best_m2, best_chi = templates_params[best_idx]
    best_match = float(matches[best_idx])
    snr_estimate = max(0.0, best_match * 24.0)
    logger.info(f"Mejor template: m1={best_m1:.1f}, m2={best_m2:.1f}, "
                f"chi_eff={best_chi:.3f}, M={best_match:.4f}, SNR_est={snr_estimate:.1f}")
    return LIGOMatchQUBOResult(
        m1_msun=float(best_m1), m2_msun=float(best_m2), chi_eff=float(best_chi),
        best_match=best_match, best_template_idx=best_idx, snr_optimal=snr_estimate,
        qubo_linear=qubo_linear, qubo_quadratic=qubo_quadratic,
        is_gr_consistent=(best_match > 0.85),
    )
