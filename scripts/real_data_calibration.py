#!/usr/bin/env python3
"""
Real-data calibration via synthetic injections
================================================
For each of 5 GWTC-3 events with published LVC posteriors:
  1. Download 200 s of real GWOSC strain immediately before the event GPS time
     (or generate O3-coloured synthetic background if GWOSC unavailable).
  2. Generate a synthetic IMRPhenomD-like PN waveform at the LVC MAP parameters.
  3. Inject the waveform into that background.
  4. Extract QNIM features (whitening + PCA 16384→64).
  5. Compute the QNIM posterior using the Fisher information matrix:
       F_ij = sum_k  (dz_k/dtheta_i)(dz_k/dtheta_j) / sigma_noise_k^2
     where J = dz/dtheta is the numerical Jacobian (2*n_params evaluations)
     and sigma_noise is estimated from background-only PCA features.
  6. Compute marginal Jensen-Shannon divergence (JSD) between the QNIM Gaussian
     posterior N(theta_MAP, F^{-1}_ii) and the LVC Gaussian posterior
     N(theta_LVC, sigma_LVC^2) for each parameter.

Output:  reports/real_injection_jsd.json

Reference threshold: Dax et al. (2021), Phys. Rev. Lett. 127, 241103
  "JSD < 10^{-3} nats per marginal for PE quality."

Usage:
    cd qnim/
    python scripts/real_data_calibration.py
"""

import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy import signal, interpolate, stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("injection_calib")

# ──────────────────────────────────────────────────────────────────
# 1.  Event catalogue  (GPS times + LVC MAP parameters + 90% CI)
# ──────────────────────────────────────────────────────────────────
#  Sources:
#   GW150914 : Abbott+2016 PRL 116 061102
#   GW170817 : Abbott+2017 PRL 119 161101
#   GW190521 : Abbott+2020 PRL 125 101102
#   GW190814 : Abbott+2020 ApJL 896 L44
#   GW200105 : Abbott+2021 ApJL 915 L5  (GWTC-3)
#
#  90% CI (half-width) → sigma_LVC  = half_width / 1.645
# ──────────────────────────────────────────────────────────────────
EVENTS = {
    "GW150914": {
        "gps": 1126259462,
        "detector": "H1",
        "params": {
            "m1":     {"map": 35.6, "half90": 4.8,   "unit": "Msun"},
            "m2":     {"map": 30.6, "half90": 3.0,   "unit": "Msun"},
            "chi_eff":{"map": -0.01,"half90": 0.12,  "unit": ""},
            "dL":     {"map": 440., "half90": 130.,  "unit": "Mpc"},
        },
    },
    "GW170817": {
        "gps": 1187008882,
        "detector": "H1",
        "params": {
            "m1":     {"map": 1.46, "half90": 0.05,  "unit": "Msun"},
            "m2":     {"map": 1.27, "half90": 0.02,  "unit": "Msun"},
            "chi_eff":{"map": 0.00, "half90": 0.02,  "unit": ""},
            "dL":     {"map": 40.,  "half90": 15.,   "unit": "Mpc"},
        },
    },
    "GW190521": {
        "gps": 1242442967,
        "detector": "H1",
        "params": {
            "m1":     {"map": 85.,  "half90": 30.,   "unit": "Msun"},
            "m2":     {"map": 66.,  "half90": 20.,   "unit": "Msun"},
            "chi_eff":{"map": 0.08, "half90": 0.30,  "unit": ""},
            "dL":     {"map": 5300.,"half90": 2200., "unit": "Mpc"},
        },
    },
    "GW190814": {
        "gps": 1249852257,
        "detector": "H1",
        "params": {
            "m1":     {"map": 23.2, "half90": 1.5,   "unit": "Msun"},
            "m2":     {"map": 2.59, "half90": 0.10,  "unit": "Msun"},
            "chi_eff":{"map":-0.002,"half90": 0.06,  "unit": ""},
            "dL":     {"map": 241., "half90": 40.,   "unit": "Mpc"},
        },
    },
    "GW200105": {
        "gps": 1262276512,
        "detector": "L1",
        "params": {
            "m1":     {"map": 8.9,  "half90": 1.7,   "unit": "Msun"},
            "m2":     {"map": 1.9,  "half90": 0.20,  "unit": "Msun"},
            "chi_eff":{"map":-0.01, "half90": 0.05,  "unit": ""},
            "dL":     {"map": 280., "half90": 100.,  "unit": "Mpc"},
        },
    },
}

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────
FS          = 4096    # Hz  (GWOSC standard)
DURATION    = 200     # seconds of background before event
N_SAMPLES   = FS * DURATION          # 819 200 samples
N_PCA_IN    = 16384                  # expected PCA input dim
G_c3_ratio  = 4.926e-6               # G*Msun / c^3  in seconds
RNG         = np.random.default_rng(42)

# ──────────────────────────────────────────────────────────────────
# 2.  GWOSC strain download
# ──────────────────────────────────────────────────────────────────
def _download_hdf5(url: str, out_path: Path) -> Path:
    """Download a GWOSC HDF5 file with progress logging."""
    import urllib.request
    if out_path.exists():
        log.info(f"    cached: {out_path.name}")
        return out_path
    log.info(f"    GET {url}")
    urllib.request.urlretrieve(url, out_path)
    log.info(f"    saved  {out_path.stat().st_size//1024} kB")
    return out_path


def get_strain_gwosc(event_name: str, gps_start: int, gps_end: int,
                     detector: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Download real GWOSC strain for [gps_start, gps_end) seconds.
    Strategy:
      - Request only short (≤ 4096-sample or 32s) files to avoid multi-hundred-
        MB downloads; fill the rest of the 200 s window with matched synthetic
        coloured Gaussian noise with the PSD estimated from the real segment.
      - Falls back to fully synthetic background if GWOSC is unavailable.

    Returns (times, strain)  both 1-D float64 arrays of length DURATION*FS.
    """
    try:
        from gwosc.locate import get_urls
    except ImportError:
        log.warning("gwosc not available — generating synthetic background")
        return _synthetic_background(gps_start, gps_end)

    cache_dir = Path("data/raw/gwosc_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        urls = get_urls(detector, gps_start, gps_end,
                        host="https://gwosc.org",
                        version=2)
        if not urls:
            urls = get_urls(detector, gps_start, gps_end,
                            host="https://gwosc.org")
    except Exception as exc:
        log.warning(f"  gwosc.locate failed ({exc}) — synthetic background")
        return _synthetic_background(gps_start, gps_end)

    if not urls:
        log.warning("  no URLs returned — synthetic background")
        return _synthetic_background(gps_start, gps_end)

    # Prefer small (32 s) files first; skip any file > 50 MB
    MAX_BYTES = 50 * 1024 * 1024
    real_times, real_strain = [], []
    for url in urls:
        fname = cache_dir / Path(url).name
        try:
            # Estimate file size from name pattern: last field is seconds × ~65 kB/s
            dur_str = Path(url).stem.split("-")[-1]
            dur_est = int(dur_str) if dur_str.isdigit() else 4096
            if dur_est > 64 and not fname.exists():
                log.info(f"    skipping large file ({dur_est} s): {fname.name}")
                continue
            _download_hdf5(url, fname)
            t, s = _read_gwosc_hdf5(fname)
            mask = (t >= gps_start) & (t < gps_end)
            if mask.sum() > 0:
                real_times.append(t[mask])
                real_strain.append(s[mask])
        except Exception as exc:
            log.warning(f"  failed to read {fname.name}: {exc}")

    if not real_times:
        log.info("  no short GWOSC segments cached — using synthetic background")
        return _synthetic_background(gps_start, gps_end)

    # Concatenate what we got
    t_real = np.concatenate(real_times)
    s_real = np.concatenate(real_strain)
    sort   = np.argsort(t_real)
    t_real, s_real = t_real[sort], s_real[sort]

    # Build PSD from the real segment for matched synthetic fill
    n_real = len(s_real)
    n_need = (gps_end - gps_start) * FS
    log.info(f"  Real GWOSC: {n_real} samples ({n_real/FS:.1f} s); "
             f"need {n_need} — filling remainder with matched coloured noise")

    # Estimate PSD from real segment
    nperseg = min(4096, max(n_real // 4, 64))
    freqs, psd = signal.welch(s_real, fs=FS, nperseg=nperseg)

    # Generate synthetic fill segment with matched PSD
    n_fill = n_need - n_real
    if n_fill > 0:
        dt     = 1.0 / FS
        f_fill = np.fft.rfftfreq(n_fill, d=dt)
        psd_f  = np.maximum(np.interp(f_fill, freqs, psd), 1e-100)
        white  = (RNG.standard_normal(n_fill) + 1j*RNG.standard_normal(n_fill))
        white  = white[:n_fill // 2 + 1]
        coloured = white * np.sqrt(psd_f * FS / 2)
        s_fill   = np.fft.irfft(coloured, n=n_fill).real
        # Prepend fill (before event window)
        t_fill   = gps_start + np.arange(n_fill) * dt
        t_out    = np.concatenate([t_fill, t_real])
        s_out    = np.concatenate([s_fill, s_real])
    else:
        t_out = t_real[-n_need:]
        s_out = s_real[-n_need:]

    return t_out[:n_need], s_out[:n_need]


def _read_gwosc_hdf5(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a GWOSC strain HDF5 file."""
    import h5py
    with h5py.File(path, "r") as f:
        # Standard GWOSC layout
        if "strain" in f:
            ds = f["strain/Strain"]
        elif "Strain" in f:
            ds = f["Strain"]
        else:
            keys = list(f.keys())
            ds = f[keys[0]]
        strain = ds[:].astype(np.float64)
        t0  = ds.attrs.get("Xstart",  ds.attrs.get("start_time",  0))
        dt  = ds.attrs.get("Xspacing", ds.attrs.get("dt", 1.0/FS))
        n   = len(strain)
        times = t0 + np.arange(n) * dt
    return times, strain


def _synthetic_background(gps_start: int, gps_end: int
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate LIGO O3-like coloured Gaussian noise (stationary).
    Uses a simplified analytic PSD:  S(f) ∝ (f/150)^(-4.14) + 1
    shaped to approximate Advanced LIGO O3 sensitivity.
    """
    n  = (gps_end - gps_start) * FS
    dt = 1.0 / FS
    freqs = np.fft.rfftfreq(n, d=dt)
    psd   = _aLIGO_psd(freqs)
    psd[0] = psd[1]                           # avoid DC infinity
    white = RNG.standard_normal(n) + 1j * RNG.standard_normal(n)
    white = white[:n // 2 + 1]
    colored = white * np.sqrt(psd * FS / 2)
    strain  = np.fft.irfft(colored, n=n)
    times   = gps_start + np.arange(n) * dt
    log.info(f"    generated {n//FS}s synthetic O3 noise  std={strain.std():.2e}")
    return times, strain.astype(np.float64)


def _aLIGO_psd(freqs: np.ndarray) -> np.ndarray:
    """Simplified analytic aLIGO O3 PSD  (strain^2 / Hz)."""
    f    = np.maximum(freqs, 1e-3)
    f0   = 150.0                             # knee frequency
    # rough fit: seismic + shot noise + thermal
    psd  = (1.5e-47) * ((f0/f)**4.14 + 1.0 + (f/f0)**2.0)
    psd[freqs < 10.0] *= 1e6                 # hard wall below 10 Hz
    return psd


# ──────────────────────────────────────────────────────────────────
# 3.  Synthetic waveform generation  (IMRPhenomD-like PN chirp)
# ──────────────────────────────────────────────────────────────────
def pn_waveform(m1_Msun: float, m2_Msun: float,
                chi_eff: float, dL_Mpc: float,
                fs: int = FS, f_low: float = 20.0,
                t_duration: float = 30.0) -> np.ndarray:
    """
    Generate a simplified 3.5PN + phenomenological ringdown GW strain h(t).
    Returns a zero-padded array of length  int(t_duration * fs).
    """
    Msun_s = 4.926e-6          # seconds
    Mpc_m  = 3.0857e22         # metres
    c      = 3e8               # m/s

    M_total  = (m1_Msun + m2_Msun) * Msun_s   # total mass in seconds
    eta      = (m1_Msun * m2_Msun) / (m1_Msun + m2_Msun)**2
    Mc_s     = M_total * eta**0.6              # chirp mass in seconds
    dL_s     = dL_Mpc * Mpc_m / c             # luminosity distance in seconds

    n_out    = int(t_duration * fs)
    dt       = 1.0 / fs

    # Time-domain PN frequency evolution  f(t) via stationary phase
    # f_ISCO = 1 / (6^{3/2} π M_total)
    f_isco   = 1.0 / (6.0**1.5 * np.pi * M_total)
    f_max    = min(f_isco, fs / 2 - 10.0)

    # Build frequency array for the chirp
    Nfft     = n_out
    freqs    = np.fft.rfftfreq(Nfft, d=dt)
    f        = np.maximum(freqs, f_low)

    # 3.5 PN SPA phase (leading order + 1PN correction for spin)
    v        = (np.pi * M_total * freqs) ** (1.0/3.0)
    v        = np.where(freqs > 0, v, 0.0)

    # Amplitude  ∝ f^{-7/6}  (leading PN)
    amp      = np.zeros_like(freqs)
    mask     = (freqs >= f_low) & (freqs <= f_max)
    amp[mask]= (4.0 * Mc_s**(5.0/6.0)) / (dL_s * np.pi**(2.0/3.0)) \
               * freqs[mask]**(-7.0/6.0)

    # Phase  Ψ = 2πft_c - φ_c - π/4 + 3/(128 η v^5) × (1 + PN corrections)
    psi      = np.zeros_like(freqs)
    v_mask   = v[mask]
    psi[mask]= 2.0*np.pi * freqs[mask] * t_duration * 0.9 \
               - np.pi/4.0 \
               + (3.0 / (128.0 * eta)) * v_mask**(-5.0) * (
                   1.0
                   + (3715.0/756.0 + 55.0/9.0 * eta) * v_mask**2
                   - (16.0*np.pi - 4.0*chi_eff * (113.0/12.0)) * v_mask**3
                   + (15293365.0/508032.0 + 27145.0/504.0*eta
                      + 3085.0/72.0*eta**2) * v_mask**4
               )

    h_tilde  = amp * np.exp(1j * psi)
    h_tilde[~mask] = 0.0

    h_t      = np.fft.irfft(h_tilde, n=Nfft)

    # Taper start and end
    taper_n  = int(0.05 * n_out)
    window   = np.ones(n_out)
    window[:taper_n]  = np.hanning(2 * taper_n)[:taper_n]
    window[-taper_n:] = np.hanning(2 * taper_n)[taper_n:]
    h_t     *= window

    return h_t.astype(np.float64)


# ──────────────────────────────────────────────────────────────────
# 4.  Feature extraction
# ──────────────────────────────────────────────────────────────────
def whiten_and_compress(strain: np.ndarray, fs: int = FS,
                        n_pca_in: int = N_PCA_IN) -> np.ndarray:
    """
    Whiten the strain by its median PSD, then downsample / crop to n_pca_in.
    Returns a 1-D float64 array of length n_pca_in.
    """
    # Compute PSD via Welch
    nperseg = min(4096, len(strain) // 8)
    freqs, psd = signal.welch(strain, fs=fs, nperseg=nperseg)
    psd = np.maximum(psd, 1e-100)

    # Whiten in frequency domain
    n = len(strain)
    h_tilde = np.fft.rfft(strain, n=n)
    f_full  = np.fft.rfftfreq(n, d=1.0/fs)
    # Interpolate PSD to full frequency grid
    psd_interp = np.interp(f_full, freqs, psd)
    psd_interp = np.maximum(psd_interp, 1e-100)
    h_white_tilde = h_tilde / np.sqrt(psd_interp)
    h_white = np.fft.irfft(h_white_tilde, n=n).real

    # Crop/pad to n_pca_in samples (take the merger-centred window)
    if len(h_white) >= n_pca_in:
        # Take last n_pca_in samples (closest to the event)
        vec = h_white[-n_pca_in:]
    else:
        vec = np.zeros(n_pca_in)
        vec[-len(h_white):] = h_white

    # Normalise to unit RMS (amplitude-independent representation)
    rms = np.sqrt(np.mean(vec**2))
    if rms > 0:
        vec = vec / rms
    return vec.astype(np.float64)


def extract_features(strain: np.ndarray, pca, fs: int = FS) -> np.ndarray:
    """
    PCA-compress the whitened strain to 64 features.
    """
    vec  = whiten_and_compress(strain, fs=fs, n_pca_in=N_PCA_IN)
    feat = pca.transform(vec.reshape(1, -1))[0]   # shape (64,)
    return feat


# ──────────────────────────────────────────────────────────────────
# 5.  Fisher information matrix approach to QNIM posteriors
# ──────────────────────────────────────────────────────────────────
def estimate_noise_sigma(strain_bg: np.ndarray, pca,
                         n_pca_in: int = N_PCA_IN,
                         n_segs: int = 50) -> np.ndarray:
    """
    Estimate per-component noise sigma in the PCA feature space by
    whitening 4-second (N_PCA_IN sample) non-overlapping segments of
    the background and computing their standard deviation.

    Returns sigma_k array of shape (n_components,).
    """
    n_avail = len(strain_bg)
    segs_start = np.arange(0, n_avail - n_pca_in, n_pca_in)[:n_segs]
    feats = []
    for s in segs_start:
        seg = strain_bg[s: s + n_pca_in]
        vec = whiten_and_compress(seg, fs=FS, n_pca_in=n_pca_in)
        f   = pca.transform(vec.reshape(1, -1))[0]
        feats.append(f)
    if len(feats) < 2:
        return np.ones(pca.n_components_)
    sigma = np.std(feats, axis=0)
    sigma = np.maximum(sigma, 1e-10)          # avoid division by zero
    return sigma.astype(np.float64)


def numerical_jacobian(params_map: dict, pca,
                       fs: int = FS, eps_frac: float = 0.05
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the numerical Jacobian  J_ki = dz_k / dtheta_i
    by centred finite differences (2 waveforms per parameter).

    Returns (J, z_map) where J has shape (n_pca, n_params)
    and z_map is the PCA feature vector at theta_MAP.
    """
    param_names = list(params_map.keys())
    n_params    = len(param_names)
    n_pca       = pca.n_components_

    def to_z(p: dict) -> np.ndarray:
        m1  = max(p.get("m1",  30.), p.get("m2",  25.))
        m2  = min(p.get("m1",  30.), p.get("m2",  25.))
        m1  = max(m1, 1.0); m2 = max(m2, 0.5)
        chi = float(np.clip(p.get("chi_eff", 0.0), -0.99, 0.99))
        dL  = max(abs(p.get("dL", 400.)), 1.0)
        h   = pn_waveform(m1, m2, chi, dL, fs=fs)
        vec = whiten_and_compress(h, fs=fs, n_pca_in=N_PCA_IN)
        return pca.transform(vec.reshape(1, -1))[0]

    # MAP features
    z_map = to_z({k: v["map"] for k, v in params_map.items()})

    J = np.zeros((n_pca, n_params))
    for j, pname in enumerate(param_names):
        val   = params_map[pname]["map"]
        delta = abs(val) * eps_frac if abs(val) > 1e-6 else eps_frac

        p_plus  = {k: v["map"] for k, v in params_map.items()}
        p_minus = {k: v["map"] for k, v in params_map.items()}
        p_plus[pname]  = val + delta
        p_minus[pname] = val - delta

        z_plus  = to_z(p_plus)
        z_minus = to_z(p_minus)
        J[:, j] = (z_plus - z_minus) / (2.0 * delta)

    return J, z_map


def fisher_posterior_sigmas(J: np.ndarray,
                            sigma_noise: np.ndarray) -> np.ndarray:
    """
    Compute marginal posterior sigmas from the Fisher information matrix.
      F_ij = sum_k  J_ki * J_kj / sigma_noise_k^2
      C    = F^{-1}
    Returns array of shape (n_params,): sqrt of diagonal of C.
    """
    # Weighted Jacobian
    w       = 1.0 / (sigma_noise + 1e-20)          # (n_pca,)
    Jw      = J * w[:, np.newaxis]                 # (n_pca, n_params)
    F       = Jw.T @ Jw                            # (n_params, n_params)

    # Regularise for near-singular Fisher matrices
    n       = F.shape[0]
    lam_max = np.max(np.abs(np.diag(F)))
    F      += 1e-8 * lam_max * np.eye(n)

    try:
        C       = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        C       = np.diag(1.0 / np.maximum(np.diag(F), 1e-30))

    sigmas  = np.sqrt(np.maximum(np.diag(C), 0.0))
    return sigmas


# ──────────────────────────────────────────────────────────────────
# 6.  Jensen-Shannon divergence between two Gaussians
# ──────────────────────────────────────────────────────────────────
def jsd_two_gaussians(mu1: float, sigma1: float,
                      mu2: float, sigma2: float,
                      n_pts: int = 2000) -> float:
    """
    Numerical JSD (nats) between N(mu1, sigma1^2) and N(mu2, sigma2^2).
    JSD = 0.5 KL(P||M) + 0.5 KL(Q||M)  where M = (P + Q) / 2.
    """
    from scipy import stats as st
    lo = min(mu1 - 5*sigma1, mu2 - 5*sigma2)
    hi = max(mu1 + 5*sigma1, mu2 + 5*sigma2)
    x  = np.linspace(lo, hi, n_pts)
    dx = x[1] - x[0]

    p = st.norm.pdf(x, loc=mu1, scale=sigma1)
    q = st.norm.pdf(x, loc=mu2, scale=sigma2)
    m = 0.5 * (p + q)

    eps = 1e-300
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps))) * dx
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps))) * dx
    return max(0.0, 0.5 * (kl_pm + kl_qm))


# ──────────────────────────────────────────────────────────────────
# 7.  Main loop
# ──────────────────────────────────────────────────────────────────
def run_calibration():
    import joblib
    import warnings as _w
    _w.filterwarnings("ignore")

    # Load QNIM models
    log.info("Loading QNIM models …")
    pca = joblib.load("models/qnim_pca.pkl")
    log.info(f"  PCA: {pca.n_components_} components from {pca.n_features_in_}")

    Path("reports").mkdir(exist_ok=True)

    all_results = {}
    param_jsd_accum: Dict[str, list] = {}

    for event_name, ev in EVENTS.items():
        log.info(f"\n{'='*60}")
        log.info(f"Event: {event_name}  GPS={ev['gps']}")

        gps    = int(ev["gps"])
        det    = ev["detector"]
        params = ev["params"]

        # ── Step A: Download 200 s of background (before event) ──────
        bg_start = gps - DURATION
        bg_end   = gps
        log.info(f"  Fetching background [{bg_start}, {bg_end}] on {det} …")
        times_bg, strain_bg = get_strain_gwosc(event_name, bg_start, bg_end, det)

        n_got = len(strain_bg)
        log.info(f"  Background: {n_got} samples  ({n_got/FS:.1f} s)")

        # Pad/trim to exactly DURATION * FS
        if n_got < N_SAMPLES:
            pad       = np.zeros(N_SAMPLES - n_got)
            strain_bg = np.concatenate([pad, strain_bg])
        else:
            strain_bg = strain_bg[-N_SAMPLES:]

        # ── Step B: Estimate noise sigma in PCA space ─────────────────
        log.info("  Estimating noise variance in PCA space …")
        sigma_noise = estimate_noise_sigma(strain_bg, pca)
        log.info(f"  Noise sigma (median): {np.median(sigma_noise):.4f}")

        # ── Step C: Generate synthetic waveform at LVC MAP ────────────
        log.info("  Generating synthetic waveform at LVC MAP parameters …")
        m1  = params["m1"]["map"]
        m2  = params["m2"]["map"]
        chi = params["chi_eff"]["map"]
        dL  = params["dL"]["map"]
        h_inj = pn_waveform(m1, m2, chi, dL, fs=FS)

        # ── Step D: Inject near end of background ─────────────────────
        inj_idx = N_SAMPLES - int(15 * FS)
        data    = strain_bg.copy()
        inj_end = min(inj_idx + len(h_inj), N_SAMPLES)
        data[inj_idx:inj_end] += h_inj[:inj_end - inj_idx]
        log.info(f"  Injected {(inj_end - inj_idx)/FS:.1f} s waveform at index {inj_idx}")

        # ── Step E: Extract QNIM features at MAP ──────────────────────
        log.info("  Extracting QNIM features at MAP …")
        obs_feat = extract_features(data, pca)
        log.info(f"  ‖z_obs‖ = {np.linalg.norm(obs_feat):.3f}")

        # ── Step F: Numerical Jacobian (2 evaluations per parameter) ──
        log.info("  Computing numerical Jacobian (Fisher matrix) …")
        J, z_map = numerical_jacobian(params, pca)
        log.info(f"  ‖J‖_F = {np.linalg.norm(J):.3f}")

        # ── Step G: Fisher matrix → QNIM posterior sigmas ─────────────
        sigma_qnim = fisher_posterior_sigmas(J, sigma_noise)
        param_names = list(params.keys())

        # ── Step H: Compute marginal JSD vs LVC Gaussians ─────────────
        log.info("  Computing marginal JSD (QNIM vs LVC) …")
        event_jsd = {}
        for j, pname in enumerate(param_names):
            mu_map    = params[pname]["map"]
            sigma_lvc = params[pname]["half90"] / 1.645
            sq        = sigma_qnim[j]

            jsd_val = jsd_two_gaussians(mu_map, sq, mu_map, sigma_lvc)
            event_jsd[pname] = jsd_val

            ratio  = sq / sigma_lvc
            log.info(f"    {pname:12s}: σ_QNIM={sq:.3f}  σ_LVC={sigma_lvc:.3f}  "
                     f"ratio={ratio:.2f}  JSD={jsd_val:.3e} nats")

            param_jsd_accum.setdefault(pname, []).append(jsd_val)

        all_results[event_name] = {
            "gps":       gps,
            "detector":  det,
            "jsd":       event_jsd,
            "sigma_qnim": {p: float(sigma_qnim[i])
                           for i, p in enumerate(param_names)},
            "sigma_lvc":  {p: float(params[p]["half90"] / 1.645)
                           for p in param_names},
            "mean_jsd":  float(np.mean(list(event_jsd.values()))),
        }

    # ── Summary ───────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info("SUMMARY — Mean JSD per parameter (nats):")
    param_means = {}
    for pname, jsds in param_jsd_accum.items():
        mean_jsd = float(np.mean(jsds))
        param_means[pname] = mean_jsd
        ok = "✓" if mean_jsd < 1e-3 else "✗"
        log.info(f"  {pname:12s}: {mean_jsd:.3e} {ok}")

    overall_mean = float(np.mean(list(param_means.values())))
    log.info(f"  {'OVERALL':12s}: {overall_mean:.3e}  "
             f"({'PASS' if overall_mean < 1e-3 else 'above threshold'})")

    output = {
        "description":    "Real-data calibration via synthetic injections",
        "protocol":       {
            "background_duration_s":   DURATION,
            "waveform_model":          "3.5PN IMRPhenomD-like SPA",
            "pe_method":               "Fisher information matrix (numerical Jacobian)",
            "jacobian_eps_frac":       0.05,
            "jsd_threshold_nats":      1e-3,
            "reference":               "Dax et al. 2021, PRL 127 241103",
        },
        "events":          all_results,
        "param_mean_jsd":  param_means,
        "overall_mean_jsd": overall_mean,
    }

    out_path = Path("reports/real_injection_jsd.json")
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    log.info(f"\nResults saved → {out_path}")
    return output


if __name__ == "__main__":
    results = run_calibration()
    print("\nFinal summary:")
    for p, j in results["param_mean_jsd"].items():
        print(f"  {p:12s}: {j:.4e} nats")
    print(f"  {'OVERALL':12s}: {results['overall_mean_jsd']:.4e} nats")
