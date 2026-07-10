"""
physics_sstg_adapter.py  (v3 — pasa chirp mass al extractor de fase)
"""

from __future__ import annotations
import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.infrastructure.dephasing_features import DephasingFeatureExtractor
from src.infrastructure.stochastic_signal_generator import (
    _inject_standard_siren, _inject_qnm_subdominant,
    _inject_pn_deformation, _inject_scalar_tensor,
    _inject_graviton_mass, _inject_chern_simons,
    _inject_liv, _inject_gup,
)


@dataclass
class PhysicsGWDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val:   np.ndarray
    y_val:   np.ndarray
    n_classes: int
    feature_extractor: DephasingFeatureExtractor
    theory_names: List[str]
    snr_train: np.ndarray = field(default_factory=lambda: np.array([]))
    snr_val:   np.ndarray = field(default_factory=lambda: np.array([]))
    metadata: dict = field(default_factory=dict)


def _chirp_mass(m1: float, m2: float) -> float:
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def _generate_base_waveform(m1, m2, distance_mpc, duration=4.0,
                              sample_rate=4096, f_lower=20.0,
                              approximant="IMRPhenomD"):
    try:
        from pycbc.waveform import get_td_waveform
        hp, _ = get_td_waveform(
            approximant=approximant, mass1=m1, mass2=m2,
            distance=distance_mpc, delta_t=1.0/sample_rate, f_lower=f_lower,
        )
        strain = hp.numpy()
        n_target = int(duration * sample_rate)
        strain = strain[-n_target:] if len(strain) >= n_target else \
                 np.pad(strain, (n_target - len(strain), 0))
        return strain, 1.0 / sample_rate
    except Exception as exc:
        warnings.warn(f"PyCBC falló ({exc}); usando chirp sinusoidal de fallback.", stacklevel=3)
        return _fallback_chirp(m1, m2, distance_mpc, duration, sample_rate, f_lower)


def _fallback_chirp(m1, m2, distance_mpc, duration, sample_rate, f_lower):
    dt  = 1.0 / sample_rate
    t   = np.arange(0, duration, dt)
    tau = duration - t + 1e-3
    f_inst = np.clip(
        (1/np.pi) * (5/(256*tau))**(3/8) * ((m1+m2)*4.93e-6)**(-5/8),
        f_lower, sample_rate/2.0,
    )
    phase = 2*np.pi*np.cumsum(f_inst)*dt
    amp   = np.exp(-(t-duration)**2/0.01) / (distance_mpc + 1.0)
    return amp * np.sin(phase), dt


class PhysicsSSTGAdapter:
    """
    Adaptador con física real: PyCBC + dispatcher 13 teorías + features de fase.
    Pasa la chirp mass real de cada evento al extractor para usar plantillas
    por bin de Mc en vez de una plantilla global.
    """

    THEORY_CLASSES = [
        "GR", "standard-siren", "qnm-21", "qnm-33", "pn-deformation",
        "extra-dimensions", "scalar-tensor", "graviton-mass", "chern-simons",
        "liv-alpha1.5", "liv-alpha4", "loop-quantum-gravity", "gup",
    ]

    def __init__(self, n_features=12, waveform_duration=4.0, sample_rate=4096,
                 f_lower=20.0, approximant="IMRPhenomD",
                 m_range=(10.0, 80.0), d_range=(100.0, 1000.0),
                 max_classes=None):
        assert n_features % 2 == 0
        self.n_features  = n_features
        self.n_bands     = n_features // 2
        self.duration    = waveform_duration
        self.sample_rate = sample_rate
        self.f_lower     = f_lower
        self.approximant = approximant
        self.m_range     = m_range
        self.d_range     = d_range
        n_cls = len(self.THEORY_CLASSES)
        self.n_classes   = min(max_classes, n_cls) if max_classes else n_cls
        self.theory_names = self.THEORY_CLASSES[:self.n_classes]

    def _generate_one(self, theory_class, rng, snr_range):
        m1 = float(rng.uniform(*self.m_range))
        m2 = float(rng.uniform(self.m_range[0], m1))
        d  = float(rng.uniform(*self.d_range))
        M  = m1 + m2
        Mc = _chirp_mass(m1, m2)

        strain, dt = _generate_base_waveform(
            m1, m2, d, self.duration, self.sample_rate,
            self.f_lower, self.approximant,
        )
        htilde = np.fft.rfft(strain)
        freqs  = np.fft.rfftfreq(len(strain), d=dt)

        if theory_class == 0:
            pass
        elif theory_class == 1:
            htilde = _inject_standard_siren(htilde, freqs, H0_true=float(rng.uniform(60,80)))
        elif theory_class == 2:
            htilde = _inject_qnm_subdominant(htilde, freqs, mode="21", rng=rng)
        elif theory_class == 3:
            htilde = _inject_qnm_subdominant(htilde, freqs, mode="33", rng=rng)
        elif theory_class == 4:
            htilde = _inject_pn_deformation(htilde, freqs, M, delta_phi_hat=float(rng.uniform(0.03,0.10)))
        elif theory_class == 5:
            htilde = _inject_liv(htilde, freqs, d, alpha=3.0, A_alpha_m=float(rng.uniform(1e8,1e9)))
        elif theory_class == 6:
            htilde = _inject_scalar_tensor(htilde, freqs, M, omega_BD_eff=float(rng.uniform(40,80)))
        elif theory_class == 7:
            htilde = _inject_graviton_mass(htilde, freqs, d, mg_eV=1.27e-23*7e21)
        elif theory_class == 8:
            htilde = _inject_chern_simons(htilde, freqs, d, kappa_gpc=float(rng.uniform(0.5,1.2)))
        elif theory_class == 9:
            htilde = _inject_liv(htilde, freqs, d, alpha=1.5, A_alpha_m=float(rng.uniform(5e9,2e10)))
        elif theory_class == 10:
            htilde = _inject_liv(htilde, freqs, d, alpha=4.0, A_alpha_m=float(rng.uniform(5e9,2e10)))
        elif theory_class == 11:
            htilde = _inject_qnm_subdominant(htilde, freqs, mode="33", rng=rng)
            htilde *= 1.3
        elif theory_class == 12:
            htilde = _inject_gup(htilde, freqs, beta_gup=float(rng.uniform(5e-3,2e-2)))

        strain_mod = np.fft.irfft(htilde, n=len(strain))
        snr_approx = float(np.clip(
            np.sqrt(np.sum(strain_mod**2)/(self.duration/self.sample_rate)) *
            rng.uniform(*snr_range) / 10.0,
            snr_range[0], snr_range[1],
        ))
        meta = {"m1": m1, "m2": m2, "Mc": Mc, "distance_mpc": d,
                "theory_class": theory_class,
                "theory_name": self.THEORY_CLASSES[theory_class]}
        return strain_mod, snr_approx, Mc, meta

    def generate_balanced_dataset(self, n_per_class=80, n_val_per_class=20,
                                   target_snr_range=(8.0, 30.0),
                                   seed=42, verbose=True) -> PhysicsGWDataset:
        rng     = np.random.default_rng(seed)
        dt      = 1.0 / self.sample_rate
        n_total = n_per_class + n_val_per_class

        all_strains, all_labels, all_snrs, all_mc = [], [], [], []

        for cls in range(self.n_classes):
            if verbose:
                print(f"  Generando clase {cls}: {self.THEORY_CLASSES[cls]}")
            for _ in range(n_total):
                strain, snr, mc, _ = self._generate_one(cls, rng, target_snr_range)
                all_strains.append(strain)
                all_labels.append(cls)
                all_snrs.append(snr)
                all_mc.append(mc)

        all_strains = np.stack(all_strains)
        all_labels  = np.array(all_labels, dtype=int)
        all_snrs    = np.array(all_snrs)
        all_mc      = np.array(all_mc)

        # Extractor: fit sobre eventos GR, pasando su chirp mass real
        gr_idx = np.where(all_labels == 0)[0][:n_per_class]
        extractor = DephasingFeatureExtractor(
            n_bands=4,  # 3 features/banda (Re+Im+dIm) → 12 total
            f_low=self.f_lower,
            f_high=min(self.sample_rate / 2.0, 512.0),
            n_mc_bins=5,
        )
        extractor.fit(
            [all_strains[i] for i in gr_idx],
            dt=dt,
            mc_values=all_mc[gr_idx],   # <-- chirp mass real, no estimada
        )

        # Transform: pasar chirp mass de cada evento
        features = extractor.transform(all_strains, dt=dt, mc_values=all_mc)

        col_mean = features.mean(axis=0)
        col_std  = features.std(axis=0) + 1e-10
        features = (features - col_mean) / col_std

        X_train_p, y_train_p, snr_train_p = [], [], []
        X_val_p,   y_val_p,   snr_val_p   = [], [], []

        for cls in range(self.n_classes):
            idx       = np.where(all_labels == cls)[0]
            train_idx = idx[:n_per_class]
            val_idx   = idx[n_per_class:n_per_class + n_val_per_class]
            X_train_p.append(features[train_idx])
            y_train_p.append(all_labels[train_idx])
            snr_train_p.append(all_snrs[train_idx])
            X_val_p.append(features[val_idx])
            y_val_p.append(all_labels[val_idx])
            snr_val_p.append(all_snrs[val_idx])

        X_train = np.concatenate(X_train_p)
        y_train = np.concatenate(y_train_p)
        snr_train = np.concatenate(snr_train_p)
        X_val   = np.concatenate(X_val_p)
        y_val   = np.concatenate(y_val_p)
        snr_val = np.concatenate(snr_val_p)

        all_snrs_c = np.concatenate([snr_train, snr_val])
        all_mc_c   = np.concatenate([all_mc[np.where(all_labels==c)[0][:n_per_class]]
                                      for c in range(self.n_classes)])
        if verbose:
            print(f"  ✅ Dataset físico generado: {len(X_train)} train / {len(X_val)} val")
            print(f"     Features: {self.n_features} (fase diferencial, plantillas por Mc)")
            print(f"     Clases: {self.n_classes}")
            print(f"     Mc: {all_mc.mean():.1f} ± {all_mc.std():.1f} M☉")
            print(f"     SNR: {all_snrs_c.mean():.1f} ± {all_snrs_c.std():.1f}")

        return PhysicsGWDataset(
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            n_classes=self.n_classes,
            feature_extractor=extractor,
            theory_names=self.theory_names,
            snr_train=snr_train, snr_val=snr_val,
            metadata={"col_mean": col_mean, "col_std": col_std,
                      "n_per_class": n_per_class, "seed": seed,
                      "approximant": self.approximant},
        )