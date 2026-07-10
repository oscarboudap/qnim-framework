"""
dephasing_features.py  (v5 — features separadas de amplitud y fase por banda)
==============================================================================
MOTIVACIÓN (resumen del diagnóstico):
    Las 13 teorías beyond-GR tienen firmas en dimensiones distintas:
    - Amplitud pura:  standard-siren (+4%), chern-simons (-55%)
    - Fase pura:      pn-deformation, graviton-mass (ambas ~3.7 rad totales)
    - Mixto pequeño:  qnm-21/33, LIV, GUP, scalar-tensor, LQG, ADD

    El overlap complejo por banda mezcla amplitud y fase en un solo número,
    lo que hace indistinguibles teorías con la misma magnitud total pero
    distinta dependencia en frecuencia (ej. pn-deform f^{-7/3} vs
    graviton-mass f^{+1}).

SOLUCIÓN — 4 tipos de features por banda de frecuencia:
    1. Re(overlap)   — amplitud relativa por banda vs plantilla GR del mismo Mc
    2. Im(overlap)   — fase diferencial por banda
    3. d(Im)/d(band) — gradiente espectral de fase (distingue f^{-7/3} de f^{+1})
    4. |overlap|     — magnitud del overlap (robusto a rotaciones de fase)

    Con n_bands=4 → 4*4=16 features → se truncan a n_features=12 tomando
    las 3 primeras por banda (Re, Im, gradiente), dejando |overlap| como
    redundante en la práctica.

    Implementación simplificada para n_features=12 (n_bands=4, 3 features/banda):
        features = [Re_0, Im_0, dIm_0, Re_1, Im_1, dIm_1, ..., Re_3, Im_3, dIm_3]

    Chirp mass: pasada explícitamente desde el generador (no estimada del espectro).
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Sequence


class DephasingFeatureExtractor:
    """
    Extractor de features de amplitud + fase + gradiente de fase por banda,
    con plantillas múltiples por bin de chirp mass.

    Parámetros
    ----------
    n_bands : int       — bandas de frecuencia. 3·n_bands features. Default 4 → 12.
    f_low : float       — Hz. Default 20.
    f_high : float      — Hz. Default 512.
    n_mc_bins : int     — bins de chirp mass. Default 5.
    epsilon : float     — regularización.
    """

    def __init__(self, n_bands=4, f_low=20.0, f_high=512.0,
                 n_mc_bins=5, epsilon=1e-10):
        self.n_bands   = n_bands
        self.f_low     = f_low
        self.f_high    = f_high
        self.n_mc_bins = n_mc_bins
        self.epsilon   = epsilon

        self._mc_bin_edges:    Optional[np.ndarray] = None
        self._templates:       Optional[list]        = None
        self._template_freqs:  Optional[np.ndarray]  = None
        self._band_edges:      Optional[np.ndarray]  = None
        self._global_template: Optional[np.ndarray]  = None

    @property
    def n_features(self):
        return 3 * self.n_bands

    @staticmethod
    def _normalize(strain: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(strain)
        return strain / norm if norm > 1e-300 else np.zeros_like(strain)

    def fit(self, gr_strains: Sequence[np.ndarray],
            dt: float = 1.0 / 4096.0,
            mc_values: Optional[np.ndarray] = None) -> "DephasingFeatureExtractor":
        n_samples = len(gr_strains[0])
        freqs = np.fft.rfftfreq(n_samples, d=dt)
        self._template_freqs = freqs

        htildes = np.stack(
            [np.fft.rfft(self._normalize(s)) for s in gr_strains], axis=0
        )
        self._global_template = htildes.mean(axis=0)

        if mc_values is not None and len(mc_values) == len(gr_strains):
            mc_arr = np.asarray(mc_values, dtype=float)
            mc_min = max(mc_arr.min() * 0.9, 1.0)
            mc_max = mc_arr.max() * 1.1
            self._mc_bin_edges = np.logspace(
                np.log10(mc_min), np.log10(mc_max), self.n_mc_bins + 1
            )
            self._templates = []
            for k in range(self.n_mc_bins):
                lo, hi = self._mc_bin_edges[k], self._mc_bin_edges[k + 1]
                mask = (mc_arr >= lo) & (mc_arr < hi)
                self._templates.append(
                    htildes[mask].mean(axis=0) if mask.sum() >= 1
                    else self._global_template.copy()
                )
        else:
            self._mc_bin_edges = None
            self._templates    = None

        f_min = max(self.f_low,  freqs[1])
        f_max = min(self.f_high, freqs[-1])
        self._band_edges = np.logspace(
            np.log10(f_min), np.log10(f_max), self.n_bands + 1
        )
        return self

    def _get_template(self, mc_msun: Optional[float]) -> np.ndarray:
        if self._templates is None or mc_msun is None:
            return self._global_template
        idx = int(np.clip(
            np.searchsorted(self._mc_bin_edges, mc_msun, side='right') - 1,
            0, self.n_mc_bins - 1
        ))
        return self._templates[idx]

    def transform(self, strains: np.ndarray,
                  dt: float = 1.0 / 4096.0,
                  mc_values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extrae 3·n_bands features por evento:
            [Re_overlap_k, Im_overlap_k, dIm_overlap_k]  para k=0..n_bands-1

        donde dIm_k = Im_k - Im_{k-1} (gradiente de fase entre bandas).
        """
        if self._global_template is None:
            raise RuntimeError("Llama a fit() antes de transform().")

        squeeze = strains.ndim == 1
        if squeeze:
            strains   = strains[np.newaxis, :]
            mc_values = np.atleast_1d(mc_values) if mc_values is not None else None

        n_events, n_samples = strains.shape
        freqs    = np.fft.rfftfreq(n_samples, d=dt)
        features = np.zeros((n_events, 3 * self.n_bands))

        for i, strain in enumerate(strains):
            mc   = float(mc_values[i]) if mc_values is not None else None
            tmpl = self._get_template(mc)

            if len(freqs) != len(self._template_freqs):
                tmpl = (np.interp(freqs, self._template_freqs, np.real(tmpl)) +
                        1j * np.interp(freqs, self._template_freqs, np.imag(tmpl)))

            denom   = np.abs(tmpl) ** 2 + self.epsilon
            htilde  = np.fft.rfft(self._normalize(strain))
            overlap = htilde * np.conj(tmpl) / denom   # complejo por frecuencia

            im_vals = []
            for k in range(self.n_bands):
                mask    = (freqs >= self._band_edges[k]) & (freqs < self._band_edges[k + 1])
                mean_ov = overlap[mask].mean() if mask.any() else 0.0 + 0.0j
                re_k    = float(np.real(mean_ov))
                im_k    = float(np.imag(mean_ov))
                im_vals.append(im_k)
                # gradiente de fase respecto a la banda anterior
                d_im_k = im_k - im_vals[-2] if k > 0 else 0.0
                features[i, 3 * k]     = re_k
                features[i, 3 * k + 1] = im_k
                features[i, 3 * k + 2] = d_im_k

        return features[0] if squeeze else features

    def get_feature_names(self) -> List[str]:
        names = []
        for k in range(self.n_bands):
            lo = self._band_edges[k]   if self._band_edges is not None else k
            hi = self._band_edges[k+1] if self._band_edges is not None else k+1
            names += [f"Re_b{k}_{lo:.0f}-{hi:.0f}Hz",
                      f"Im_b{k}_{lo:.0f}-{hi:.0f}Hz",
                      f"dIm_b{k}_{lo:.0f}-{hi:.0f}Hz"]
        return names