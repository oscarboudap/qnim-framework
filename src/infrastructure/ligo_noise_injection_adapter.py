"""Inject beyond-GR waveforms into disjoint off-source detector noise segments."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence

import numpy as np
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import MinMaxScaler

from src.infrastructure.ligo_pca_adapter import GWDataset


@dataclass(frozen=True)
class NoiseSegment:
    segment_id: str
    data: np.ndarray
    sample_rate: int
    detector: str
    gps_start: Optional[float] = None


def _matched_filter_scores(
    queries: np.ndarray,
    templates: np.ndarray,
    segment: NoiseSegment,
    sample_rate: int,
    f_low: float,
    f_high: float,
) -> np.ndarray:
    """Score centered whitened queries against a per-class template bank."""
    from scipy.signal import hilbert

    queries = np.asarray(queries, dtype=float)
    templates = np.asarray(templates, dtype=float)
    if queries.ndim != 2 or templates.ndim != 3:
        raise ValueError("queries debe ser 2D y templates debe ser 3D")
    if queries.shape[1] != templates.shape[2] or len(segment.data) < queries.shape[1]:
        raise ValueError("queries y templates deben alinearse con el segmento")
    if segment.sample_rate != sample_rate:
        raise ValueError("El segmento y la configuración deben usar el mismo sample_rate")
    if not np.isfinite(queries).all() or not np.isfinite(templates).all():
        raise ValueError("queries y templates deben ser finitos")

    noise = np.asarray(segment.data[:queries.shape[1]], dtype=float)
    frequencies, psd = welch(
        noise,
        fs=sample_rate,
        nperseg=min(len(noise), max(256, sample_rate)),
        detrend="constant",
        scaling="density",
    )
    bins = np.fft.rfftfreq(queries.shape[1], d=1.0 / sample_rate)
    psd_bins = np.interp(bins, frequencies, psd)
    usable = (bins >= f_low) & (bins <= f_high) & (psd_bins > 0)
    if not np.any(usable):
        raise ValueError(f"El segmento {segment.segment_id} no tiene bins PSD utilizables")

    scale = np.sqrt(psd_bins[usable] * (sample_rate / queries.shape[1]))
    query_fft = np.fft.rfft(queries, axis=-1)
    query_white_fft = np.zeros_like(query_fft)
    query_white_fft[..., usable] = query_fft[..., usable] / scale
    queries_white = np.fft.irfft(
        query_white_fft, n=queries.shape[1], axis=-1
    )

    template_fft = np.fft.rfft(templates, axis=-1)
    quadrature = hilbert(templates, axis=-1).imag
    quadrature_fft = np.fft.rfft(quadrature, axis=-1)
    real_white_fft = np.zeros_like(template_fft)
    quad_white_fft = np.zeros_like(quadrature_fft)
    real_white_fft[..., usable] = template_fft[..., usable] / scale
    quad_white_fft[..., usable] = quadrature_fft[..., usable] / scale
    real_white = np.fft.irfft(real_white_fft, n=queries.shape[1], axis=-1)
    quad_white = np.fft.irfft(quad_white_fft, n=queries.shape[1], axis=-1)
    template_norm = np.sqrt(
        np.sum(real_white**2 + quad_white**2, axis=-1)
    )
    if np.any(template_norm <= 0) or not np.isfinite(template_norm).all():
        raise ValueError("El banco contiene plantillas con norma no válida")

    real_overlap = np.einsum(
        "in,ctn->ict", queries_white, real_white, optimize=True
    )
    quad_overlap = np.einsum(
        "in,ctn->ict", queries_white, quad_white, optimize=True
    )
    scores = np.sqrt(real_overlap**2 + quad_overlap**2) / template_norm[None, :, :]
    return scores.max(axis=-1)


class LIGONoiseInjectionAdapter:
    """Build a leakage-resistant dataset from supplied off-source noise segments.

    Supply independent segment IDs for train, val, and test. Network fetching is
    explicit through ``fetch_gwosc_segments``; construction never contacts GWOSC.
    """

    THEORY_CLASSES = (
        "GR", "standard-siren", "qnm-21", "qnm-33", "pn-deformation",
        "extra-dimensions", "scalar-tensor", "graviton-mass", "chern-simons",
        "liv-alpha1.5", "liv-alpha4", "loop-quantum-gravity", "gup",
    )

    def __init__(
        self,
        noise_segments: dict[str, Sequence[NoiseSegment]],
        sample_rate: int = 4096,
        duration: float = 4.0,
        n_components: int = 32,
        n_pca_fit: int = 32,
        target_snr_range: tuple[float, float] = (8.0, 30.0),
        approximant: str = "IMRPhenomD",
        max_classes: Optional[int] = None,
        waveform_generator: Optional[Callable] = None,
        f_low: float = 20.0,
        f_high: float = 1024.0,
        snr_normalization: Literal["reference_gr", "per_waveform"] = "reference_gr",
        matched_filter_bank_size: int = 0,
    ):
        if sample_rate < 1 or duration <= 0:
            raise ValueError("sample_rate y duration deben ser positivos")
        if n_components < 1 or n_pca_fit < n_components:
            raise ValueError("Se requiere 1 <= n_components <= n_pca_fit")
        if not 0 < target_snr_range[0] <= target_snr_range[1]:
            raise ValueError("target_snr_range debe ser positivo y ordenado")
        if not 0 < f_low < f_high <= sample_rate / 2:
            raise ValueError("Se requiere 0 < f_low < f_high <= Nyquist")
        if snr_normalization not in ("reference_gr", "per_waveform"):
            raise ValueError("snr_normalization debe ser reference_gr o per_waveform")
        if matched_filter_bank_size < 0:
            raise ValueError("matched_filter_bank_size no puede ser negativo")

        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(round(sample_rate * duration))
        self.n_components = n_components
        self.n_pca_fit = n_pca_fit
        self.target_snr_range = target_snr_range
        self.approximant = approximant
        self.f_low = f_low
        self.f_high = f_high
        self.snr_normalization = snr_normalization
        self.matched_filter_bank_size = matched_filter_bank_size
        self.waveform_generator = waveform_generator
        self.n_classes = min(max_classes, len(self.THEORY_CLASSES)) if max_classes else len(self.THEORY_CLASSES)
        self.theory_names = list(self.THEORY_CLASSES[:self.n_classes])
        self.noise_segments = self._validate_noise_segments(noise_segments)

    def _validate_noise_segments(self, splits):
        if set(splits) != {"train", "val", "test"}:
            raise ValueError("noise_segments debe contener exactamente train, val y test")
        seen_ids = set()
        gps_intervals = []
        validated = {}
        for split, segments in splits.items():
            segments = tuple(segments)
            if not segments:
                raise ValueError(f"El split {split} necesita al menos un segmento de ruido")
            for segment in segments:
                if segment.segment_id in seen_ids:
                    raise ValueError("Los IDs de segmentos deben ser disjuntos entre splits")
                seen_ids.add(segment.segment_id)
                if segment.gps_start is not None:
                    start = float(segment.gps_start)
                    end = start + self.duration
                    for detector, previous_start, previous_end, previous_split in gps_intervals:
                        if (
                            detector == segment.detector
                            and start < previous_end
                            and previous_start < end
                        ):
                            raise ValueError("Los intervalos GPS deben ser disjuntos")
                    gps_intervals.append((segment.detector, start, end, split))
                data = np.asarray(segment.data, dtype=float)
                if segment.sample_rate != self.sample_rate:
                    raise ValueError("Todos los segmentos deben usar el sample_rate configurado")
                if data.ndim != 1 or data.size < self.n_samples or not np.isfinite(data).all():
                    raise ValueError(f"El segmento {segment.segment_id} no es una serie finita utilizable")
            validated[split] = segments
        return validated

    @staticmethod
    def fetch_gwosc_segments(
        detector: str,
        gps_ranges: dict[str, Sequence[tuple[float, float]] | tuple[float, float]],
        sample_rate: int = 4096,
        segment_duration: float = 4.0,
    ) -> dict[str, list[NoiseSegment]]:
        """Fetch caller-selected O3b strain ranges through the OSDF datafind backend."""
        if sample_rate != 4096:
            raise ValueError("El acceso OSDF configurado soporta O3b_4KHZ_R1 (4096 Hz)")
        from gwpy.timeseries import TimeSeries

        n_samples = int(round(sample_rate * segment_duration))
        result = {"train": [], "val": [], "test": []}
        if set(gps_ranges) != set(result):
            raise ValueError("gps_ranges debe definir rangos independientes para train, val y test")

        for split, configured_ranges in gps_ranges.items():
            if (
                isinstance(configured_ranges, (tuple, list))
                and len(configured_ranges) == 2
                and all(np.isscalar(value) for value in configured_ranges)
            ):
                intervals = [configured_ranges]
            else:
                intervals = list(configured_ranges)
            for gps_start, gps_end in intervals:
                if gps_end <= gps_start:
                    raise ValueError(f"Rango GPS inválido para {split}")
                series = TimeSeries.get(
                    f"{detector}:GWOSC-4KHZ_R1_STRAIN",
                    gps_start,
                    gps_end,
                    source="gwdatafind",
                    frametype=f"{detector}_GWOSC_O3b_4KHZ_R1",
                    host="datafind.gwosc.org",
                    urltype="osdf",
                    verbose=False,
                )
                data = np.asarray(series.value, dtype=float)
                count = len(data) // n_samples
                for index in range(count):
                    start_index = index * n_samples
                    segment_start = gps_start + index * segment_duration
                    result[split].append(NoiseSegment(
                        segment_id=f"{detector}:{segment_start:.3f}",
                        data=data[start_index:start_index + n_samples],
                        sample_rate=sample_rate,
                        detector=detector,
                        gps_start=segment_start,
                    ))
            if not result[split]:
                raise ValueError(f"GWOSC no devolvió un segmento completo para {split}")
        return result

    def _make_waveform(
        self,
        theory_class: int,
        rng: np.random.Generator,
        paired_base_waveform: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict, np.ndarray]:
        if self.waveform_generator is not None:
            waveform, metadata = self.waveform_generator(theory_class, rng)
        else:
            from src.infrastructure.physics_sstg_adapter import PhysicsSSTGAdapter

            generator = PhysicsSSTGAdapter(
                n_features=12,
                waveform_duration=self.duration,
                sample_rate=self.sample_rate,
                approximant=self.approximant,
                max_classes=self.n_classes,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                waveform, _, _, metadata = generator._generate_one(
                    theory_class,
                    rng,
                    self.target_snr_range,
                    base_waveform=paired_base_waveform,
                )
            if any("PyCBC falló" in str(item.message) for item in caught):
                raise RuntimeError("PyCBC falló; no se permite usar waveform fallback en este adaptador")

        waveform = np.asarray(waveform, dtype=float)
        if waveform.shape != (self.n_samples,) or not np.isfinite(waveform).all():
            raise ValueError("El waveform debe ser finito y tener la longitud configurada")
        metadata = dict(metadata)
        reference_waveform = np.asarray(
            metadata.pop("_normalization_reference_waveform", waveform), dtype=float
        )
        if reference_waveform.shape != waveform.shape or not np.isfinite(reference_waveform).all():
            raise ValueError("El waveform GR de referencia debe ser finito y tener la longitud configurada")
        return waveform, metadata, reference_waveform

    def _build_matched_filter_bank(
        self,
        train_event_metadata: list[dict],
        seed: int,
    ) -> tuple[np.ndarray, list[int]]:
        train_pairs = train_event_metadata[::self.n_classes]
        bank_size = min(self.matched_filter_bank_size, len(train_pairs))
        if bank_size < 1:
            raise ValueError("El banco requiere al menos una pareja de train")

        chirp_masses = np.asarray(
            [item.get("Mc", np.nan) for item in train_pairs], dtype=float
        )
        if np.isfinite(chirp_masses).all():
            order = np.argsort(chirp_masses, kind="stable")
        else:
            order = np.arange(len(train_pairs))
        positions = np.rint(
            np.linspace(0, len(order) - 1, num=bank_size)
        ).astype(int)
        selected_pairs = order[positions].tolist()
        templates = np.empty(
            (self.n_classes, bank_size, self.n_samples), dtype=np.float64
        )

        split_id = 0
        for bank_index, event_index in enumerate(selected_pairs):
            pair_seed = np.random.SeedSequence([seed, split_id, event_index])
            base_waveform = None
            for theory_class in range(self.n_classes):
                waveform_rng = np.random.default_rng(pair_seed)
                waveform, _, _ = self._make_waveform(
                    theory_class, waveform_rng, base_waveform
                )
                if theory_class == 0:
                    base_waveform = waveform
                templates[theory_class, bank_index] = waveform
        return templates, selected_pairs

    def _inject_and_whiten(
        self,
        waveform: np.ndarray,
        segment: NoiseSegment,
        target_snr: float,
        reference_waveform: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        noise = np.asarray(segment.data[:self.n_samples], dtype=float)
        frequencies, psd = welch(
            noise,
            fs=self.sample_rate,
            nperseg=min(self.n_samples, max(256, self.sample_rate)),
            detrend="constant",
            scaling="density",
        )
        bins = np.fft.rfftfreq(self.n_samples, d=1.0 / self.sample_rate)
        psd_bins = np.interp(bins, frequencies, psd)
        usable = (bins >= self.f_low) & (bins <= self.f_high) & (psd_bins > 0)
        if not np.any(usable):
            raise ValueError(f"El segmento {segment.segment_id} no tiene bins PSD utilizables")

        waveform_fft = np.fft.rfft(waveform)
        reference_fft = np.fft.rfft(
            waveform if reference_waveform is None else reference_waveform
        )
        delta_f = self.sample_rate / self.n_samples
        reference_snr = np.sqrt(
            4.0 * delta_f * np.sum(np.abs(reference_fft[usable]) ** 2 / psd_bins[usable])
        )
        modified_snr = np.sqrt(
            4.0 * delta_f * np.sum(np.abs(waveform_fft[usable]) ** 2 / psd_bins[usable])
        )
        if not np.isfinite(reference_snr) or reference_snr <= 0:
            raise ValueError("No se pudo calcular un SNR válido para la señal")

        normalization_snr = (
            modified_snr if self.snr_normalization == "per_waveform" else reference_snr
        )
        if not np.isfinite(normalization_snr) or normalization_snr <= 0:
            raise ValueError("No se pudo calcular el SNR usado para normalizar la señal")

        scale = target_snr / normalization_snr
        reference_scale = target_snr / reference_snr
        scaled_waveform = waveform * scale
        observed_snr = float(modified_snr * scale)
        injected = noise + scaled_waveform
        injected_fft = np.fft.rfft(injected)
        whitened_fft = np.zeros_like(injected_fft)
        whitened_fft[usable] = injected_fft[usable] / np.sqrt(psd_bins[usable] * delta_f)
        whitened = np.fft.irfft(whitened_fft, n=self.n_samples)
        reference_white_fft = np.zeros_like(reference_fft)
        reference_white_fft[usable] = (
            reference_fft[usable] * reference_scale / np.sqrt(psd_bins[usable] * delta_f)
        )
        reference_whitened = np.fft.irfft(reference_white_fft, n=self.n_samples)
        return whitened, observed_snr, reference_whitened

    def generate_balanced_dataset(
        self,
        n_per_class: int = 500,
        n_val_per_class: int = 50,
        n_test_per_class: int = 100,
        seed: int = 42,
        verbose: bool = True,
    ) -> GWDataset:
        if min(n_per_class, n_val_per_class, n_test_per_class) < 1:
            raise ValueError("Cada split debe tener al menos una muestra por clase")
        rng = np.random.default_rng(seed)
        raw_by_split = {"train": [], "val": [], "test": []}
        reference_by_split = {"train": [], "val": [], "test": []}
        labels_by_split = {"train": [], "val": [], "test": []}
        snr_by_split = {"train": [], "val": [], "test": []}
        event_metadata = {"train": [], "val": [], "test": []}
        counts = {"train": n_per_class, "val": n_val_per_class, "test": n_test_per_class}
        split_ids = {"train": 0, "val": 1, "test": 2}
        used_segments = {"train": [], "val": [], "test": []}

        for split, count in counts.items():
            segments = self.noise_segments[split]
            required_segments = count
            if len(segments) < required_segments:
                raise ValueError(
                    f"El split {split} requiere {required_segments} segmentos GPS únicos para pares; "
                    f"solo hay {len(segments)}"
                )
            segment_indices = rng.choice(
                len(segments), size=required_segments, replace=False
            )
            for event_index in range(count):
                segment = segments[int(segment_indices[event_index])]
                used_segments[split].append(segment.segment_id)
                target_snr = float(rng.uniform(*self.target_snr_range))
                pair_id = f"{split}-{event_index:06d}"
                pair_seed = np.random.SeedSequence([seed, split_ids[split], event_index])
                paired_base_waveform = None
                for theory_class in range(self.n_classes):
                    waveform_rng = np.random.default_rng(pair_seed)
                    waveform, metadata, reference_waveform = self._make_waveform(
                        theory_class, waveform_rng, paired_base_waveform
                    )
                    if theory_class == 0:
                        paired_base_waveform = waveform
                    injected, actual_snr, reference_whitened = self._inject_and_whiten(
                        waveform, segment, target_snr, reference_waveform
                    )
                    reference_by_split[split].append(reference_whitened)
                    raw_by_split[split].append(injected)
                    labels_by_split[split].append(theory_class)
                    snr_by_split[split].append(actual_snr)
                    event_metadata[split].append({
                        **metadata,
                        "pair_id": pair_id,
                        "noise_segment_id": segment.segment_id,
                        "detector": segment.detector,
                        "gps_start": segment.gps_start,
                        "base_target_snr": target_snr,
                        "observed_snr": actual_snr,
                    })

        X_train_raw = np.stack(raw_by_split["train"])
        X_val_raw = np.stack(raw_by_split["val"])
        X_test_raw = np.stack(raw_by_split["test"])
        y_train = np.asarray(labels_by_split["train"], dtype=int)
        y_val = np.asarray(labels_by_split["val"], dtype=int)
        y_test = np.asarray(labels_by_split["test"], dtype=int)

        n_pca = min(self.n_pca_fit, len(X_train_raw) - 1, X_train_raw.shape[1])
        if n_pca < self.n_components:
            raise ValueError("El conjunto de entrenamiento no permite ajustar n_pca_fit componentes")
        pca = PCA(n_components=n_pca, svd_solver="randomized", random_state=seed)
        pca.fit(X_train_raw)
        X_train_pca = pca.transform(X_train_raw)[:, :self.n_components]
        X_val_pca = pca.transform(X_val_raw)[:, :self.n_components]
        X_test_pca = pca.transform(X_test_raw)[:, :self.n_components]
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        X_train = np.clip(scaler.fit_transform(X_train_pca), -1.0, 1.0)
        X_val = np.clip(scaler.transform(X_val_pca), -1.0, 1.0)
        X_test = np.clip(scaler.transform(X_test_pca), -1.0, 1.0)

        X_matched_filter_val = None
        matched_filter_metadata = None
        if self.matched_filter_bank_size:
            template_bank, selected_pairs = self._build_matched_filter_bank(
                event_metadata["train"], seed
            )
            segments_by_id = {
                segment.segment_id: segment for segment in self.noise_segments["val"]
            }
            matched_filter_scores = []
            for pair_index in range(len(X_val_raw) // self.n_classes):
                start = pair_index * self.n_classes
                pair_metadata = event_metadata["val"][start]
                segment = segments_by_id[pair_metadata["noise_segment_id"]]
                matched_filter_scores.append(_matched_filter_scores(
                    X_val_raw[start:start + self.n_classes],
                    template_bank,
                    segment,
                    self.sample_rate,
                    self.f_low,
                    self.f_high,
                ))
            X_matched_filter_val = np.concatenate(matched_filter_scores, axis=0)
            matched_filter_metadata = {
                "method": "maximum_phase_marginalized_matched_filter",
                "bank_size_per_class": int(template_bank.shape[1]),
                "template_source_split": "train_only",
                "template_pair_indices": selected_pairs,
                "template_selection": "uniform_quantiles_of_train_chirp_mass",
                "validation_time_alignment": "centered_windows_no_time_shift_search",
                "test_evaluated": False,
            }

        from src.infrastructure.dephasing_features import MatchedFilterFeatureExtractor

        gr_references = np.stack(reference_by_split["train"])[::self.n_classes]
        physical_extractor = MatchedFilterFeatureExtractor(
            n_bands=16, f_low=self.f_low, f_high=self.f_high
        ).fit(gr_references, dt=1.0 / self.sample_rate)
        X_physical_train = physical_extractor.transform(
            X_train_raw,
            dt=1.0 / self.sample_rate,
            templates=np.stack(reference_by_split["train"]),
        )
        X_physical_val = physical_extractor.transform(
            X_val_raw,
            dt=1.0 / self.sample_rate,
            templates=np.stack(reference_by_split["val"]),
        )
        X_physical_test = physical_extractor.transform(
            X_test_raw,
            dt=1.0 / self.sample_rate,
            templates=np.stack(reference_by_split["test"]),
        )
        if len(gr_references) > 1:
            gr_spectra = np.fft.rfft(gr_references, axis=1)
            train_templates = np.fft.irfft(
                (gr_spectra.sum(axis=0, keepdims=True) - gr_spectra)
                / (len(gr_references) - 1),
                n=self.n_samples,
                axis=1,
            )
            train_templates = np.repeat(train_templates, self.n_classes, axis=0)
        else:
            train_templates = None
        X_global_template_train = physical_extractor.transform(
            X_train_raw,
            dt=1.0 / self.sample_rate,
            templates=train_templates,
        )
        X_global_template_val = physical_extractor.transform(
            X_val_raw,
            dt=1.0 / self.sample_rate,
        )
        X_global_template_test = physical_extractor.transform(
            X_test_raw,
            dt=1.0 / self.sample_rate,
        )

        fisher_scores, fisher_pvalues = f_classif(pca.transform(X_train_raw), y_train)
        variance = pca.explained_variance_ratio_
        metadata = {
            "seed": seed,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "n_components": self.n_components,
            "n_pca_fit": n_pca,
            "n_per_class": n_per_class,
            "n_val_per_class": n_val_per_class,
            "n_test_per_class": n_test_per_class,
            "approximant": self.approximant,
            "noise_segments": {
                split: used_segments[split]
                for split in used_segments
            },
            "paired_injections": {
                "shared_source_parameters": True,
                "shared_noise_within_pair": True,
                "shared_base_target_snr": True,
                "normalization_reference": (
                    "unmodified_GR_waveform"
                    if self.snr_normalization == "reference_gr"
                    else "each_theory_waveform"
                ),
                "snr_normalization": self.snr_normalization,
                "waveform_generation": (
                    "one_GR_waveform_reused_per_pair"
                    if self.waveform_generator is None else "custom_generator"
                ),
            },
            "physical_features": {
                "method": "complex_band_overlap_against_paired_GR_reference",
                "n_bands": physical_extractor.n_bands,
                "feature_names": physical_extractor.get_feature_names(),
                "template_fit_split": "train_GR_only_for_global_fallback",
                "paired_reference_is_oracle": True,
                "retains_amplitude_and_phase": True,
            },
            "global_template_features": {
                "method": "complex_band_overlap_against_train_GR_mean_template",
                "n_bands": physical_extractor.n_bands,
                "feature_names": physical_extractor.get_feature_names(),
                "template_fit_split": "train_GR_only_leave_pair_out_for_train",
                "train_template_cross_fitted_by_pair": len(gr_references) > 1,
                "paired_reference_is_oracle": False,
                "retains_amplitude_and_phase": True,
            },
            "matched_filter_bank": matched_filter_metadata,
            "pca_analysis": {
                "explained_variance": variance.tolist(),
                "cumulative_variance": variance.cumsum().tolist(),
                "class_fisher_scores": [float(x) if np.isfinite(x) else None for x in fisher_scores],
                "class_fisher_pvalues": [float(x) if np.isfinite(x) else None for x in fisher_pvalues],
            },
            "event_metadata": event_metadata,
        }
        if verbose:
            print(
                f"Noise-injected dataset: train={X_train.shape}, val={X_val.shape}, "
                f"test={X_test.shape}, detector segments are split-disjoint"
            )

        return GWDataset(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            n_classes=self.n_classes,
            theory_names=self.theory_names,
            pca=pca,
            scaler=scaler,
            snr_train=np.asarray(snr_by_split["train"]),
            snr_val=np.asarray(snr_by_split["val"]),
            metadata=metadata,
            X_physical_train=X_physical_train,
            X_physical_val=X_physical_val,
            X_physical_test=X_physical_test,
            X_global_template_train=X_global_template_train,
            X_global_template_val=X_global_template_val,
            X_global_template_test=X_global_template_test,
            X_matched_filter_val=X_matched_filter_val,
        )
