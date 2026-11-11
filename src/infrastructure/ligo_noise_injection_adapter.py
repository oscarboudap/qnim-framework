"""Inject beyond-GR waveforms into disjoint off-source detector noise segments."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

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
    ):
        if sample_rate < 1 or duration <= 0:
            raise ValueError("sample_rate y duration deben ser positivos")
        if n_components < 1 or n_pca_fit < n_components:
            raise ValueError("Se requiere 1 <= n_components <= n_pca_fit")
        if not 0 < target_snr_range[0] <= target_snr_range[1]:
            raise ValueError("target_snr_range debe ser positivo y ordenado")
        if not 0 < f_low < f_high <= sample_rate / 2:
            raise ValueError("Se requiere 0 < f_low < f_high <= Nyquist")

        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(round(sample_rate * duration))
        self.n_components = n_components
        self.n_pca_fit = n_pca_fit
        self.target_snr_range = target_snr_range
        self.approximant = approximant
        self.f_low = f_low
        self.f_high = f_high
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

    def _make_waveform(self, theory_class: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
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
                    theory_class, rng, self.target_snr_range
                )
            if any("PyCBC falló" in str(item.message) for item in caught):
                raise RuntimeError("PyCBC falló; no se permite usar waveform fallback en este adaptador")

        waveform = np.asarray(waveform, dtype=float)
        if waveform.shape != (self.n_samples,) or not np.isfinite(waveform).all():
            raise ValueError("El waveform debe ser finito y tener la longitud configurada")
        return waveform, dict(metadata)

    def _inject_and_whiten(
        self,
        waveform: np.ndarray,
        segment: NoiseSegment,
        target_snr: float,
    ) -> tuple[np.ndarray, float]:
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
        delta_f = self.sample_rate / self.n_samples
        optimal_snr = np.sqrt(
            4.0 * delta_f * np.sum(np.abs(waveform_fft[usable]) ** 2 / psd_bins[usable])
        )
        if not np.isfinite(optimal_snr) or optimal_snr <= 0:
            raise ValueError("No se pudo calcular un SNR válido para la señal")

        scaled_waveform = waveform * (target_snr / optimal_snr)
        injected = noise + scaled_waveform
        injected_fft = np.fft.rfft(injected)
        whitened_fft = np.zeros_like(injected_fft)
        whitened_fft[usable] = injected_fft[usable] / np.sqrt(psd_bins[usable] * delta_f)
        whitened = np.fft.irfft(whitened_fft, n=self.n_samples)
        return whitened, target_snr

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
        labels_by_split = {"train": [], "val": [], "test": []}
        snr_by_split = {"train": [], "val": [], "test": []}
        event_metadata = {"train": [], "val": [], "test": []}
        counts = {"train": n_per_class, "val": n_val_per_class, "test": n_test_per_class}

        for split, count in counts.items():
            segments = self.noise_segments[split]
            required_segments = count * self.n_classes
            if len(segments) < required_segments:
                raise ValueError(
                    f"El split {split} requiere {required_segments} segmentos GPS únicos; "
                    f"solo hay {len(segments)}"
                )
            segment_indices = rng.choice(
                len(segments), size=required_segments, replace=False
            )
            segment_cursor = 0
            for theory_class in range(self.n_classes):
                for _ in range(count):
                    waveform, metadata = self._make_waveform(theory_class, rng)
                    segment = segments[int(segment_indices[segment_cursor])]
                    segment_cursor += 1
                    target_snr = float(rng.uniform(*self.target_snr_range))
                    injected, actual_snr = self._inject_and_whiten(waveform, segment, target_snr)
                    raw_by_split[split].append(injected)
                    labels_by_split[split].append(theory_class)
                    snr_by_split[split].append(actual_snr)
                    event_metadata[split].append({
                        **metadata,
                        "noise_segment_id": segment.segment_id,
                        "detector": segment.detector,
                        "gps_start": segment.gps_start,
                        "target_snr": actual_snr,
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
        pca = PCA(n_components=n_pca, svd_solver="full", random_state=seed)
        pca.fit(X_train_raw)
        X_train_pca = pca.transform(X_train_raw)[:, :self.n_components]
        X_val_pca = pca.transform(X_val_raw)[:, :self.n_components]
        X_test_pca = pca.transform(X_test_raw)[:, :self.n_components]
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        X_train = np.clip(scaler.fit_transform(X_train_pca), -1.0, 1.0)
        X_val = np.clip(scaler.transform(X_val_pca), -1.0, 1.0)
        X_test = np.clip(scaler.transform(X_test_pca), -1.0, 1.0)

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
                split: [segment.segment_id for segment in segments]
                for split, segments in self.noise_segments.items()
            },
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
        )
