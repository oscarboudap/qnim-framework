"""
gwosc_real_events.py
=====================
Descarga y procesa eventos reales de GWTC-3 desde GWOSC.

PIPELINE:
    1. Descarga strain HDF5 de 4s alrededor de la coalescencia (pocos MB/evento)
    2. Blanquea con PSD real de O3 (misma función que LIGOPCAAdapter)
    3. Proyecta con el PCA ajustado sobre datos sintéticos
    4. Clasifica con el VQC entrenado

USO:
    # Primero entrena el VQC con datos sintéticos:
    from src.infrastructure.ligo_pca_adapter import LIGOPCAAdapter
    from src.infrastructure.qiskit_vqc_trainer import QiskitVQCTrainer

    adapter = LIGOPCAAdapter()
    dataset = adapter.generate_balanced_dataset(n_per_class=200, n_val_per_class=50, seed=42)
    trainer = QiskitVQCTrainer(mode='sim', readout_hidden_size=32)
    result  = trainer.train_and_evaluate(dataset, n_qubits=12, shots=1024, max_iterations=80)

    # Luego clasifica eventos reales:
    from gwosc_real_events import GWOSCClassifier
    clf = GWOSCClassifier(trainer=trainer, adapter=adapter)
    results = clf.classify_gwtc3()

REQUISITOS:
    pip install gwpy requests h5py
    (gwpy ya incluye acceso a GWOSC)
"""

from __future__ import annotations
import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path

# Eventos GWTC-3 confirmados con SNR > 8
# Fuente: Abbott et al. 2023b (Phys. Rev. X, 13, 041039)
# Formato: (nombre, GPS_coalescencia, detector_primario)
GWTC3_EVENTS = [
    ("GW150914", 1126259462.4, "H1"),
    ("GW151012", 1128678900.4, "H1"),
    ("GW151226", 1135136350.6, "L1"),
    ("GW170104", 1167559936.6, "H1"),
    ("GW170608", 1180922494.5, "H1"),
    ("GW170729", 1185389807.3, "H1"),
    ("GW170809", 1186302519.7, "H1"),
    ("GW170814", 1186741861.5, "H1"),
    ("GW170818", 1187058327.1, "H1"),
    ("GW170823", 1187529256.5, "H1"),
    ("GW190408_181802", 1238782700.3, "H1"),
    ("GW190412",        1239082262.2, "H1"),
    ("GW190413_052954", 1239168612.5, "H1"),
    ("GW190413_134308", 1239198206.7, "H1"),
    ("GW190421_213856", 1239917954.3, "H1"),
    ("GW190424_180648", 1240164426.1, "L1"),
    ("GW190425",        1240215503.0, "L1"),
    ("GW190503_185404", 1240944862.3, "H1"),
    ("GW190512_180714", 1241719652.4, "H1"),
    ("GW190513_205428", 1241816086.8, "H1"),
    ("GW190514_065416", 1241852074.8, "H1"),
    ("GW190517_055101", 1242107479.8, "H1"),
    ("GW190519_153544", 1242315362.4, "H1"),
    ("GW190521",        1242442967.4, "H1"),
    ("GW190521_074359", 1242459857.5, "L1"),
    ("GW190527_092055", 1242984073.8, "H1"),
    ("GW190602_175927", 1243533585.1, "H1"),
    ("GW190620_030421", 1245035079.3, "L1"),
    ("GW190630_185205", 1245955943.2, "L1"),
    ("GW190701_203306", 1246048404.4, "H1"),
    ("GW190706_222641", 1246487219.3, "H1"),
    ("GW190707_093326", 1246527224.2, "H1"),
    ("GW190708_232457", 1246610715.4, "L1"),
    ("GW190719_215514", 1247608532.9, "H1"),
    ("GW190720_000836", 1247616534.7, "H1"),
    ("GW190727_060333", 1248242631.9, "H1"),
    ("GW190728_064510", 1248331528.2, "H1"),
    ("GW190731_140936", 1248617394.6, "H1"),
    ("GW190803_022701", 1248834439.7, "H1"),
    ("GW190814",        1249852257.0, "L1"),
    ("GW190828_063405", 1251009263.8, "H1"),
    ("GW190828_065509", 1251010527.9, "H1"),
    ("GW190910_112807", 1252150105.3, "L1"),
    ("GW190915_235702", 1252627040.7, "H1"),
    ("GW190924_021846", 1253326744.9, "L1"),
    ("GW190929_012149", 1253755327.5, "H1"),
    ("GW190930_133541", 1253885759.2, "L1"),
    ("GW191103_012549", 1256655967.5, "H1"),
    ("GW191105_143521", 1256843739.0, "H1"),
    ("GW191109_010717", 1257296855.3, "H1"),
    ("GW191113_071753", 1257679091.8, "L1"),
    ("GW191126_115259", 1258799597.0, "H1"),
    ("GW191127_050227", 1258872165.4, "L1"),
    ("GW191129_134029", 1259060447.5, "H1"),
    ("GW191204_110529", 1259503547.3, "H1"),
    ("GW191204_171526", 1259526944.2, "H1"),
    ("GW191215_223052", 1260490270.4, "H1"),
    ("GW191216_213338", 1260575636.8, "H1"),
    ("GW191222_033537", 1261020955.3, "H1"),
    ("GW191230_180458", 1261764316.6, "H1"),
    ("GW200105_162426", 1262276684.3, "L1"),
    ("GW200112_155838", 1262879936.5, "H1"),
    ("GW200115_042309", 1263069807.3, "L1"),
    ("GW200128_022011", 1264213229.3, "H1"),
    ("GW200129_065458", 1264316116.4, "H1"),
    ("GW200202_154313", 1264690011.8, "H1"),
    ("GW200208_130117", 1265202095.8, "H1"),
    ("GW200208_222617", 1265236995.4, "H1"),
    ("GW200209_085452", 1265293910.3, "H1"),
    ("GW200210_092254", 1265380992.3, "H1"),
    ("GW200216_220804", 1265932102.5, "H1"),
    ("GW200219_094415", 1266192273.9, "H1"),
    ("GW200220_061928", 1266219586.7, "L1"),
    ("GW200220_124850", 1266241748.4, "H1"),
    ("GW200224_222234", 1266618172.4, "H1"),
    ("GW200225_060421", 1266645879.4, "H1"),
    ("GW200302_015811", 1267187909.7, "H1"),
    ("GW200306_093714", 1267616252.2, "L1"),
    ("GW200308_173609", 1267801987.4, "H1"),
    ("GW200311_115853", 1268082751.2, "H1"),
    ("GW200316_215756", 1268616494.0, "H1"),
    ("GW200322_091133", 1269158111.6, "H1"),
]


@dataclass
class RealEventResult:
    name: str
    gps: float
    detector: str
    predicted_class: int
    predicted_theory: str
    class_probs: np.ndarray
    snr_approx: float
    features_pca: np.ndarray
    metadata: dict = field(default_factory=dict)


class GWOSCClassifier:
    """
    Descarga eventos reales de GWOSC y los clasifica con el VQC entrenado.

    Parámetros
    ----------
    trainer : QiskitVQCTrainer entrenado
    adapter : LIGOPCAAdapter con PCA ajustado sobre datos sintéticos
    cache_dir : directorio para cachear los HDF5 descargados
    sample_rate : Hz. Default 4096.
    segment_duration : segundos alrededor de la coalescencia. Default 4.0.
    """

    GWOSC_URL = "https://gwosc.org/eventapi/json/GWTC-3-confident"
    STRAIN_URL = "https://gwosc.org/eventapi/json/event/{name}"

    def __init__(self,
                 trainer,
                 adapter,
                 cache_dir: str = "data/gwosc_cache",
                 sample_rate: int = 4096,
                 segment_duration: float = 4.0):
        self.trainer          = trainer
        self.adapter          = adapter
        self.cache_dir        = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate      = sample_rate
        self.segment_duration = segment_duration

    def _download_strain(self, name: str, gps: float,
                          detector: str) -> Optional[np.ndarray]:
        """
        Descarga el strain de 4s alrededor de GPS usando gwpy.
        Cachea en HDF5 local para no repetir descargas.
        """
        cache_file = self.cache_dir / f"{name}_{detector}.npy"
        if cache_file.exists():
            return np.load(cache_file)

        try:
            from gwpy.timeseries import TimeSeries
            t_start = gps - self.segment_duration / 2 - 2
            t_end   = gps + self.segment_duration / 2 + 2
            ts = TimeSeries.fetch_open_data(
                detector, t_start, t_end,
                sample_rate=self.sample_rate,
                verbose=False,
            )
            # Extraer segmento central de segment_duration segundos
            n = int(self.segment_duration * self.sample_rate)
            center = len(ts) // 2
            half   = n // 2
            strain = ts.value[center - half: center + half]
            if len(strain) < n:
                strain = np.pad(strain, (0, n - len(strain)))
            strain = strain[:n]
            np.save(cache_file, strain)
            return strain
        except Exception as exc:
            warnings.warn(f"No se pudo descargar {name}/{detector}: {exc}")
            return None

    def _strain_to_features(self, strain: np.ndarray) -> np.ndarray:
        """
        Aplica el mismo pipeline que LIGOPCAAdapter:
            blanqueo PSD O3 → PCA (ya ajustado) → MinMaxScaler → clip [-1,1]
        """
        from src.infrastructure.ligo_pca_adapter import _whiten_strain
        dt = 1.0 / self.sample_rate

        # Blanqueo
        strain_white = _whiten_strain(
            strain, dt,
            f_low=self.adapter.f_low_white,
            f_high=self.adapter.f_high_white,
        )

        # PCA (transform, no fit — usa el PCA ya ajustado sobre sintéticos)
        pca_scores = self.adapter.pca.transform(strain_white[np.newaxis, :])
        pca_12     = pca_scores[:, :self.adapter.n_components]

        # Escalar con el scaler ya ajustado
        features = self.adapter.scaler.transform(pca_12)
        features = np.clip(features, -1.0, 1.0)
        return features[0]

    def _predict_one(self, features: np.ndarray) -> tuple:
        """Clasifica un evento usando el VQC entrenado."""
        from src.infrastructure.qiskit_vqc_trainer import (
            compute_chebyshev_stats, chebyshev_preprocess,
            _build_feature_map_and_ansatz, _bind_sample,
            _qubit_marginal_probs, _readout_mlp_probs,
            _split_readout_mlp, _n_readout_params,
        )
        from qiskit.primitives import StatevectorSampler

        n_qubits   = 12
        n_classes  = self.adapter.n_classes
        theta_full = self.trainer._theta  # pesos entrenados

        combined, x_params, ansatz_params = _build_feature_map_and_ansatz(
            n_qubits, reps=self.trainer.ansatz_reps
        )
        combined.measure_all()

        train_stats = compute_chebyshev_stats(
            self.adapter.scaler.inverse_transform(
                np.zeros((1, self.adapter.n_components))
            )
        )
        x_cheb = chebyshev_preprocess(features[np.newaxis], stats=train_stats)[0]

        n_ansatz = len(ansatz_params)
        theta_ansatz = theta_full[:n_ansatz]

        sampler = StatevectorSampler()
        bound   = _bind_sample(combined, x_params, ansatz_params,
                                x_cheb, theta_ansatz)
        counts  = sampler.run([(bound,)], shots=4096).result()[0].data.meas.get_counts()
        qprobs  = _qubit_marginal_probs(counts, n_qubits)

        W1, b1, W2, b2 = _split_readout_mlp(
            theta_full[n_ansatz:], n_qubits, n_classes,
            self.trainer.readout_hidden_size
        )
        probs = _readout_mlp_probs(qprobs, W1, b1, W2, b2)
        pred  = int(np.argmax(probs))
        return pred, probs

    def classify_event(self, name: str, gps: float,
                        detector: str) -> Optional[RealEventResult]:
        """Descarga y clasifica un evento individual."""
        strain = self._download_strain(name, gps, detector)
        if strain is None:
            return None

        features = self._strain_to_features(strain)

        # SNR aproximado
        snr = float(np.sqrt(np.sum(strain**2) /
                             (self.segment_duration / self.sample_rate)))

        pred_class, probs = self._predict_one(features)
        pred_theory = self.adapter.theory_names[pred_class]

        return RealEventResult(
            name=name, gps=gps, detector=detector,
            predicted_class=pred_class,
            predicted_theory=pred_theory,
            class_probs=probs,
            snr_approx=snr,
            features_pca=features,
        )

    def classify_gwtc3(self,
                        events: Optional[List] = None,
                        max_events: int = 90,
                        verbose: bool = True) -> List[RealEventResult]:
        """
        Clasifica todos los eventos de GWTC-3 (o un subconjunto).

        Parámetros
        ----------
        events : lista de (nombre, gps, detector). Si None, usa GWTC3_EVENTS.
        max_events : máximo de eventos a procesar.
        verbose : imprimir progreso.

        Devuelve
        --------
        Lista de RealEventResult con la clasificación de cada evento.
        """
        if events is None:
            events = GWTC3_EVENTS[:max_events]

        results = []
        gr_count = 0

        for i, (name, gps, det) in enumerate(events):
            if verbose:
                print(f"  [{i+1}/{len(events)}] {name} ({det})...", end=" ")

            result = self.classify_event(name, gps, det)
            if result is None:
                if verbose:
                    print("SKIP (descarga fallida)")
                continue

            results.append(result)
            theory = result.predicted_theory
            prob   = result.class_probs[result.predicted_class]
            is_gr  = result.predicted_class == 0

            if is_gr:
                gr_count += 1

            if verbose:
                print(f"{theory} (p={prob:.3f})")

        if verbose and results:
            print(f"\n  ✅ Procesados: {len(results)} eventos")
            print(f"     GR-consistentes: {gr_count}/{len(results)} "
                  f"({100*gr_count/len(results):.1f}%)")
            print(f"\n  Top anomalías (no-GR con mayor probabilidad):")
            non_gr = [(r.name, r.predicted_theory,
                       r.class_probs[r.predicted_class])
                      for r in results if r.predicted_class != 0]
            non_gr.sort(key=lambda x: -x[2])
            for name, theory, prob in non_gr[:5]:
                print(f"    {name}: {theory} (p={prob:.3f})")

        return results

    def summary_table(self, results: List[RealEventResult]) -> str:
        """Genera tabla resumen para el TFM."""
        lines = [
            f"{'Evento':<30} {'Teoría predicha':<20} {'p*':>6} {'GR?':>5}",
            "-" * 65,
        ]
        for r in sorted(results, key=lambda x: -x.class_probs[x.predicted_class]):
            is_gr = "✓" if r.predicted_class == 0 else "✗"
            p     = r.class_probs[r.predicted_class]
            lines.append(
                f"{r.name:<30} {r.predicted_theory:<20} {p:>6.3f} {is_gr:>5}"
            )
        return "\n".join(lines)