"""Fetch disjoint GWOSC off-source ranges and prepare an injected dataset."""

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class O3bNoiseSegment:
    segment_id: str
    data: object
    sample_rate: int
    detector: str
    gps_start: float


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare train/validation/test data from GWOSC off-source noise"
    )
    parser.add_argument("--gps-ranges", required=True, help="JSON with distinct train, val, test GPS ranges")
    parser.add_argument("--detector", required=True, choices=["H1", "L1"])
    parser.add_argument(
        "--inventory",
        default=None,
        help="O3b HDF5 frame inventory JSON (defaults to the selected detector inventory)",
    )
    parser.add_argument("--frame-cache", default="data/o3b/frames")
    parser.add_argument("--output-dir", default="results/scaling_study/dataset_paired128")
    parser.add_argument("--sample-rate", type=int, default=4096)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--n-per-class", type=int, default=499)
    parser.add_argument("--n-val-per-class", type=int, default=49)
    parser.add_argument("--n-test-per-class", type=int, default=100)
    parser.add_argument("--n-components", type=int, default=128)
    parser.add_argument("--n-pca-fit", type=int, default=128)
    parser.add_argument("--snr-min", type=float, default=8.0)
    parser.add_argument("--snr-max", type=float, default=30.0)
    parser.add_argument("--approximant", default="IMRPhenomD")
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_o3b_segments(gps_ranges, detector, inventory_path, frame_cache, sample_rate, duration):
    import h5py
    import numpy as np

    inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    frames = [
        item for item in inventory["strain"]
        if item.get("detector") == detector and item.get("format") == "hdf5"
    ]
    frames.sort(key=lambda item: item["GPSstart"])
    if not frames:
        raise ValueError(f"El inventario no contiene frames HDF5 para {detector}")

    cache_dir = Path(frame_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame_data = {}
    frame_handles = []
    segments = {"train": [], "val": [], "test": []}
    samples_per_window = int(round(sample_rate * duration))

    for split, configured_ranges in gps_ranges.items():
        target_split = "val" if split == "validation" else split
        if target_split not in segments:
            raise ValueError(f"Split desconocido: {split}")
        if (
            isinstance(configured_ranges, (tuple, list))
            and len(configured_ranges) == 2
            and all(np.isscalar(value) for value in configured_ranges)
        ):
            intervals = [configured_ranges]
        else:
            intervals = list(configured_ranges)

        for gps_start, gps_end in intervals:
            window_count = round((gps_end - gps_start) / duration)
            if window_count < 1 or abs(gps_start + window_count * duration - gps_end) > 1e-6:
                raise ValueError(f"El intervalo {split} no se alinea a ventanas de {duration}s")

            for window_index in range(window_count):
                window_gps = float(gps_start + window_index * duration)
                frame = next((
                    item for item in frames
                    if item["GPSstart"] <= window_gps
                    and window_gps + duration <= item["GPSstart"] + item["duration"]
                ), None)
                if frame is None:
                    raise ValueError(f"No hay frame {detector} para GPS {window_gps}")

                frame_path = cache_dir / Path(frame["url"]).name
                if not frame_path.is_file():
                    temporary_path = frame_path.with_suffix(frame_path.suffix + ".part")
                    try:
                        print(f"Downloading {frame_path.name}", flush=True)
                        with urlopen(frame["url"], timeout=60) as response:
                            with temporary_path.open("wb") as output_file:
                                shutil.copyfileobj(response, output_file)
                        temporary_path.replace(frame_path)
                    except Exception:
                        temporary_path.unlink(missing_ok=True)
                        raise

                cache_key = str(frame_path)
                if cache_key not in frame_data:
                    hdf = h5py.File(frame_path, "r")
                    frame_handles.append(hdf)
                    strain_ds = hdf["strain/Strain"]
                    frame_gps = float(hdf["meta/GPSstart"][()])
                    spacing = float(strain_ds.attrs["Xspacing"])
                    frame_rate = 1.0 / spacing
                    if not np.isclose(frame_rate, sample_rate):
                        raise ValueError(f"Tasa inesperada en {frame_path}: {frame_rate}")
                    frame_data[cache_key] = (strain_ds, frame_gps)

                strain_ds, frame_gps = frame_data[cache_key]
                start_index_float = (window_gps - frame_gps) * sample_rate
                start_index = int(round(start_index_float))
                if not np.isclose(start_index_float, start_index, atol=1e-6):
                    raise ValueError(f"GPS {window_gps} no cae en una muestra de {sample_rate} Hz")
                window = strain_ds[start_index:start_index + samples_per_window]
                if len(window) != samples_per_window:
                    raise ValueError(f"Frame incompleto para GPS {window_gps}")
                segments[target_split].append(O3bNoiseSegment(
                    segment_id=f"{detector}:{window_gps:.3f}",
                    data=np.asarray(window, dtype=float),
                    sample_rate=sample_rate,
                    detector=detector,
                    gps_start=window_gps,
                ))

    for hdf in frame_handles:
        hdf.close()
    return segments


def main():
    args = parse_args()
    import numpy as np

    from src.infrastructure.ligo_noise_injection_adapter import LIGONoiseInjectionAdapter

    gps_ranges = json.loads(Path(args.gps_ranges).read_text(encoding="utf-8"))
    inventory_path = args.inventory or f"data/o3b/events/O3b_4KHZ_R1_{args.detector}.json"
    segments = load_o3b_segments(
        gps_ranges=gps_ranges,
        detector=args.detector,
        inventory_path=inventory_path,
        frame_cache=args.frame_cache,
        sample_rate=args.sample_rate,
        duration=args.duration,
    )
    adapter = LIGONoiseInjectionAdapter(
        noise_segments=segments,
        sample_rate=args.sample_rate,
        duration=args.duration,
        n_components=args.n_components,
        n_pca_fit=args.n_pca_fit,
        target_snr_range=(args.snr_min, args.snr_max),
        approximant=args.approximant,
        max_classes=args.max_classes,
    )
    dataset = adapter.generate_balanced_dataset(
        n_per_class=args.n_per_class,
        n_val_per_class=args.n_val_per_class,
        n_test_per_class=args.n_test_per_class,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
        np.save(output_dir / f"{name}.npy", getattr(dataset, name), allow_pickle=False)
    for split in ("train", "val", "test"):
        physical_features = getattr(dataset, f"X_physical_{split}")
        if physical_features is not None:
            np.save(
                output_dir / f"X_physical_{split}.npy",
                physical_features,
                allow_pickle=False,
            )
    (output_dir / "labels.json").write_text(
        json.dumps({"theory_names": dataset.theory_names}, indent=2), encoding="utf-8"
    )
    metadata = {
        **dataset.metadata,
        "detector": args.detector,
        "gps_ranges": gps_ranges,
        "snr_train": dataset.snr_train.tolist(),
        "snr_val": dataset.snr_val.tolist(),
    }
    (output_dir / "pca_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Dataset guardado en {output_dir.resolve()}")
    print(f"train={dataset.X_train.shape}, val={dataset.X_val.shape}, test={dataset.X_test.shape}")


if __name__ == "__main__":
    main()
