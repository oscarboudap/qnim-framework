"""Fetch disjoint GWOSC off-source ranges and prepare an injected dataset."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare train/validation/test data from GWOSC off-source noise"
    )
    parser.add_argument("--gps-ranges", required=True, help="JSON with distinct train, val, test GPS ranges")
    parser.add_argument("--detector", required=True, choices=["H1", "L1", "V1"])
    parser.add_argument("--output-dir", default="results/scaling_study/dataset")
    parser.add_argument("--sample-rate", type=int, default=4096)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--n-per-class", type=int, default=500)
    parser.add_argument("--n-val-per-class", type=int, default=50)
    parser.add_argument("--n-test-per-class", type=int, default=100)
    parser.add_argument("--n-components", type=int, default=32)
    parser.add_argument("--n-pca-fit", type=int, default=32)
    parser.add_argument("--snr-min", type=float, default=8.0)
    parser.add_argument("--snr-max", type=float, default=30.0)
    parser.add_argument("--approximant", default="IMRPhenomD")
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    import numpy as np

    from src.infrastructure.ligo_noise_injection_adapter import LIGONoiseInjectionAdapter

    gps_ranges_raw = json.loads(Path(args.gps_ranges).read_text(encoding="utf-8"))
    gps_ranges = {split: tuple(bounds) for split, bounds in gps_ranges_raw.items()}
    segments = LIGONoiseInjectionAdapter.fetch_gwosc_segments(
        detector=args.detector,
        gps_ranges=gps_ranges,
        sample_rate=args.sample_rate,
        segment_duration=args.duration,
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
