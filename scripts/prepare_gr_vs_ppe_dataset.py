"""Prepare a GR versus phenomenological ppE -1PN hypothesis dataset."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a binary GR versus ppE(-1PN) injection dataset"
    )
    parser.add_argument("--gps-ranges", required=True)
    parser.add_argument("--detector", required=True, choices=["H1", "L1"])
    parser.add_argument("--inventory", default=None)
    parser.add_argument("--frame-cache", default="data/o3b/frames")
    parser.add_argument(
        "--output-dir",
        default="results/scaling_study/dataset_gr_vs_ppe",
    )
    parser.add_argument("--sample-rate", type=int, default=4096)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--n-train-pairs", type=int, default=499)
    parser.add_argument("--n-val-pairs", type=int, default=49)
    parser.add_argument("--n-test-pairs", type=int, default=100)
    parser.add_argument("--n-components", type=int, default=32)
    parser.add_argument("--n-pca-fit", type=int, default=64)
    parser.add_argument("--snr-min", type=float, default=8.0)
    parser.add_argument("--snr-max", type=float, default=30.0)
    parser.add_argument("--approximant", default="IMRPhenomD")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    from scripts.prepare_noise_injection_dataset import load_o3b_segments
    from src.infrastructure.ligo_noise_injection_adapter import (
        LIGONoiseInjectionAdapter,
    )

    gps_ranges = json.loads(
        Path(args.gps_ranges).read_text(encoding="utf-8")
    )
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
    )
    dataset = adapter.generate_gr_vs_ppe_dataset(
        n_train_pairs=args.n_train_pairs,
        n_val_pairs=args.n_val_pairs,
        n_test_pairs=args.n_test_pairs,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in dataset.items():
        if name.startswith(("X_", "y_")):
            np.save(output_dir / f"{name}.npy", values, allow_pickle=False)
    metadata = {
        **dataset["metadata"],
        "detector": args.detector,
        "gps_ranges": gps_ranges,
        "sample_rate": args.sample_rate,
        "duration": args.duration,
        "approximant": args.approximant,
        "seed": args.seed,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "labels.json").write_text(
        json.dumps({"0": "GR", "1": "pn-deformation"}, indent=2),
        encoding="utf-8",
    )
    print(f"Dataset guardado en {output_dir.resolve()}")
    print(
        f"train={dataset['X_train'].shape}, val={dataset['X_val'].shape}, "
        f"test={dataset['X_test'].shape}"
    )


if __name__ == "__main__":
    main()
