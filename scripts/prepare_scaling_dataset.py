"""Generate and persist one shared 32-component LIGO-PCA dataset."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare one train/validation/test dataset for QNIM scaling experiments"
    )
    parser.add_argument("--output-dir", default="results/scaling_study/dataset")
    parser.add_argument("--n-per-class", type=int, default=500)
    parser.add_argument("--n-val-per-class", type=int, default=50)
    parser.add_argument("--n-test-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--approximant", default="IMRPhenomD")
    parser.add_argument("--max-classes", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    import numpy as np

    from src.infrastructure.ligo_pca_adapter import LIGOPCAAdapter

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = LIGOPCAAdapter(
        n_components=32,
        n_pca_fit=32,
        approximant=args.approximant,
        max_classes=args.max_classes,
    )
    dataset = adapter.generate_balanced_dataset(
        n_per_class=args.n_per_class,
        n_val_per_class=args.n_val_per_class,
        n_test_per_class=args.n_test_per_class,
        seed=args.seed,
    )

    for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
        np.save(output_dir / f"{name}.npy", getattr(dataset, name), allow_pickle=False)

    (output_dir / "labels.json").write_text(
        json.dumps({"theory_names": dataset.theory_names}, indent=2),
        encoding="utf-8",
    )
    metadata = {
        **dataset.metadata,
        "n_components": 32,
        "n_pca_fit": 32,
        "n_classes": dataset.n_classes,
        "theory_names": dataset.theory_names,
        "seed": args.seed,
        "approximant": args.approximant,
    }
    (output_dir / "pca_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Dataset guardado en {output_dir.resolve()}")
    print(
        f"train={dataset.X_train.shape}, val={dataset.X_val.shape}, "
        f"test={dataset.X_test.shape}"
    )


if __name__ == "__main__":
    main()