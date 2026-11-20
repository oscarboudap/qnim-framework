import numpy as np

from scripts.prepare_qnim_q1_6_dataset_v2_pca64 import (
    _advance_rng_without_waveform,
    _markdown_report,
    _numerical_rank,
)
from src.infrastructure import ligo_pca_adapter
from src.infrastructure.ligo_pca_adapter import LIGOPCAAdapter
from sklearn.decomposition import PCA


def test_discarded_rng_advance_matches_generator_without_waveforms(monkeypatch):
    def fake_base_waveform(
        m1,
        m2,
        distance_mpc,
        duration=4.0,
        sample_rate=4096,
        f_lower=20.0,
        approximant="IMRPhenomD",
        strict_physics=False,
    ):
        return np.zeros(int(duration * sample_rate)), 1.0 / sample_rate

    monkeypatch.setattr(
        ligo_pca_adapter, "_generate_base_waveform", fake_base_waveform
    )
    adapter = LIGOPCAAdapter(
        n_components=64,
        n_pca_fit=64,
        max_classes=13,
        strict_physics=True,
    )

    for theory_class in range(adapter.n_classes):
        source_rng = np.random.default_rng(42)
        advanced_rng = np.random.default_rng(42)
        adapter._generate_one(theory_class, source_rng)
        _advance_rng_without_waveform(
            theory_class, 1, adapter, advanced_rng
        )
        assert source_rng.bit_generator.state == advanced_rng.bit_generator.state


def test_numerical_rank_uses_full_centered_singular_values():
    features = np.array(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0], [2.0, 1.0, 3.0]]
    )
    pca = PCA(n_components=2, svd_solver="full").fit(features)

    singular_values, report = _numerical_rank(features, pca)

    assert singular_values.shape == (3,)
    assert report["centered_rank_upper_bound"] == 3
    assert report["rank_by_tolerance"]["relative_tolerance_1e-10"] == 2


def test_markdown_report_reads_recorded_runtime_versions():
    subspaces = {
        str(dimension): {
            "mean_of_all_angles_degrees": 0.1,
            "maximum_angle_degrees": 0.2,
        }
        for dimension in (16, 32, 64)
    }
    metadata = {
        "generation": {
            "class_count": 13,
            "train_per_class": 200,
            "validation_per_class": 50,
            "dataset_seed": 42,
            "train_examples": 2600,
            "validation_examples": 650,
        },
        "runtime_versions": {
            "python": "3.12.3",
            "numpy": "2.2.6",
            "scikit_learn": "1.7.2",
            "pycbc": "2.9.0",
        },
        "historical_prefix_check": {
            "labels_and_order_match": True,
            "tolerances": {"rtol": 1e-8, "atol": 1e-10},
            "signs": [1.0] * 16,
            "train": {"matches": True, "max_absolute_error": 0.0},
            "validation": {"matches": True, "max_absolute_error": 0.0},
        },
        "runtime": {"total_seconds": 1.0, "peak_rss_bytes": 1024},
    }
    variance = {
        "numerical_rank": {
            "rank_by_tolerance": {
                "numpy_default_matrix_rank": 2,
                "relative_tolerance_1e-12": 2,
                "relative_tolerance_1e-10": 2,
                "relative_tolerance_1e-8": 2,
            }
        },
        "stability": {
            "principal_angle_comparison": {"subspaces": subspaces}
        },
        "cumulative_ratio_at_16_32_64": {
            "16": 0.5,
            "32": 0.8,
            "64": 0.9,
        },
    }

    report = _markdown_report(metadata, variance)

    assert "Python 3.12.3, NumPy 2.2.6" in report
