import importlib
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _load_adapter(monkeypatch):
    package = ModuleType("src.infrastructure")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "infrastructure")]
    monkeypatch.setitem(sys.modules, "src.infrastructure", package)
    for module_name in (
        "src.infrastructure.ligo_pca_adapter",
        "src.infrastructure.ligo_noise_injection_adapter",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    return importlib.import_module("src.infrastructure.ligo_noise_injection_adapter")


def _noise_segments(module):
    rng = np.random.default_rng(42)
    counts = {"train": 18, "val": 6, "test": 9}
    return {
        split: [
            module.NoiseSegment(
                segment_id=f"H1-{split}-segment-{index}",
                data=rng.normal(0.0, 1e-21, 128),
                sample_rate=128,
                detector="H1",
                gps_start=float(index * 2 + {"train": 0, "val": 100, "test": 200}[split]),
            )
            for index in range(counts[split])
        ]
        for split in ("train", "val", "test")
    }


def _waveform_generator(sample_rate=128, size=128):
    def generate(theory_class, rng):
        time = np.arange(size) / sample_rate
        waveform = np.sin(2 * np.pi * (8 + 4 * theory_class) * time)
        return waveform, {"theory_class": theory_class}
    return generate


def test_noise_injection_dataset_has_disjoint_balanced_splits(monkeypatch):
    module = _load_adapter(monkeypatch)
    adapter = module.LIGONoiseInjectionAdapter(
        noise_segments=_noise_segments(module),
        sample_rate=128,
        duration=1.0,
        n_components=4,
        n_pca_fit=6,
        target_snr_range=(8.0, 12.0),
        max_classes=3,
        waveform_generator=_waveform_generator(),
        f_low=2.0,
        f_high=60.0,
    )

    dataset = adapter.generate_balanced_dataset(
        n_per_class=6,
        n_val_per_class=2,
        n_test_per_class=3,
        seed=7,
        verbose=False,
    )

    assert dataset.X_train.shape == (18, 4)
    assert dataset.X_val.shape == (6, 4)
    assert dataset.X_test.shape == (9, 4)
    np.testing.assert_array_equal(np.bincount(dataset.y_train), [6, 6, 6])
    np.testing.assert_array_equal(np.bincount(dataset.y_val), [2, 2, 2])
    np.testing.assert_array_equal(np.bincount(dataset.y_test), [3, 3, 3])
    assert len(set(dataset.metadata["noise_segments"]["train"]).intersection(
        dataset.metadata["noise_segments"]["test"]
    )) == 0
    assert np.all((dataset.snr_train >= 8.0) & (dataset.snr_train <= 12.0))
    assert len(dataset.metadata["event_metadata"]["test"]) == 9


def test_noise_injection_rejects_reused_segment_ids_between_splits(monkeypatch):
    module = _load_adapter(monkeypatch)
    segments = _noise_segments(module)
    segments["test"][0] = module.NoiseSegment(
        segment_id=segments["train"][0].segment_id,
        data=segments["test"][0].data,
        sample_rate=128,
        detector="H1",
        gps_start=200.0,
    )

    with pytest.raises(ValueError, match="IDs de segmentos deben ser disjuntos"):
        module.LIGONoiseInjectionAdapter(
            noise_segments=segments,
            sample_rate=128,
            duration=1.0,
            n_components=4,
            n_pca_fit=6,
            max_classes=3,
            waveform_generator=_waveform_generator(),
            f_low=2.0,
            f_high=60.0,
        )


def test_noise_injection_rejects_overlapping_gps_intervals_with_different_ids(monkeypatch):
    module = _load_adapter(monkeypatch)
    segments = _noise_segments(module)
    segments["test"][0] = module.NoiseSegment(
        segment_id="new-id-for-overlapping-data",
        data=segments["test"][0].data,
        sample_rate=128,
        detector="H1",
        gps_start=segments["train"][0].gps_start + 0.5,
    )

    with pytest.raises(ValueError, match="intervalos GPS deben ser disjuntos"):
        module.LIGONoiseInjectionAdapter(
            noise_segments=segments,
            sample_rate=128,
            duration=1.0,
            n_components=4,
            n_pca_fit=6,
            max_classes=3,
            waveform_generator=_waveform_generator(),
            f_low=2.0,
            f_high=60.0,
        )


def test_gwosc_segment_loader_accepts_multiple_ranges_per_split(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    module = _load_adapter(monkeypatch)
    gwpy_module = ModuleType("gwpy")
    timeseries_module = ModuleType("gwpy.timeseries")
    fetched = []

    class FakeTimeSeries:
        @staticmethod
        def get(channel, start, end, **kwargs):
            fetched.append((channel, start, end, kwargs))
            values = np.arange(int((end - start) * 4096), dtype=float)
            return SimpleNamespace(value=values)

    timeseries_module.TimeSeries = FakeTimeSeries
    gwpy_module.timeseries = timeseries_module
    monkeypatch.setitem(sys.modules, "gwpy", gwpy_module)
    monkeypatch.setitem(sys.modules, "gwpy.timeseries", timeseries_module)

    segments = module.LIGONoiseInjectionAdapter.fetch_gwosc_segments(
        detector="H1",
        gps_ranges={
            "train": [(100, 104), (200, 204)],
            "val": [(300, 304)],
            "test": [(400, 404)],
        },
        sample_rate=4096,
        segment_duration=2,
    )

    assert len(fetched) == 4
    assert all(call[0] == "H1:GWOSC-4KHZ_R1_STRAIN" for call in fetched)
    assert all(call[3]["source"] == "gwdatafind" for call in fetched)
    assert all(call[3]["urltype"] == "osdf" for call in fetched)
    assert all(call[3]["host"] == "datafind.gwosc.org" for call in fetched)
    assert [segment.gps_start for segment in segments["train"]] == [100, 102, 200, 202]
    assert segments["train"][0].segment_id != segments["train"][2].segment_id
