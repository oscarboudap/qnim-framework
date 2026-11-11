import numpy as np

from src.domain.astrophysics.sstg.injectors.layer5_beyond_gr_complete import (
    Layer5BeyondGRInjector,
)
from src.domain.astrophysics.sstg.injectors.layer6_horizon_topology_complete import (
    Layer6HorizonTopologyInjector,
)
from src.infrastructure import physics_sstg_adapter


def test_extra_dimension_and_lqg_labels_use_their_own_injectors(monkeypatch):
    sample_rate = 128
    duration = 1.0
    base_strain = np.sin(2.0 * np.pi * 8.0 * np.arange(sample_rate) / sample_rate)
    monkeypatch.setattr(
        physics_sstg_adapter,
        "_generate_base_waveform",
        lambda *args, **kwargs: (base_strain.copy(), 1.0 / sample_rate),
    )
    calls = []

    def add_injector(h_plus, h_cross, **kwargs):
        calls.append(("add", kwargs))
        return h_plus * 0.5, h_cross

    def lqg_injector(h_plus, h_cross, **kwargs):
        calls.append(("lqg", kwargs))
        return h_plus * 0.75, h_cross

    monkeypatch.setattr(
        Layer5BeyondGRInjector, "inject_extra_dimensions_leakage", add_injector
    )
    monkeypatch.setattr(
        Layer6HorizonTopologyInjector, "inject_lqg_discrete_spectrum", lqg_injector
    )
    adapter = physics_sstg_adapter.PhysicsSSTGAdapter(
        waveform_duration=duration, sample_rate=sample_rate
    )

    add_waveform, _, _, add_metadata = adapter._generate_one(
        5, np.random.default_rng(12), (8.0, 30.0)
    )
    lqg_waveform, _, _, lqg_metadata = adapter._generate_one(
        11, np.random.default_rng(12), (8.0, 30.0)
    )

    assert [call[0] for call in calls] == ["add", "lqg"]
    assert calls[0][1]["n_extra_dims"] == 1
    assert calls[0][1]["fs"] == sample_rate
    assert calls[1][1]["area_quantum"] == 0.1
    assert calls[1][1]["fs"] == sample_rate
    assert add_metadata["theory_name"] == "extra-dimensions"
    assert lqg_metadata["theory_name"] == "loop-quantum-gravity"
    assert add_metadata["m1"] == lqg_metadata["m1"]
    assert add_metadata["m2"] == lqg_metadata["m2"]
    assert add_metadata["distance_mpc"] == lqg_metadata["distance_mpc"]
    assert not np.allclose(add_waveform, base_strain)
    assert not np.allclose(lqg_waveform, base_strain)
