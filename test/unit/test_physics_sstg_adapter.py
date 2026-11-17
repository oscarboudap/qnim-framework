import numpy as np
import pytest
import pycbc.waveform as pycbc_waveform

from src.domain.astrophysics.sstg.injectors.layer5_beyond_gr_complete import (
    Layer5BeyondGRInjector,
)
from src.domain.astrophysics.sstg.injectors.layer6_horizon_topology_complete import (
    Layer6HorizonTopologyInjector,
)
from src.infrastructure import physics_sstg_adapter
from src.infrastructure.stochastic_signal_generator import (
    _inject_graviton_mass,
    _inject_pn_deformation,
    _inject_qnm_subdominant,
)


def test_base_waveform_window_keeps_the_merger_peak(monkeypatch):
    full_strain = np.zeros(100)
    full_strain[70] = 1.0

    class FakeTimeSeries:
        def numpy(self):
            return full_strain

    monkeypatch.setattr(
        pycbc_waveform,
        "get_td_waveform",
        lambda **kwargs: (FakeTimeSeries(), FakeTimeSeries()),
    )

    window, dt = physics_sstg_adapter._generate_base_waveform(
        30.0, 20.0, 300.0, duration=0.2, sample_rate=100, f_lower=20.0
    )

    assert window.shape == (20,)
    assert dt == pytest.approx(0.01)
    assert np.argmax(np.abs(window)) == 18
    assert np.max(np.abs(window)) == pytest.approx(1.0)


def test_strict_physics_rejects_pycbc_failure_without_fallback(monkeypatch):
    def fail_generation(**kwargs):
        raise ValueError("invalid approximant")

    monkeypatch.setattr(pycbc_waveform, "get_td_waveform", fail_generation)
    monkeypatch.setattr(
        physics_sstg_adapter,
        "_fallback_chirp",
        lambda *args: (_ for _ in ()).throw(AssertionError("fallback was called")),
    )

    with pytest.raises(RuntimeError, match="strict_physics=True"):
        physics_sstg_adapter._generate_base_waveform(
            30.0, 20.0, 300.0, strict_physics=True
        )

    adapter = physics_sstg_adapter.PhysicsSSTGAdapter(strict_physics=True)
    assert adapter.strict_physics is True


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


def test_graviton_mass_uses_observational_scale_not_stress_multiplier(monkeypatch):
    sample_rate = 128
    duration = 1.0
    base_strain = np.sin(2.0 * np.pi * 8.0 * np.arange(sample_rate) / sample_rate)
    monkeypatch.setattr(
        physics_sstg_adapter,
        "_generate_base_waveform",
        lambda *args, **kwargs: (base_strain.copy(), 1.0 / sample_rate),
    )
    adapter = physics_sstg_adapter.PhysicsSSTGAdapter(
        waveform_duration=duration, sample_rate=sample_rate
    )

    _, _, _, metadata = adapter._generate_one(
        7, np.random.default_rng(12), (8.0, 30.0)
    )

    assert metadata["injection_parameters"]["graviton_mass_eV"] == 1.27e-23


def test_graviton_mass_phase_uses_non_reduced_compton_wavelength():
    frequency = np.array([100.0])
    spectrum = np.ones(1, dtype=complex)
    mass_ev = 1.27e-23
    distance_mpc = 200.0
    hbar_ev_s = 6.582e-16
    lambda_g_s = 2 * np.pi * hbar_ev_s / mass_ev
    distance_s = distance_mpc * 3.086e22 / 2.998e8
    expected_phase = -np.pi * distance_s / (lambda_g_s**2 * frequency[0])

    injected = _inject_graviton_mass(
        spectrum, frequency, distance_mpc, mg_eV=mass_ev
    )

    assert np.angle(injected[0]) == pytest.approx(expected_phase)


def test_theory_injection_can_reuse_paired_gr_waveform(monkeypatch):
    sample_rate = 128
    duration = 1.0
    base_strain = np.sin(2.0 * np.pi * 8.0 * np.arange(sample_rate) / sample_rate)
    monkeypatch.setattr(
        physics_sstg_adapter,
        "_generate_base_waveform",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("waveform regenerated")),
    )
    adapter = physics_sstg_adapter.PhysicsSSTGAdapter(
        waveform_duration=duration, sample_rate=sample_rate
    )

    waveform, _, _, metadata = adapter._generate_one(
        2,
        np.random.default_rng(12),
        (8.0, 30.0),
        base_waveform=base_strain,
    )

    assert waveform.shape == base_strain.shape
    assert metadata["m1"] > 0


def test_ppE_reference_phase_is_bounded_in_detector_band():
    frequencies = np.array([20.0, 50.0, 100.0, 200.0, 500.0])
    spectrum = np.ones(frequencies.shape, dtype=complex)

    deformed = _inject_pn_deformation(
        spectrum, frequencies, M_total_msun=60.0, delta_phi_hat=0.01
    )
    phase = np.angle(deformed / spectrum)

    assert np.isfinite(phase).all()
    assert np.max(np.abs(phase)) < 0.5


def test_qnm_injection_uses_coherent_phase_across_frequency_bins():
    frequencies = np.arange(1.0, 101.0)
    spectrum = np.ones(frequencies.shape, dtype=complex)
    rng = np.random.default_rng(42)

    injected = _inject_qnm_subdominant(spectrum, frequencies, mode="21", rng=rng)
    added = injected - spectrum
    active = np.abs(added) > np.max(np.abs(added)) * 1e-8

    assert active.any()
    assert np.std(np.unwrap(np.angle(added[active]))) < 1e-7


def test_physics_dataset_fits_templates_and_scaler_on_train_only(monkeypatch):
    sample_rate = 128
    duration = 1.0
    generated = {"count": 0}
    fit_calls = []
    transform_calls = []

    def generate_base(*args, **kwargs):
        generated["count"] += 1
        return (
            np.full(sample_rate, float(generated["count"])),
            1.0 / sample_rate,
        )

    class RecordingFeatureExtractor:
        def __init__(self, **kwargs):
            self.raw_features = None

        def fit(self, gr_strains, dt, mc_values=None):
            fit_calls.append((len(gr_strains), mc_values))
            return self

        def transform(self, strains, dt, mc_values=None):
            transform_calls.append(mc_values)
            self.raw_features = np.asarray(strains)[:, :12].copy()
            return self.raw_features.copy()

    monkeypatch.setattr(physics_sstg_adapter, "_generate_base_waveform", generate_base)
    monkeypatch.setattr(
        physics_sstg_adapter, "DephasingFeatureExtractor", RecordingFeatureExtractor
    )
    adapter = physics_sstg_adapter.PhysicsSSTGAdapter(
        waveform_duration=duration,
        sample_rate=sample_rate,
        max_classes=2,
    )

    dataset = adapter.generate_balanced_dataset(
        n_per_class=3,
        n_val_per_class=2,
        seed=7,
        verbose=False,
    )

    train_indices = np.array([0, 1, 2, 5, 6, 7])
    raw_features = dataset.feature_extractor.raw_features
    expected_mean = raw_features[train_indices].mean(axis=0)
    np.testing.assert_allclose(dataset.metadata["col_mean"], expected_mean)
    assert not np.allclose(dataset.metadata["col_mean"], raw_features.mean(axis=0))
    assert fit_calls == [(3, None)]
    assert transform_calls == [None]
    assert dataset.metadata["template_fit_split"] == "train_GR_only"
    assert dataset.metadata["chirp_mass_input"] is False
    assert dataset.metadata["scaler_fit_split"] == "train_only"
