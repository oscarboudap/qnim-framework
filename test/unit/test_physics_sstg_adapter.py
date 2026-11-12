import numpy as np
import pytest

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
