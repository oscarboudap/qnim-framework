import numpy as np
import pytest

from scripts.diagnose_ppe_pipeline import _colored_noise


def test_colored_noise_matches_absolute_psd_scale():
    sample_rate = 256
    n_samples = 8192
    psd = 4e-6

    noise = _colored_noise(
        np.array([0.0, sample_rate / 2]),
        np.full(2, psd),
        sample_rate,
        n_samples,
        np.random.default_rng(42),
    )

    assert np.std(noise) == pytest.approx(
        np.sqrt(psd * sample_rate / 2), rel=0.03
    )
