import numpy as np

from src.infrastructure.dephasing_features import MatchedFilterFeatureExtractor


def test_matched_filter_features_retain_phase_and_amplitude():
    sample_rate = 256
    sample_count = 1024
    time = np.arange(sample_count) / sample_rate
    reference = np.sin(2 * np.pi * 32 * time)
    shifted = 1.5 * np.sin(2 * np.pi * 32 * time + 0.4)
    extractor = MatchedFilterFeatureExtractor(n_bands=8, f_low=8, f_high=100)
    extractor.fit(np.stack([reference, reference]), dt=1 / sample_rate)

    features = extractor.transform(np.stack([reference, shifted]), dt=1 / sample_rate)

    assert features.shape == (2, 16)
    assert np.linalg.norm(features[0] - features[1]) > 0.1
    assert len(extractor.get_feature_names()) == features.shape[1]


def test_matched_filter_features_use_per_event_templates_without_blowup():
    sample_rate = 256
    sample_count = 1024
    time = np.arange(sample_count) / sample_rate
    reference = np.sin(2 * np.pi * 32 * time)
    shifted = 1.2 * np.sin(2 * np.pi * 32 * time + 0.3)
    extractor = MatchedFilterFeatureExtractor(n_bands=8, f_low=8, f_high=100)
    extractor.fit(np.stack([reference, reference]), dt=1 / sample_rate)

    features = extractor.transform(
        np.stack([reference, shifted]),
        dt=1 / sample_rate,
        templates=np.stack([reference, reference]),
    )

    assert np.isfinite(features).all()
    assert np.max(np.abs(features)) < 2.0
    assert np.linalg.norm(features[0] - features[1]) > 0.1
