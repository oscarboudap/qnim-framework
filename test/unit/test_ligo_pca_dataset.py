import importlib
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def _load_ligo_adapter(monkeypatch):
    package = ModuleType("src.infrastructure")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "infrastructure")]
    monkeypatch.setitem(sys.modules, "src.infrastructure", package)
    module_name = "src.infrastructure.ligo_pca_adapter"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return importlib.import_module(module_name).LIGOPCAAdapter


def _fake_strain(label, rng):
    return rng.normal(size=16) + label


def test_ligo_dataset_splits_before_train_only_pca(monkeypatch):
    adapter_type = _load_ligo_adapter(monkeypatch)
    seed = 13
    n_classes = 3
    n_train, n_val, n_test = 6, 2, 3
    n_total = n_train + n_val + n_test
    adapter = adapter_type(n_components=4, n_pca_fit=6, max_classes=n_classes)
    adapter._generate_one = _fake_strain

    dataset = adapter.generate_balanced_dataset(
        n_per_class=n_train,
        n_val_per_class=n_val,
        n_test_per_class=n_test,
        seed=seed,
        verbose=False,
    )

    rng = np.random.default_rng(seed)
    generated = [
        np.stack([_fake_strain(label, rng) for _ in range(n_total)])
        for label in range(n_classes)
    ]
    expected_train_raw = np.concatenate([rows[:n_train] for rows in generated])
    expected_test_raw = np.concatenate([
        rows[n_train + n_val:n_train + n_val + n_test] for rows in generated
    ])
    expected_pca = PCA(n_components=6, random_state=seed).fit(expected_train_raw)
    expected_train_pca = expected_pca.transform(expected_train_raw)[:, :4]
    expected_test_pca = expected_pca.transform(expected_test_raw)[:, :4]
    expected_scaler = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(expected_train_pca)

    assert dataset.X_train.shape == (n_classes * n_train, 4)
    assert dataset.X_val.shape == (n_classes * n_val, 4)
    assert dataset.X_test.shape == (n_classes * n_test, 4)
    np.testing.assert_array_equal(np.bincount(dataset.y_test), [n_test] * n_classes)
    np.testing.assert_allclose(dataset.pca.components_, expected_pca.components_)
    np.testing.assert_allclose(
        dataset.X_test,
        np.clip(expected_scaler.transform(expected_test_pca), -1.0, 1.0),
    )
    assert dataset.metadata["n_test_per_class"] == n_test
    assert len(dataset.metadata["pca_analysis"]["class_fisher_scores"]) == 6