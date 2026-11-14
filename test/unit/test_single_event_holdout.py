import json

import numpy as np

from scripts.evaluate_single_event_holdout import evaluate


def test_holdout_evaluator_loads_global_template_features(tmp_path):
    y_train = np.tile(np.arange(2), 6)
    y_test = np.tile(np.arange(2), 4)
    X_train = np.column_stack((y_train, np.arange(len(y_train))))
    X_test = np.column_stack((y_test, np.arange(len(y_test))))
    metadata = {
        "noise_segments": {
            "train": [f"train-{index}" for index in range(6)],
            "val": ["val-0", "val-1"],
            "test": [f"test-{index}" for index in range(4)],
        }
    }

    (tmp_path / "pca_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.save(tmp_path / "X_global_template_train.npy", X_train)
    np.save(tmp_path / "X_global_template_test.npy", X_test)
    np.save(tmp_path / "y_train.npy", y_train)
    np.save(tmp_path / "y_test.npy", y_test)

    result = evaluate(
        tmp_path,
        iterations=20,
        seed=42,
        feature_source="global_template",
    )

    assert result["feature_source"] == "global_template"
    assert result["n_classes"] == 2
    assert result["n_test_pairs"] == 4
    assert result["test_used_for_training_or_tuning"] is False
