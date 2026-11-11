import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "O3b_4KHZ_R1_timeline_package"
    / "scripts"
    / "download_o3b_timeline.py"
)


def _load_timeline_script(monkeypatch):
    requests_stub = ModuleType("requests")
    monkeypatch.setitem(sys.modules, "requests", requests_stub)
    spec = importlib.util.spec_from_file_location("o3b_timeline_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_assign_split_windows_uses_exact_balanced_targets():
    from unittest.mock import patch

    module = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location("o3b_timeline_script", SCRIPT_PATH)
    )
    spec = importlib.util.spec_from_file_location("o3b_timeline_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"requests": ModuleType("requests")}):
        spec.loader.exec_module(module)

    windows = [(start, start + 4) for start in range(0, 8450 * 4, 4)]
    targets = {"train": 6500, "validation": 650, "test": 1300}
    split_windows, assignment = module.assign_split_windows(windows, targets)

    assert {split: len(items) for split, items in split_windows.items()} == targets
    assert len(assignment) == sum(targets.values())
    assert split_windows["train"][-1][1] == split_windows["validation"][0][0]
    assert split_windows["validation"][-1][1] == split_windows["test"][0][0]


def test_assign_split_windows_does_not_assign_incomplete_targets(monkeypatch):
    module = _load_timeline_script(monkeypatch)
    windows = [(start, start + 4) for start in range(0, 100, 4)]

    split_windows, assignment = module.assign_split_windows(
        windows, {"train": 20, "validation": 5, "test": 5}
    )

    assert all(not items for items in split_windows.values())
    assert assignment == {}


def test_get_segments_fails_closed_when_required_flag_is_missing(monkeypatch):
    module = _load_timeline_script(monkeypatch)

    with pytest.raises(KeyError, match="H1_DATA"):
        module.get_segments({}, "H1_DATA")


def test_ascii_segments_are_written_as_real_lines(tmp_path, monkeypatch):
    module = _load_timeline_script(monkeypatch)
    output = tmp_path / "segments.txt"

    module.write_ascii(output, [(100, 104), (108, 112)])

    assert output.read_text(encoding="utf-8") == "100 104 4\n108 112 4\n"
