#!/usr/bin/env python3
"""Build candidate 4 s O3b splits from local GWOSC segment JSON and masks."""
import argparse
import csv
import json
import math
from pathlib import Path

GPS_START = 1256655618
GPS_END = 1269363618
TARGETS = {"train": 6495, "validation": 645, "test": 1305}
DETECTORS = ("H1", "L1")
FLAGS = ("DATA", "CBC_CAT1", "CBC_CAT2", "CBC_CAT3", "NO_CBC_HW_INJ")


def merge_segments(segments):
    merged = []
    for start, end in sorted((float(a), float(b)) for a, b in segments if b > a):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def intersect_segments(left, right):
    result = []
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if start < end:
            result.append((start, end))
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return result


def subtract_segments(segments, vetoes):
    result = []
    for start, end in segments:
        cursor = start
        for veto_start, veto_end in vetoes:
            if veto_end <= cursor:
                continue
            if veto_start >= end:
                break
            if cursor < veto_start:
                result.append((cursor, min(veto_start, end)))
            cursor = max(cursor, veto_end)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return result


def load_flag_segments(events_dir, detector, flag):
    path = events_dir / f"{detector}_{flag}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("dataset") != "O3b" or payload.get("id") != f"{detector}_{flag}"
            or payload.get("start") != GPS_START or payload.get("end") != GPS_END):
        raise ValueError(f"Unexpected timeline metadata in {path}")
    return merge_segments(payload.get("segments", []))


def load_gate_masks(mask_dir):
    masks = []
    with (mask_dir / "exclusion_intervals.csv").open(
            newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if row.get("kind") == "gate":
                masks.append((float(row["start"]), float(row["end"])))
    return merge_segments(masks)


def load_event_masks(catalog_path, margin):
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    events = payload.get("events")
    if not isinstance(events, dict):
        raise ValueError(f"Expected an events object in {catalog_path}")
    masks = []
    event_count = 0
    for event in events.values():
        if not isinstance(event, dict) or event.get("GPS") is None:
            continue
        gps = float(event["GPS"])
        if GPS_START <= gps < GPS_END:
            event_count += 1
            masks.append((max(GPS_START, gps - margin),
                          min(GPS_END, gps + margin)))
    return merge_segments(masks), event_count


def iter_windows(segments, window_seconds):
    for start, end in segments:
        window_start = math.ceil(start / window_seconds) * window_seconds
        while window_start + window_seconds <= end:
            yield int(window_start), int(window_start + window_seconds)
            window_start += window_seconds


def merge_windows(windows):
    merged = []
    for start, end in windows:
        if merged and merged[-1][1] == start:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-dir", default="data/o3b/events")
    parser.add_argument("--masks-dir", default="data/o3b/exclusions")
    parser.add_argument("--event-catalog", default="data/o3b/events/GWTC-3-confident.json")
    parser.add_argument("--event-margin", type=float, default=2.0)
    parser.add_argument("--output", default="data/o3b/windows")
    parser.add_argument("--window", type=int, default=4)
    args = parser.parse_args()
    if args.window <= 0 or args.event_margin < 0:
        parser.error("--window must be positive and --event-margin non-negative")

    events_dir = Path(args.events_dir)
    masks_dir = Path(args.masks_dir)
    event_catalog = Path(args.event_catalog)
    good = None
    for detector in DETECTORS:
        for flag in FLAGS:
            current = load_flag_segments(events_dir, detector, flag)
            good = current if good is None else intersect_segments(good, current)

    gate_masks = load_gate_masks(masks_dir)
    event_masks, event_count = load_event_masks(event_catalog, args.event_margin)
    masks = merge_segments(gate_masks + event_masks)
    valid_segments = merge_segments(subtract_segments(good or [], masks))
    targets = dict(TARGETS)
    target_total = sum(targets.values())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    selected = []
    available_windows = 0
    split_names = [name for name, count in targets.items() for _ in range(count)]
    for window in iter_windows(valid_segments, args.window):
        available_windows += 1
        if len(selected) < target_total:
            selected.append((window[0], window[1], split_names[len(selected)]))

    csv_path = output / f"candidate_windows_{args.window}s.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["gps_start", "gps_end", "duration_s", "detectors", "split"])
        for start, end, split in selected:
            writer.writerow([start, end, end - start, "H1+L1", split])

    split_ranges = {}
    for split in targets:
        windows = [(start, end) for start, end, label in selected if label == split]
        split_ranges[split] = [list(pair) for pair in merge_windows(windows)]
    (output / "gps_ranges.json").write_text(
        json.dumps(split_ranges, indent=2), encoding="utf-8"
    )

    summary = {
        "dataset": "O3b_4KHZ_R1",
        "gps_start": GPS_START,
        "gps_end_exclusive": GPS_END,
        "window_seconds": args.window,
        "quality_flags_intersected": [f"{detector}_{flag}"
                                       for detector in DETECTORS for flag in FLAGS],
        "gate_mask_csv": str(masks_dir / "exclusion_intervals.csv"),
        "event_catalog": str(event_catalog),
        "events_in_o3b": event_count,
        "event_margin_seconds": args.event_margin,
        "gate_policy": "exclude H1/L1 gate intervals including taper, as listed in the local mask CSV",
        "candidate_windows_available": available_windows,
        "target_windows": targets,
        "assigned_windows": {split: sum(1 for _, _, label in selected if label == split)
                             for split in targets},
        "targets_met": available_windows >= target_total,
        "split_policy": "chronological and disjoint; no strain files downloaded",
        "status": "candidate, requires review of catalog/veto methodology before final use",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()