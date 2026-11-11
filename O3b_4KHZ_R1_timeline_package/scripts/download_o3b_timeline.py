#!/usr/bin/env python3
"""Download O3b timelines and create conservative, event-masked 4 s windows."""
import argparse
import csv
import json
from pathlib import Path
import requests

GPS_START = 1256655618
GPS_END = 1269363618
DURATION = GPS_END - GPS_START
TIMELINE_URL = (
    f"https://gwosc.org/timeline/segments/all/O3b/{GPS_START}/{DURATION}/"
)
EVENT_URL = "https://gwosc.org/eventapi/json/GWTC-3-confident/"


def fetch_json(url):
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    return r.json()


def get_segments(payload, flag):
    if flag not in payload:
        raise KeyError(f"Timeline response is missing required flag: {flag}")
    obj = payload[flag]
    if not isinstance(obj, dict) or "segments" not in obj:
        raise ValueError(f"Timeline flag has an invalid shape: {flag}")
    return [(int(a), int(b)) for a, b in obj.get("segments", []) if b > a]


def intersect_many(segment_lists):
    if not segment_lists or any(not x for x in segment_lists):
        return []
    current = sorted(segment_lists[0])
    for other in segment_lists[1:]:
        other = sorted(other)
        out, i, j = [], 0, 0
        while i < len(current) and j < len(other):
            a = max(current[i][0], other[j][0])
            b = min(current[i][1], other[j][1])
            if a < b:
                out.append((a, b))
            if current[i][1] <= other[j][1]:
                i += 1
            else:
                j += 1
        current = out
    return current


def subtract_many(segments, vetoes):
    """Subtract the union of veto intervals from segments."""
    vetoes = sorted(vetoes)
    result = []
    for start, end in segments:
        pieces = [(start, end)]
        for va, vb in vetoes:
            if vb <= start:
                continue
            if va >= end:
                break
            new = []
            for a, b in pieces:
                if vb <= a or va >= b:
                    new.append((a, b))
                else:
                    if a < va:
                        new.append((a, va))
                    if vb < b:
                        new.append((vb, b))
            pieces = new
            if not pieces:
                break
        result.extend(pieces)
    return result


def extract_events(payload):
    """Extract GPS times from GWTC-3 event JSON (handles common API shapes)."""
    events = payload.get("events", {})
    found = []
    if isinstance(events, dict):
        for name, item in events.items():
            if not isinstance(item, dict):
                continue
            gps = item.get("GPS") or item.get("gps") or item.get("GPS_time")
            if gps is not None:
                try:
                    found.append((str(name), float(gps)))
                except (TypeError, ValueError):
                    pass
    return sorted(found, key=lambda x: x[1])


def write_ascii(path, segments):
    with path.open("w", encoding="utf-8") as f:
        for a, b in sorted(segments):
            f.write(f"{a} {b} {b-a}\n")


def merge_windows(windows):
    """Merge consecutive fixed-size GPS windows into fetchable ranges."""
    merged = []
    for start, end in sorted(windows):
        if merged and merged[-1][1] == start:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def assign_split_windows(windows, targets):
    """Assign chronological windows only when every requested target is met."""
    if any(count <= 0 for count in targets.values()):
        raise ValueError("Split targets must be positive window counts")
    if len(windows) < sum(targets.values()):
        return {split: [] for split in targets}, {}

    split_windows = {}
    split_for_window = {}
    cursor = 0
    for split, count in targets.items():
        selected = windows[cursor:cursor + count]
        split_windows[split] = selected
        split_for_window.update({window: split for window in selected})
        cursor += count
    return split_windows, split_for_window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detectors", nargs="+", default=["H1", "L1"],
                        choices=["H1", "L1", "V1"])
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--event-margin", type=float, default=2.0,
                        help="Seconds excluded on either side of catalog event GPS.")
    parser.add_argument("--train-windows", type=int, default=6500)
    parser.add_argument("--validation-windows", type=int, default=650)
    parser.add_argument("--test-windows", type=int, default=1300)
    parser.add_argument("--output", default="data/o3b")
    args = parser.parse_args()
    if args.window <= 0 or args.event_margin < 0:
        raise ValueError("window debe ser positivo y event-margin no negativo")
    if len(set(args.detectors)) != len(args.detectors):
        raise ValueError("No repitas detectores")

    out = Path(args.output)
    tl_dir, ev_dir, win_dir = out / "timeline", out / "events", out / "windows"
    for p in (tl_dir, ev_dir, win_dir):
        p.mkdir(parents=True, exist_ok=True)

    print("Downloading O3b timeline from GWOSC...")
    timeline = fetch_json(TIMELINE_URL)
    (tl_dir / "O3b_all_timelines.json").write_text(
        json.dumps(timeline, indent=2), encoding="utf-8"
    )
    for flag, info in timeline.items():
        write_ascii(tl_dir / f"{flag}.txt",
                    [(int(a), int(b)) for a, b in info.get("segments", [])])

    # Coincident DATA across requested detectors.
    good = intersect_many([get_segments(timeline, f"{d}_DATA")
                           for d in args.detectors])

    # Conservative CBC veto: exclude CAT1/2/3 and CBC hardware-injection times.
    vetoes = []
    for d in args.detectors:
        for cat in (1, 2, 3):
            vetoes.extend(get_segments(timeline, f"{d}_CBC_CAT{cat}"))
        no_inj = get_segments(timeline, f"{d}_NO_CBC_HW_INJ")
        # Convert NO_INJ to veto = complement within O3b.
        vetoes.extend(subtract_many([(GPS_START, GPS_END)], no_inj))

    good = subtract_many(good, vetoes)

    # Download GWTC-3 confident event catalog and mask event neighborhoods.
    events = []
    try:
        event_payload = fetch_json(EVENT_URL)
        (ev_dir / "GWTC-3-confident.json").write_text(
            json.dumps(event_payload, indent=2), encoding="utf-8"
        )
        events = extract_events(event_payload)
    except Exception as exc:
        (ev_dir / "EVENT_DOWNLOAD_ERROR.txt").write_text(
            f"Could not download event catalog: {exc}\n"
            f"Please inspect {EVENT_URL} manually.\n", encoding="utf-8"
        )
        print(f"WARNING: event catalog download failed: {exc}")

    event_vetoes = [
        (max(GPS_START, int(gps - args.event_margin)),
         min(GPS_END, int(gps + args.event_margin)))
        for _, gps in events
    ]
    good = subtract_many(good, event_vetoes)

    # Align windows to multiples of window seconds relative to GPS epoch.
    windows = []
    for a, b in sorted(good):
        start = ((a + args.window - 1) // args.window) * args.window
        while start + args.window <= b:
            windows.append((start, start + args.window))
            start += args.window

    targets = {
        "train": args.train_windows,
        "validation": args.validation_windows,
        "test": args.test_windows,
    }
    target_total = sum(targets.values())
    split_windows, split_for_window = assign_split_windows(windows, targets)
    if len(windows) < target_total:
        print(
            f"WARNING: only {len(windows)} quality windows are available; "
            f"{target_total} required. No split assignment was made."
        )

    csv_path = win_dir / "candidate_windows_4s.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gps_start", "gps_end", "duration_s", "detectors",
                    "split"])
        for a, b in windows:
            w.writerow([
                a, b, b-a, "+".join(args.detectors),
                split_for_window.get((a, b), "UNASSIGNED"),
            ])

    split_ranges = {
        split: [[a, b] for a, b in merge_windows(selected)]
        for split, selected in split_windows.items()
    }
    if len(windows) >= target_total:
        (win_dir / "gps_ranges.json").write_text(
            json.dumps(split_ranges, indent=2), encoding="utf-8"
        )

    summary = {
        "dataset": "O3b_4KHZ_R1",
        "gps_start": GPS_START,
        "gps_end_exclusive": GPS_END,
        "detectors": args.detectors,
        "window_seconds": args.window,
        "event_margin_seconds": args.event_margin,
        "catalog_events_found": len(events),
        "candidate_windows": len(windows),
        "candidate_duration_seconds": len(windows) * args.window,
        "target_windows": targets,
        "assigned_windows": {split: len(items) for split, items in split_windows.items()},
        "targets_met": len(windows) >= target_total,
        "split_policy": "chronological, disjoint ranges; same test set across configurations",
        "note": "Targets assume 13 balanced classes: 500/50/100 examples per class."
    }
    (win_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
