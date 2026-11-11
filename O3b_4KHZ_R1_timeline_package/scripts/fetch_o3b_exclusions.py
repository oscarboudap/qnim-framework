#!/usr/bin/env python3
"""Fetch small O3b gate and event metadata files and export exclusion masks."""
import argparse
import csv
import json
from pathlib import Path

import requests

GPS_START = 1256655618
GPS_END = 1269363618
ZENODO_FILES = {
    "H1": "https://zenodo.org/api/records/5636796/files/H1-O3_GATES_1238166018-31197600.txt/content",
    "L1": "https://zenodo.org/api/records/5636796/files/L1-O3_GATES_1238166018-31197600.txt/content",
}
IAS_EVENTS_URL = "https://zenodo.org/api/records/15540558/files/ias-o3b.json/content"


def fetch_text(url):
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    return response.text


def fetch_json(url):
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    return response.json()


def parse_gate_rows(text, detector):
    intervals = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) < 3:
            raise ValueError(f"Invalid {detector} gate row {line_number}: {line}")
        center, half_width, taper = map(float, fields[:3])
        start = max(GPS_START, center - half_width - taper)
        end = min(GPS_END, center + half_width + taper)
        if start < end:
            intervals.append({"start": start, "end": end, "detector": detector,
                              "kind": "gate", "source": "GWTC-3 Zenodo DQ products"})
    return intervals


def parse_ias_events(payload, margin):
    event_rows = payload.get("events")
    if not isinstance(event_rows, list):
        raise ValueError("IAS catalog has no events list")
    intervals = []
    for event in event_rows:
        if not isinstance(event, dict) or event.get("gps") is None:
            continue
        gps = float(event["gps"])
        start = max(GPS_START, gps - margin)
        end = min(GPS_END, gps + margin)
        if start < end:
            intervals.append({"start": start, "end": end, "detector": "H1+L1",
                              "kind": "catalog_event", "name": event.get("event_name", "unknown"),
                              "source": "IAS O3b catalog, Zenodo 15540558"})
    return intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-margin", type=float, default=2.0,
                        help="Seconds to mask on either side of each IAS event GPS time")
    parser.add_argument("--output", default="data/o3b/exclusions")
    args = parser.parse_args()
    if args.event_margin < 0:
        parser.error("--event-margin must be non-negative")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    gate_intervals = []
    for detector, url in ZENODO_FILES.items():
        gate_text = fetch_text(url)
        (output / f"{detector}_O3_gates.txt").write_text(gate_text, encoding="utf-8")
        gate_intervals.extend(parse_gate_rows(gate_text, detector))

    event_payload = fetch_json(IAS_EVENTS_URL)
    (output / "ias_o3b.json").write_text(
        json.dumps(event_payload, indent=2), encoding="utf-8"
    )
    event_intervals = parse_ias_events(event_payload, args.event_margin)
    all_intervals = sorted(gate_intervals + event_intervals,
                           key=lambda item: (item["start"], item["end"], item["kind"]))

    csv_path = output / "exclusion_intervals.csv"
    columns = ["start", "end", "duration_s", "detector", "kind", "name", "source"]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for interval in all_intervals:
            writer.writerow({**interval, "duration_s": interval["end"] - interval["start"],
                             "name": interval.get("name", "")})

    manifest = {
        "gps_start": GPS_START,
        "gps_end_exclusive": GPS_END,
        "event_catalog": "IAS O3b, 23 events; not a complete substitute for GWTC-3",
        "event_margin_seconds": args.event_margin,
        "gate_interval_policy": "center +/- zero half-width +/- taper, clipped to O3b",
        "gate_intervals": len(gate_intervals),
        "event_intervals": len(event_intervals),
        "intervals_csv": str(csv_path),
        "availability_segments_included": False,
        "warning": "These are exclusion masks only; they do not establish H1_DATA/L1_DATA availability.",
        "sources": {**ZENODO_FILES, "ias_events": IAS_EVENTS_URL},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()