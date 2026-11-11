"""Inspect GWOSC open-data quality segments for candidate GPS windows."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="List active GWOSC DATA-quality intervals for candidate off-source windows"
    )
    parser.add_argument("--gps-start", type=float, required=True)
    parser.add_argument("--gps-end", type=float, required=True)
    parser.add_argument("--detectors", nargs="+", choices=["H1", "L1", "V1"], default=["H1", "L1"])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gps_end <= args.gps_start:
        raise ValueError("gps-end debe ser mayor que gps-start")

    from gwpy.segments import DataQualityFlag

    result = {
        "gps_start": args.gps_start,
        "gps_end": args.gps_end,
        "detectors": {},
    }
    for detector in args.detectors:
        flag = DataQualityFlag.fetch_open_data(
            f"{detector}_DATA", args.gps_start, args.gps_end
        )
        result["detectors"][detector] = [
            [float(segment.start), float(segment.end)] for segment in flag.active
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Quality intervals written to {args.output}")


if __name__ == "__main__":
    main()
