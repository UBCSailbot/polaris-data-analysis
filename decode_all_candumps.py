#!/usr/bin/env python3
"""Decode candumps into one decoded_signals.csv per on-water-test session.

Candumps live under ``data/<session>/`` (e.g. ``data/26Jun6_owt/``). Each
session's candumps are parsed and decoded with the same logic as
analyze_can_frames.py, and its decoded signal rows are streamed into
``outputs/<session>/decoded_signals.csv``.

Within a session file the absolute ISO ``timestamp`` column distinguishes
sources; ``elapsed_s`` is per-file and repeats across the session's files.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from polaris_can_analysis.processing import (
    CANDUMP_GLOB,
    decode_frames,
    discover_candumps,
    parse_csv,
    session_of,
)

DECODED_COLUMNS = [
    "timestamp",
    "elapsed_s",
    "can_id",
    "frame_name",
    "signal",
    "value",
    "unit",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help=(
            "Directory containing candump CSVs. Searched recursively, so the "
            "default sweeps every data/<session>/ folder; pass a single session "
            "directory to decode just that test."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Root for per-session output folders. Default: outputs",
    )
    parser.add_argument(
        "--filename",
        default="decoded_signals.csv",
        help="Name of the decoded CSV written inside each session folder.",
    )
    parser.add_argument(
        "--glob",
        default=CANDUMP_GLOB,
        help="Glob pattern for candump files within --data-dir (matched at any depth).",
    )
    return parser.parse_args()


def group_by_session(candumps: List[Path], data_dir: Path) -> Dict[str, List[Path]]:
    sessions: Dict[str, List[Path]] = defaultdict(list)
    for path in candumps:
        sessions[session_of(path, data_dir)].append(path)
    return dict(sorted(sessions.items()))


def decode_session(paths: List[Path], output_path: Path) -> tuple[int, int]:
    """Decode one session's candumps into ``output_path``. Returns (frames, signals)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    total_signals = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DECODED_COLUMNS)
        writer.writeheader()
        for path in paths:
            frames = parse_csv(path)
            decoded_rows = decode_frames(frames)
            writer.writerows(decoded_rows)
            total_frames += len(frames)
            total_signals += len(decoded_rows)
            print(
                f"    {path.name:<34} frames={len(frames):>8,}  "
                f"signals={len(decoded_rows):>9,}"
            )
    return total_frames, total_signals


def main() -> None:
    args = parse_args()
    candumps = discover_candumps(args.data_dir, args.glob)
    if not candumps:
        raise FileNotFoundError(
            f"No files matching {args.glob} under {args.data_dir} (searched recursively)"
        )

    sessions = group_by_session(candumps, args.data_dir)
    grand_frames = 0
    grand_signals = 0
    written: List[Path] = []

    for session, paths in sessions.items():
        output_path = args.output_dir / session / args.filename
        print(f"{session}  ({len(paths)} files)")
        frames, signals = decode_session(paths, output_path)
        print(f"  -> {output_path}  frames={frames:,}  signals={signals:,}\n")
        grand_frames += frames
        grand_signals += signals
        written.append(output_path)

    print(f"Sessions: {len(sessions)}")
    print(f"Files: {len(candumps)}")
    print(f"Total frames: {grand_frames:,}")
    print(f"Total decoded signal rows: {grand_signals:,}")
    for path in written:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
