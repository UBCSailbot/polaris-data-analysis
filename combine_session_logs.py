#!/usr/bin/env python3
"""Combine each session's candumps into one chronological CAN frame CSV.

A session directory (``data/26Jul26_owt/``) holds many capture files, each a
slice of the same test. This merges every ``candump_*.csv`` in a session into a
single ``outputs/<session>/combined_can_frames.csv`` ordered by the absolute
ISO ``Timestamp``.

The output keeps the candump schema verbatim -- ``Timestamp``,
``Elapsed_Time_s``, ``CAN_Message`` -- so it drops straight into the existing
parsing path. AIS is carried in the CAN frames themselves (ID ``060``,
``SAIL_AIS``) and so is included; the separate ``ais_values_*.csv`` files are
already-decoded ship reports, not CAN frames, and are not merged here.

Note that ``Elapsed_Time_s`` is measured per capture file and therefore
restarts partway through the combined output; ``Timestamp`` is the only
session-wide ordering key.

Capture files are each written chronologically, so they are stream-merged
rather than read into memory.
"""

from __future__ import annotations

import argparse
import csv
import heapq
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from polaris_can_analysis.processing import (
    CANDUMP_GLOB,
    discover_candumps,
    session_of,
)

CANDUMP_COLUMNS = ["Timestamp", "Elapsed_Time_s", "CAN_Message"]

# Sorts after every real timestamp, so unparseable rows land at the end
# instead of silently jumping to the front of the session.
_SORTS_LAST = datetime.max


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help=(
            "Directory containing candump CSVs. Searched recursively, so the "
            "default sweeps every data/<session>/ folder; pass a single session "
            "directory to combine just that test."
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
        default="combined_can_frames.csv",
        help="Name of the combined CSV written inside each session folder.",
    )
    parser.add_argument(
        "--glob",
        default=CANDUMP_GLOB,
        help="Glob pattern for candump files within --data-dir (matched at any depth).",
    )
    return parser.parse_args()


def group_by_session(paths: List[Path], data_dir: Path) -> Dict[str, List[Path]]:
    sessions: Dict[str, List[Path]] = defaultdict(list)
    for path in paths:
        sessions[session_of(path, data_dir)].append(path)
    return dict(sorted(sessions.items()))


def sort_key(timestamp: str) -> datetime:
    try:
        return datetime.fromisoformat(timestamp)
    except ValueError:
        return _SORTS_LAST


def candump_rows(path: Path) -> Iterator[Tuple[datetime, List[str]]]:
    """Yield ``(sort key, row)`` for one candump, passing its fields through."""
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            message = (raw.get("CAN_Message") or "").strip()
            if not message:
                continue
            timestamp = (raw.get("Timestamp") or "").strip()
            elapsed = (raw.get("Elapsed_Time_s") or "").strip()
            yield sort_key(timestamp), [timestamp, elapsed, message]


def combine_session(paths: List[Path], output_path: Path) -> int:
    """Merge one session's candumps into ``output_path``. Returns the row count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CANDUMP_COLUMNS)
        # Every capture file is written chronologically, so merging on the
        # absolute timestamp yields a sorted session without buffering rows.
        merged = heapq.merge(
            *(candump_rows(path) for path in paths), key=lambda item: item[0]
        )
        for _, row in merged:
            writer.writerow(row)
            rows += 1
    return rows


def main() -> None:
    args = parse_args()
    candumps = discover_candumps(args.data_dir, args.glob)
    if not candumps:
        raise FileNotFoundError(
            f"No files matching {args.glob} under {args.data_dir} (searched recursively)"
        )

    sessions = group_by_session(candumps, args.data_dir)
    written: List[Path] = []
    grand_rows = 0

    for session, paths in sessions.items():
        output_path = args.output_dir / session / args.filename
        print(f"{session}  ({len(paths)} candumps)")
        rows = combine_session(paths, output_path)
        print(f"  -> {output_path}  frames={rows:,}\n")
        grand_rows += rows
        written.append(output_path)

    print(f"Sessions: {len(sessions)}")
    print(f"Files: {len(candumps)}")
    print(f"Total frames: {grand_rows:,}")
    for path in written:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
