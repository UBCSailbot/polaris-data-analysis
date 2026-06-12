from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from polaris_can_analysis.processing import parse_can_message

CSV_FIELDS = ["Timestamp", "Elapsed_Time_s", "CAN_Message"]
FILENAME_DATE_RE = re.compile(r"(?P<date>20\d{6})(?:[_-]?(?P<time>\d{6}))?")
GPS_FRAME_ID = "070"


@dataclass
class LogEvent:
    source_idx: int
    row_idx: int
    timestamp: str
    elapsed_s: float
    local_s: float
    raw_message: str
    canonical_message: str
    gps_seconds_of_day: Optional[float] = None
    global_s: Optional[float] = None


@dataclass
class SourceLog:
    path: Path
    idx: int
    events: List[LogEvent] = field(default_factory=list)
    header_only: bool = False
    malformed_rows: int = 0
    time_notes: List[str] = field(default_factory=list)
    gps_offset_s: Optional[float] = None
    gps_residual_s: Optional[float] = None
    alignment_method: str = "unplaced"
    local_to_global_offset_s: Optional[float] = None

    @property
    def nonempty(self) -> bool:
        return len(self.events) > 0


@dataclass
class SequenceLink:
    a_idx: int
    b_idx: int
    b_offset_minus_a_offset_s: float
    matches: int
    residual_s: float
    span_s: float


@dataclass
class AppendStats:
    input_files: int
    header_only_files: int
    raw_events: int
    written_events: int
    duplicate_events: int
    sequence_links: List[SequenceLink]
    fallback_sources: List[Path]


def canonicalize_can_message(raw_message: str) -> str:
    try:
        interface, can_id, dlc, data, _warning = parse_can_message(raw_message)
    except ValueError:
        return " ".join(raw_message.strip().split())

    payload = " ".join(f"{byte:02X}" for byte in data)
    base = f"{interface}  {can_id}  [{dlc:02d}]"
    return f"{base}  {payload}" if payload else base


def gps_seconds_from_message(raw_message: str) -> Optional[float]:
    try:
        _interface, can_id, _dlc, data, _warning = parse_can_message(raw_message)
    except ValueError:
        return None

    if can_id != GPS_FRAME_ID or len(data) < 14:
        return None

    millis = data[8] | (data[9] << 8) | (data[10] << 16) | (data[11] << 24)
    minutes = data[12]
    hours = data[13]
    if hours > 23 or minutes > 59 or millis >= 60_000:
        return None
    return float(hours * 3600 + minutes * 60) + (millis / 1000.0)


def parse_iso_datetime(value: str) -> Optional[datetime]:
    value = value.strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def datetime_from_filename(path: Path) -> Optional[datetime]:
    match = FILENAME_DATE_RE.search(path.name)
    if not match:
        return None
    date_text = match.group("date")
    time_text = match.group("time") or "000000"
    try:
        return datetime.strptime(f"{date_text}{time_text}", "%Y%m%d%H%M%S")
    except ValueError:
        return None


def fallback_start_datetime(source: SourceLog) -> Optional[datetime]:
    if source.events:
        parsed = parse_iso_datetime(source.events[0].timestamp)
        if parsed is not None:
            return parsed
    return datetime_from_filename(source.path)


def date_for_output(sources: Sequence[SourceLog]) -> datetime:
    dates: List[datetime] = []
    for source in sources:
        path_dt = datetime_from_filename(source.path)
        if path_dt is not None:
            dates.append(path_dt.replace(hour=0, minute=0, second=0, microsecond=0))
    if dates:
        most_common_date = Counter(dates).most_common(1)[0][0]
        return most_common_date

    for source in sources:
        fallback = fallback_start_datetime(source)
        if fallback is not None:
            return fallback.replace(hour=0, minute=0, second=0, microsecond=0)
    return datetime(1970, 1, 1)


def seconds_from_base(value: datetime, base_date: datetime) -> float:
    return (value - base_date).total_seconds()


def normalize_periodic_offset(offset_s: float, period_s: float = 24 * 3600.0) -> float:
    return offset_s - round(offset_s / period_s) * period_s


def read_source(path: Path, idx: int) -> SourceLog:
    source = SourceLog(path=path, idx=idx)
    if path.stat().st_size == 0:
        source.header_only = True
        return source

    previous_local: Optional[float] = None
    first_elapsed: Optional[float] = None
    first_timestamp: Optional[datetime] = None

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            source.header_only = True
            return source

        for row_idx, row in enumerate(reader):
            raw_message = (row.get("CAN_Message") or "").strip()
            if not raw_message:
                continue

            elapsed_raw = (row.get("Elapsed_Time_s") or "").strip()
            timestamp_raw = (row.get("Timestamp") or "").strip()
            elapsed = math.nan
            try:
                elapsed = float(elapsed_raw)
            except ValueError:
                pass

            local_s: Optional[float] = None
            if math.isfinite(elapsed):
                if first_elapsed is None:
                    first_elapsed = elapsed
                local_s = elapsed - first_elapsed
            else:
                parsed_timestamp = parse_iso_datetime(timestamp_raw)
                if parsed_timestamp is not None:
                    if first_timestamp is None:
                        first_timestamp = parsed_timestamp
                    local_s = (parsed_timestamp - first_timestamp).total_seconds()

            if local_s is None or not math.isfinite(local_s):
                local_s = 0.0 if previous_local is None else previous_local + 0.001
                source.time_notes.append(
                    "generated local time for rows with invalid timestamps"
                )

            if previous_local is not None and local_s <= previous_local:
                local_s = previous_local + 0.001
                source.time_notes.append("forced non-monotonic local time to increase")
            previous_local = local_s

            canonical_message = canonicalize_can_message(raw_message)
            if canonical_message == " ".join(raw_message.strip().split()):
                source.malformed_rows += 1

            source.events.append(
                LogEvent(
                    source_idx=idx,
                    row_idx=row_idx,
                    timestamp=timestamp_raw,
                    elapsed_s=elapsed,
                    local_s=local_s,
                    raw_message=raw_message,
                    canonical_message=canonical_message,
                    gps_seconds_of_day=gps_seconds_from_message(raw_message),
                )
            )

    source.header_only = len(source.events) == 0
    return source


def unwrap_seconds_of_day(
    values: Iterable[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    unwrapped: List[Tuple[float, float]] = []
    day_offset = 0.0
    previous: Optional[float] = None
    for local_s, seconds_of_day in values:
        gps_s = seconds_of_day + day_offset
        if previous is not None and gps_s < previous - 12 * 3600:
            day_offset += 24 * 3600
            gps_s = seconds_of_day + day_offset
        previous = gps_s
        unwrapped.append((local_s, gps_s))
    return unwrapped


def estimate_gps_offsets(
    sources: Sequence[SourceLog],
    min_points: int,
    max_residual_s: float,
) -> None:
    for source in sources:
        anchors = [
            (event.local_s, event.gps_seconds_of_day)
            for event in source.events
            if event.gps_seconds_of_day is not None
        ]
        if len(anchors) < min_points:
            continue

        unwrapped = unwrap_seconds_of_day((local_s, gps_s) for local_s, gps_s in anchors)
        offsets = [gps_s - local_s for local_s, gps_s in unwrapped]
        gps_offset = median(offsets)
        residuals = [abs(offset - gps_offset) for offset in offsets]
        gps_residual = median(residuals)
        if gps_residual <= max_residual_s:
            source.gps_offset_s = gps_offset
            source.gps_residual_s = gps_residual
            source.local_to_global_offset_s = gps_offset
            source.alignment_method = "gps"


def infer_wall_to_gps_shift_s(sources: Sequence[SourceLog], base_date: datetime) -> float:
    shifts: List[float] = []
    for source in sources:
        if (
            source.gps_offset_s is None
            or source.local_to_global_offset_s is None
            or not source.events
        ):
            continue
        fallback_dt = fallback_start_datetime(source)
        if fallback_dt is None:
            continue

        raw_gps_start_s = source.events[0].local_s + source.local_to_global_offset_s
        wall_start_s = seconds_from_base(fallback_dt, base_date)
        shifts.append(normalize_periodic_offset(raw_gps_start_s - wall_start_s))

    return median(shifts) if shifts else 0.0


def normalize_gps_day_offsets(
    sources: Sequence[SourceLog],
    base_date: datetime,
    wall_to_gps_shift_s: float,
) -> None:
    for source in sources:
        if source.gps_offset_s is None or source.local_to_global_offset_s is None:
            continue
        fallback_dt = fallback_start_datetime(source)
        if fallback_dt is None or not source.events:
            continue

        raw_gps_start_s = source.events[0].local_s + source.local_to_global_offset_s
        target_start_s = seconds_from_base(fallback_dt, base_date) + wall_to_gps_shift_s
        day_shift_s = round((target_start_s - raw_gps_start_s) / (24 * 3600.0)) * (
            24 * 3600.0
        )
        source.gps_offset_s += day_shift_s
        source.local_to_global_offset_s += day_shift_s


def build_kgram_index(
    source: SourceLog,
    k: int,
    max_positions_per_key: int,
) -> Dict[Tuple[str, ...], List[Tuple[int, float]]]:
    index: Dict[Tuple[str, ...], List[Tuple[int, float]]] = defaultdict(list)
    messages = [event.canonical_message for event in source.events]
    if len(messages) < k:
        return index

    for pos in range(0, len(messages) - k + 1):
        key = tuple(messages[pos : pos + k])
        positions = index[key]
        if len(positions) < max_positions_per_key:
            positions.append((pos, source.events[pos].local_s))
    return index


def cluster_offsets(
    offsets: List[Tuple[float, float]], tolerance_s: float
) -> Tuple[float, int, float, float]:
    if not offsets:
        return 0.0, 0, math.inf, 0.0

    offsets = sorted(offsets, key=lambda item: item[0])
    best_cluster: List[Tuple[float, float]] = []
    window: Deque[Tuple[float, float]] = deque()
    for item in offsets:
        window.append(item)
        while window and item[0] - window[0][0] > tolerance_s:
            window.popleft()
        if len(window) > len(best_cluster):
            best_cluster = list(window)

    values = [item[0] for item in best_cluster]
    times = [item[1] for item in best_cluster]
    center = median(values)
    residual = median([abs(value - center) for value in values])
    span = max(times) - min(times) if len(times) > 1 else 0.0
    return center, len(best_cluster), residual, span


def find_sequence_links(
    sources: Sequence[SourceLog],
    k: int,
    min_matches: int,
    max_residual_s: float,
    cluster_tolerance_s: float,
    max_positions_per_key: int,
) -> List[SequenceLink]:
    indexes = [
        build_kgram_index(source, k=k, max_positions_per_key=max_positions_per_key)
        for source in sources
    ]
    links: List[SequenceLink] = []

    for a_idx in range(len(sources)):
        if not sources[a_idx].nonempty:
            continue
        for b_idx in range(a_idx + 1, len(sources)):
            if not sources[b_idx].nonempty:
                continue

            common_keys = set(indexes[a_idx]).intersection(indexes[b_idx])
            if not common_keys:
                continue

            offsets: List[Tuple[float, float]] = []
            for key in common_keys:
                for _a_pos, a_time in indexes[a_idx][key]:
                    for _b_pos, b_time in indexes[b_idx][key]:
                        offsets.append((a_time - b_time, a_time))

            offset, matches, residual, span = cluster_offsets(
                offsets, cluster_tolerance_s
            )
            if matches >= min_matches and residual <= max_residual_s:
                links.append(
                    SequenceLink(
                        a_idx=a_idx,
                        b_idx=b_idx,
                        b_offset_minus_a_offset_s=offset,
                        matches=matches,
                        residual_s=residual,
                        span_s=span,
                    )
                )

    links.sort(
        key=lambda link: (link.matches, link.span_s, -link.residual_s), reverse=True
    )
    return links


def propagate_sequence_offsets(
    sources: Sequence[SourceLog], links: Sequence[SequenceLink]
) -> None:
    changed = True
    while changed:
        changed = False
        for link in links:
            a = sources[link.a_idx]
            b = sources[link.b_idx]
            if (
                a.local_to_global_offset_s is not None
                and b.local_to_global_offset_s is None
            ):
                b.local_to_global_offset_s = (
                    a.local_to_global_offset_s + link.b_offset_minus_a_offset_s
                )
                b.alignment_method = "sequence"
                changed = True
            elif (
                b.local_to_global_offset_s is not None
                and a.local_to_global_offset_s is None
            ):
                a.local_to_global_offset_s = (
                    b.local_to_global_offset_s - link.b_offset_minus_a_offset_s
                )
                a.alignment_method = "sequence"
                changed = True


def place_unaligned_sources(
    sources: Sequence[SourceLog],
    base_date: datetime,
    wall_to_gps_shift_s: float,
) -> List[Path]:
    fallback_sources: List[Path] = []
    for source in sources:
        if not source.nonempty or source.local_to_global_offset_s is not None:
            continue

        fallback_dt = fallback_start_datetime(source)
        if fallback_dt is not None:
            source.local_to_global_offset_s = (
                seconds_from_base(fallback_dt, base_date)
                + wall_to_gps_shift_s
                - source.events[0].local_s
            )
            source.alignment_method = "wall-clock-fallback"
        else:
            latest_global = max(
                (
                    event.local_s + (known.local_to_global_offset_s or 0.0)
                    for known in sources
                    for event in known.events
                    if known.local_to_global_offset_s is not None
                ),
                default=0.0,
            )
            source.local_to_global_offset_s = latest_global + 1.0
            source.alignment_method = "append-after-previous-fallback"
        fallback_sources.append(source.path)
    return fallback_sources


def assign_global_times(sources: Sequence[SourceLog]) -> None:
    for source in sources:
        if source.local_to_global_offset_s is None:
            continue
        for event in source.events:
            event.global_s = event.local_s + source.local_to_global_offset_s


def duplicate_key(event: LogEvent) -> str:
    return event.canonical_message


def deduplicate_events(
    events: Sequence[LogEvent],
    tolerance_s: float,
) -> Tuple[List[LogEvent], int]:
    valid_events = [event for event in events if event.global_s is not None]
    events_by_key: Dict[str, List[LogEvent]] = defaultdict(list)
    for event in valid_events:
        events_by_key[duplicate_key(event)].append(event)

    dropped: set[Tuple[int, int]] = set()

    for key_events in events_by_key.values():
        if len(key_events) < 2:
            continue

        ordered = sorted(
            key_events,
            key=lambda item: (
                item.global_s if item.global_s is not None else math.inf,
                item.source_idx,
                item.row_idx,
            ),
        )
        n = len(ordered)
        parent = list(range(n))
        cluster_sources = [{event.source_idx} for event in ordered]
        cluster_members = [{idx} for idx in range(n)]

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(a_idx: int, b_idx: int) -> None:
            a_root = find(a_idx)
            b_root = find(b_idx)
            if a_root == b_root:
                return
            if cluster_sources[a_root].intersection(cluster_sources[b_root]):
                return
            if len(cluster_members[a_root]) < len(cluster_members[b_root]):
                a_root, b_root = b_root, a_root
            parent[b_root] = a_root
            cluster_sources[a_root].update(cluster_sources[b_root])
            cluster_members[a_root].update(cluster_members[b_root])

        candidates: List[Tuple[float, int, int]] = []
        for left_idx, left_event in enumerate(ordered):
            if left_event.global_s is None:
                continue
            right_idx = left_idx + 1
            while right_idx < n:
                right_event = ordered[right_idx]
                if right_event.global_s is None:
                    break
                delta_s = right_event.global_s - left_event.global_s
                if delta_s > tolerance_s:
                    break
                if left_event.source_idx != right_event.source_idx:
                    candidates.append((delta_s, left_idx, right_idx))
                right_idx += 1

        for _delta_s, left_idx, right_idx in sorted(candidates):
            union(left_idx, right_idx)

        clusters: Dict[int, List[int]] = defaultdict(list)
        for idx in range(n):
            clusters[find(idx)].append(idx)

        for member_indices in clusters.values():
            if len(member_indices) < 2:
                continue
            cluster_times = [
                ordered[idx].global_s
                for idx in member_indices
                if ordered[idx].global_s is not None
            ]
            cluster_center_s = median(cluster_times)
            keep_idx = min(
                member_indices,
                key=lambda idx: (
                    abs((ordered[idx].global_s or 0.0) - cluster_center_s),
                    ordered[idx].source_idx,
                    ordered[idx].row_idx,
                ),
            )
            for idx in member_indices:
                if idx == keep_idx:
                    continue
                event = ordered[idx]
                dropped.add((event.source_idx, event.row_idx))

    kept = [
        event
        for event in sorted(
            valid_events,
            key=lambda item: (
                item.global_s if item.global_s is not None else math.inf,
                item.source_idx,
                item.row_idx,
            ),
        )
        if (event.source_idx, event.row_idx) not in dropped
    ]
    return kept, len(dropped)


def iter_input_files(inputs: Sequence[Path]) -> List[Path]:
    paths: List[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            paths.extend(sorted(input_path.glob("*.csv")))
        else:
            paths.append(input_path)
    return sorted(dict.fromkeys(paths))


def append_logs(
    inputs: Sequence[Path],
    output_path: Path,
    *,
    sequence_length: int = 8,
    min_sequence_matches: int = 5,
    gps_min_points: int = 3,
    gps_max_residual_s: float = 2.0,
    sequence_cluster_tolerance_s: float = 0.050,
    sequence_max_residual_s: float = 0.025,
    max_positions_per_key: int = 20,
    dedupe_tolerance_s: float = 0.250,
) -> AppendStats:
    input_files = iter_input_files(inputs)
    sources = [read_source(path, idx) for idx, path in enumerate(input_files)]
    output_base_date = date_for_output(sources)

    estimate_gps_offsets(
        sources,
        min_points=gps_min_points,
        max_residual_s=gps_max_residual_s,
    )
    wall_to_gps_shift_s = infer_wall_to_gps_shift_s(sources, output_base_date)
    normalize_gps_day_offsets(sources, output_base_date, wall_to_gps_shift_s)
    links = find_sequence_links(
        sources,
        k=sequence_length,
        min_matches=min_sequence_matches,
        max_residual_s=sequence_max_residual_s,
        cluster_tolerance_s=sequence_cluster_tolerance_s,
        max_positions_per_key=max_positions_per_key,
    )
    propagate_sequence_offsets(sources, links)
    fallback_sources = place_unaligned_sources(
        sources,
        base_date=output_base_date,
        wall_to_gps_shift_s=wall_to_gps_shift_s,
    )
    assign_global_times(sources)

    all_events = [event for source in sources for event in source.events]
    kept_events, duplicate_count = deduplicate_events(
        all_events, tolerance_s=dedupe_tolerance_s
    )
    if kept_events:
        min_global_s = min(event.global_s or 0.0 for event in kept_events)
    else:
        min_global_s = 0.0

    output_base_datetime = output_base_date + timedelta(seconds=min_global_s)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for event in kept_events:
            elapsed_s = (event.global_s or 0.0) - min_global_s
            timestamp = output_base_datetime + timedelta(seconds=elapsed_s)
            writer.writerow(
                {
                    "Timestamp": timestamp.isoformat(timespec="microseconds"),
                    "Elapsed_Time_s": f"{elapsed_s:.6f}",
                    "CAN_Message": event.canonical_message,
                }
            )

    return AppendStats(
        input_files=len(input_files),
        header_only_files=sum(1 for source in sources if source.header_only),
        raw_events=len(all_events),
        written_events=len(kept_events),
        duplicate_events=duplicate_count,
        sequence_links=links,
        fallback_sources=fallback_sources,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append overlapping POLARIS CAN dump CSV logs into one de-duplicated CSV."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="CSV files or directories containing CSV files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--dedupe-tolerance-s",
        type=float,
        default=0.250,
        help=(
            "One-to-one match identical frames from different files within this "
            "time window."
        ),
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=8,
        help="Number of consecutive CAN messages used as an exact overlap anchor.",
    )
    parser.add_argument(
        "--min-sequence-matches",
        type=int,
        default=5,
        help="Minimum repeated sequence anchors required to align two files.",
    )
    parser.add_argument(
        "--gps-min-points",
        type=int,
        default=3,
        help="Minimum ID 070 GPS UTC frames required to trust bus time in one file.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    stats = append_logs(
        args.inputs,
        args.output,
        sequence_length=args.sequence_length,
        min_sequence_matches=args.min_sequence_matches,
        gps_min_points=args.gps_min_points,
        dedupe_tolerance_s=args.dedupe_tolerance_s,
    )

    print(f"Input files: {stats.input_files:,}")
    print(f"Header-only/blank files: {stats.header_only_files:,}")
    print(f"Raw frames read: {stats.raw_events:,}")
    print(f"Duplicate overlap frames removed: {stats.duplicate_events:,}")
    print(f"Frames written: {stats.written_events:,}")
    print(f"Sequence overlap links found: {len(stats.sequence_links):,}")
    if stats.fallback_sources:
        print("Files placed with wall-clock/name fallback:")
        for path in stats.fallback_sources:
            print(f"  {path}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
