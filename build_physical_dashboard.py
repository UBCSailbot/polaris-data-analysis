#!/usr/bin/env python3
"""Render dashboards per on-water-test session, or across the whole dataset.

The stock CLI renders dashboards from a single candump. This script groups the
candumps under ``data/`` by session folder (e.g. ``data/26Jun6_owt/``), rebases
each session's files onto one clock built from their absolute timestamps (so the
per-file timers don't overlap), and writes that session's dashboards to
``outputs/<session>/``.

With ``--combined`` it instead renders one set of dashboards over every session
at once, into ``outputs/full/``.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "matplotlib")

from polaris_can_analysis.config import DASHBOARD_CONFIG
from polaris_can_analysis.models import ParsedFrame
from polaris_can_analysis.plotting import configure_basemap, create_dashboard
from polaris_can_analysis.processing import (
    CANDUMP_GLOB,
    decode_frames,
    discover_candumps,
    parse_csv,
    session_of,
)


def _parse_iso(ts: str):
    raw = ts.strip().rstrip("Z")
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help=(
            "Directory containing candump CSVs. Searched recursively, so the "
            "default sweeps every data/<session>/ folder; pass a single session "
            "directory to render just that test."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Root for per-session output folders. Default: outputs",
    )
    parser.add_argument(
        "--config-key",
        nargs="*",
        default=None,
        help=(
            "Dashboard(s) to render, as keys into DASHBOARD_CONFIG "
            f"({', '.join(DASHBOARD_CONFIG)}). Default: all of them."
        ),
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help=(
            "Render one set of dashboards across every session onto a single "
            "global time axis, written to <output-dir>/full/ instead."
        ),
    )
    parser.add_argument("--glob", default=CANDUMP_GLOB)
    parser.add_argument("--basemap", choices=["satellite", "none"], default="satellite")
    parser.add_argument("--tile-cache-dir", type=Path, default=Path("data/tile_cache"))
    parser.add_argument("--basemap-offline", action="store_true")
    parser.add_argument(
        "--timezone",
        default=None,
        help=(
            "If set, the x-axis shows wall-clock time in this IANA tz "
            "(e.g. America/Los_Angeles for PDT/PST). Default: elapsed H:MM."
        ),
    )
    return parser.parse_args()


def rebase_to_global_clock(frames: List[ParsedFrame]) -> Optional[datetime]:
    """Put every frame on one clock from its ISO timestamp; return elapsed==0."""
    times = [(_parse_iso(f.timestamp), f) for f in frames]
    valid = [dt for dt, _ in times if dt is not None]
    if not valid:
        return None
    start = min(valid)
    for dt, frame in times:
        if dt is not None:
            frame.elapsed_s = (dt - start).total_seconds()
    # Candump timestamps are UTC; this is the absolute instant for elapsed == 0.
    return start.replace(tzinfo=timezone.utc)


def render_group(
    candumps: List[Path],
    output_dir: Path,
    config_keys: List[str],
    source_name: str,
    subtitle_extra: str,
    display_tz: Optional[ZoneInfo],
    name_suffix: str = "",
) -> List[Path]:
    all_frames: List[ParsedFrame] = []
    for path in candumps:
        all_frames.extend(parse_csv(path))
    print(f"  Parsed {len(all_frames):,} frames from {len(candumps)} files")

    time_origin = rebase_to_global_clock(all_frames)
    decoded_rows = decode_frames(all_frames)
    print(f"  Decoded {len(decoded_rows):,} signal rows")

    written: List[Path] = []
    for key in config_keys:
        cfg = DASHBOARD_CONFIG[key]
        panels = [str(p) for p in cfg.get("panels", [])]
        base = Path(key)
        stem = base.stem if base.suffix else base.name
        suffix = base.suffix if base.suffix else ".png"
        output_path = output_dir / f"{stem}{name_suffix}{suffix}"
        create_dashboard(
            all_frames,
            decoded_rows,
            output_path,
            source_name=source_name,
            title=str(cfg.get("title", "POLARIS Physical Data Dashboard")),
            panels=panels,
            on_water_start_s=None,
            show_on_water_marker=False,
            subtitle_extra=subtitle_extra,
            time_margin_frac=0.01,
            time_origin=time_origin,
            display_tz=display_tz,
        )
        print(f"  Wrote: {output_path}")
        written.append(output_path)
    return written


def main() -> None:
    args = parse_args()
    candumps = discover_candumps(args.data_dir, args.glob)
    if not candumps:
        raise FileNotFoundError(
            f"No files matching {args.glob} under {args.data_dir} (searched recursively)"
        )

    config_keys = args.config_key if args.config_key else list(DASHBOARD_CONFIG)
    unknown = [key for key in config_keys if key not in DASHBOARD_CONFIG]
    if unknown:
        raise SystemExit(
            f"Unknown --config-key {unknown}. Choose from: {', '.join(DASHBOARD_CONFIG)}"
        )

    display_tz = ZoneInfo(args.timezone) if args.timezone else None
    configure_basemap(
        enabled=(args.basemap == "satellite"),
        cache_dir=args.tile_cache_dir,
        provider_key="esri_world_imagery",
        offline=args.basemap_offline,
    )

    written: List[Path] = []
    if args.combined:
        sessions = sorted({session_of(p, args.data_dir) for p in candumps})
        print(f"combined ({len(candumps)} files across {len(sessions)} sessions)")
        written += render_group(
            candumps,
            args.output_dir / "full",
            config_keys,
            source_name=f"all candumps ({len(candumps)} files)",
            subtitle_extra=f"combined sessions: {', '.join(sessions)}",
            display_tz=display_tz,
            name_suffix="_full",
        )
    else:
        grouped: Dict[str, List[Path]] = {}
        for path in candumps:
            grouped.setdefault(session_of(path, args.data_dir), []).append(path)
        for session, paths in sorted(grouped.items()):
            print(f"{session}  ({len(paths)} files)")
            written += render_group(
                paths,
                args.output_dir / session,
                config_keys,
                source_name=f"{session} ({len(paths)} file{'s' if len(paths) != 1 else ''})",
                subtitle_extra=session,
                display_tz=display_tz,
            )
            print()

    print(f"Wrote {len(written)} dashboards.")


if __name__ == "__main__":
    main()
