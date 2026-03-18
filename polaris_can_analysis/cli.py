"""Parse and visualize POLARIS CAN dump CSV logs.

The script expects CSV rows with the schema:
Timestamp,Elapsed_Time_s,CAN_Message

Example CAN_Message:
can0  204  [16]  A6 28 D6 44 DB 29 00 00 04 29 30 75 30 75 00 00

Output files:
1) parsed_frames.csv   (one row per CAN frame)
2) decoded_signals.csv (one row per decoded signal)
3) *_full.png and *_trimmed.png dashboards in output subfolders
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Dict, List

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "matplotlib")

from polaris_can_analysis.analytics import (
    detect_on_water_start,
    filter_decoded_rows_by_start,
    filter_frames_by_start,
)
from polaris_can_analysis.config import DASHBOARD_CONFIG
from polaris_can_analysis.models import ParsedFrame
from polaris_can_analysis.plotting import (
    configure_basemap,
    configure_wind_smoothing,
    create_dashboard,
)
from polaris_can_analysis.processing import (
    decode_frames,
    default_input_csv,
    parse_csv,
    write_decoded_signals_csv,
    write_parsed_frames_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input CSV file. Default: first CSV in ./data",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for parsed CSVs and plots",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for quick iteration",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--basemap",
        choices=["satellite", "none"],
        default="satellite",
        help="Background imagery for GPS/AIS panels.",
    )
    parser.add_argument(
        "--tile-cache-dir",
        type=Path,
        default=Path("data/tile_cache"),
        help="Directory used to read/write cached basemap tiles.",
    )
    parser.add_argument(
        "--basemap-offline",
        action="store_true",
        help="Use cached tiles only; do not fetch any tiles from the internet.",
    )
    parser.add_argument(
        "--wind-rolling-avg",
        action="store_true",
        help="Overlay rolling-average traces on wind angle/speed plots.",
    )
    parser.add_argument(
        "--wind-rolling-window-s",
        type=float,
        default=30.0,
        help="Rolling-average window in seconds for wind plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_basemap(
        enabled=(args.basemap == "satellite"),
        cache_dir=args.tile_cache_dir,
        provider_key="esri_world_imagery",
        offline=args.basemap_offline,
    )
    configure_wind_smoothing(
        enabled=args.wind_rolling_avg,
        window_s=args.wind_rolling_window_s,
    )

    input_csv = args.input if args.input is not None else default_input_csv()
    frames = parse_csv(input_csv, max_rows=args.max_rows)
    decoded_rows = decode_frames(frames)
    on_water_detection = detect_on_water_start(decoded_rows)
    on_water_start_s = on_water_detection.start_s

    parsed_frames_path = args.outdir / "parsed_frames.csv"
    decoded_signals_path = args.outdir / "decoded_signals.csv"
    write_parsed_frames_csv(frames, parsed_frames_path)
    write_decoded_signals_csv(decoded_rows, decoded_signals_path)

    dashboard_paths: List[Path] = []
    if not args.skip_plot:
        full_dashboard_dir = args.outdir / "full"
        on_water_dashboard_dir = args.outdir / "on_water"

        full_subtitle_extra = ""
        on_water_frames: List[ParsedFrame] = []
        on_water_decoded_rows: List[Dict[str, object]] = []
        if (
            on_water_start_s is not None
            and on_water_detection.threshold_us_cm is not None
            and on_water_detection.steady_state_us_cm is not None
        ):
            full_subtitle_extra = f"On-water start: {on_water_start_s:.0f}s"
            on_water_frames = filter_frames_by_start(frames, on_water_start_s)
            on_water_decoded_rows = filter_decoded_rows_by_start(decoded_rows, on_water_start_s)

        for output_name, cfg in DASHBOARD_CONFIG.items():
            title = str(cfg.get("title", "POLARIS CAN Dashboard"))
            panels_raw = cfg.get("panels", [])
            panels = [str(item) for item in panels_raw] if isinstance(panels_raw, list) else []
            base_output = Path(output_name)
            output_suffix = base_output.suffix if base_output.suffix else ".png"
            output_stem = base_output.stem if base_output.suffix else base_output.name
            full_output_name = base_output.with_name(f"{output_stem}_full{output_suffix}")
            trimmed_output_name = base_output.with_name(
                f"{output_stem}_trimmed{output_suffix}"
            )
            full_panels_dir = full_dashboard_dir / f"{full_output_name.stem}_panels"
            trimmed_panels_dir = (
                on_water_dashboard_dir / f"{trimmed_output_name.stem}_panels"
            )

            output_path = full_dashboard_dir / full_output_name
            create_dashboard(
                frames,
                decoded_rows,
                output_path,
                source_name=input_csv.name,
                title=title,
                panels=panels,
                on_water_start_s=on_water_start_s,
                show_on_water_marker=on_water_start_s is not None,
                subtitle_extra=full_subtitle_extra,
                time_margin_frac=0.01,
                individual_panels_dir=full_panels_dir,
            )
            dashboard_paths.append(output_path)

            if on_water_start_s is not None and len(on_water_frames) > 0:
                on_water_output_path = on_water_dashboard_dir / trimmed_output_name
                create_dashboard(
                    on_water_frames,
                    on_water_decoded_rows,
                    on_water_output_path,
                    source_name=input_csv.name,
                    title=f"{title} (On-Water Segment)",
                    panels=panels,
                    on_water_start_s=None,
                    show_on_water_marker=False,
                    subtitle_extra=f"On-water Start: {on_water_start_s:.0f}s",
                    time_margin_frac=0.0,
                    individual_panels_dir=trimmed_panels_dir,
                )
                dashboard_paths.append(on_water_output_path)

    malformed = sum(1 for frame in frames if frame.parse_warning)
    print(f"Input file: {input_csv}")
    print(f"Total frames: {len(frames):,}")
    print(f"Frames with parse warnings: {malformed:,}")
    print(f"Decoded signal rows: {len(decoded_rows):,}")
    if on_water_start_s is not None:
        threshold_text = (
            f"{on_water_detection.threshold_us_cm:.2f}" if on_water_detection.threshold_us_cm is not None else "n/a"
        )
        steady_text = (
            f"{on_water_detection.steady_state_us_cm:.2f}"
            if on_water_detection.steady_state_us_cm is not None
            else "n/a"
        )
        print(
            "Detected on-water start from conductivity: "
            f"{on_water_start_s:.2f}s (threshold={threshold_text} uS/cm, steady={steady_text} uS/cm)"
        )
    else:
        print("On-water start not detected from conductivity; generated full-dataset dashboards only.")
    print(f"Wrote: {parsed_frames_path}")
    print(f"Wrote: {decoded_signals_path}")
    if args.skip_plot:
        print("Skipped dashboard plot (--skip-plot)")
    else:
        for dashboard_path in dashboard_paths:
            print(f"Wrote: {dashboard_path}")
