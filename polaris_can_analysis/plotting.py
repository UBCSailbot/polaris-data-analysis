from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter

from polaris_can_analysis.analytics import (
    break_wrapped_angle_series,
    estimate_can_utilization_series,
    extract_ais_tracks,
    extract_gps_points,
    filter_ais_tracks_near_gps,
    project_to_local_km,
    robust_rolling_mean,
    trailing_rolling_max,
)
from polaris_can_analysis.basemap import add_satellite_basemap
from polaris_can_analysis.config import (
    AIS_TRACK_CONNECT_MAX_JUMP_KM,
    GEO_HORIZONTAL_PAD_FRAC,
    GEO_MARGIN_FRAC,
    SUBPLOT_HSPACE,
    SUBPLOT_WSPACE,
    UTILIZATION_MAX_WINDOW_S,
)
from polaris_can_analysis.models import ParsedFrame
from polaris_can_analysis.processing import signal_series


@dataclass
class BasemapSettings:
    enabled: bool = True
    cache_dir: Path = Path("data/tile_cache")
    provider_key: str = "esri_world_imagery"
    offline: bool = False
    alpha: float = 0.90
    max_tiles: int = 64


BASEMAP_SETTINGS = BasemapSettings()


def configure_basemap(
    enabled: bool = True,
    cache_dir: Path = Path("data/tile_cache"),
    provider_key: str = "esri_world_imagery",
    offline: bool = False,
    alpha: float = 0.90,
    max_tiles: int = 64,
) -> None:
    global BASEMAP_SETTINGS
    BASEMAP_SETTINGS = BasemapSettings(
        enabled=bool(enabled),
        cache_dir=Path(cache_dir),
        provider_key=provider_key,
        offline=bool(offline),
        alpha=float(alpha),
        max_tiles=max(1, int(max_tiles)),
    )


def style_axis(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor("#111827")
    ax.set_title(title, color="#E5E7EB", fontsize=11, pad=8)
    ax.grid(color="#374151", alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#374151")
    ax.tick_params(colors="#D1D5DB", labelsize=9)
    ax.xaxis.label.set_color("#D1D5DB")
    ax.yaxis.label.set_color("#D1D5DB")


def annotate_no_data(ax: plt.Axes, text: str = "No decoded data") -> None:
    ax.text(
        0.5,
        0.5,
        text,
        color="#9CA3AF",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
    )


def style_twin_y_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("none")
    ax.set_axisbelow(True)
    ax.tick_params(colors="#D1D5DB", labelsize=9)
    ax.yaxis.label.set_color("#D1D5DB")
    for spine in ax.spines.values():
        spine.set_color("#374151")


def apply_distance_axis_units(
    ax: plt.Axes, x_lo_km: float, x_hi_km: float, y_lo_km: float, y_hi_km: float
) -> None:
    """Use meters for small extents, kilometers otherwise."""
    span_km = max(abs(x_hi_km - x_lo_km), abs(y_hi_km - y_lo_km))
    if span_km < 1.0:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 1000:.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y * 1000:.0f}"))
        ax.set_xlabel("East/West Offset (m)")
        ax.set_ylabel("North/South Offset (m)")
    else:
        ax.set_xlabel("East/West Offset (km)")
        ax.set_ylabel("North/South Offset (km)")


def flatten_ais_track_points(
    ais_tracks: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    lon_parts: List[np.ndarray] = []
    lat_parts: List[np.ndarray] = []
    for lon, lat, _ in ais_tracks.values():
        if len(lon) == 0:
            continue
        lon_parts.append(lon)
        lat_parts.append(lat)
    if not lon_parts:
        return np.array([]), np.array([])
    return np.concatenate(lon_parts), np.concatenate(lat_parts)


def add_gps_speed_line(
    fig: plt.Figure,
    ax: plt.Axes,
    gps_x_km: np.ndarray,
    gps_y_km: np.ndarray,
    gps_speed_kmh: np.ndarray,
) -> None:
    """Draw GPS track as a line-only path, colored by speed when possible."""
    if len(gps_x_km) == 0:
        return

    if len(gps_x_km) == 1:
        ax.plot(gps_x_km, gps_y_km, color="#93C5FD", linewidth=1.8, alpha=0.95)
        ax.plot([], [], color="#93C5FD", linewidth=1.8, label="GPS track")
        return

    points = np.column_stack([gps_x_km, gps_y_km]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_speed = np.full(len(segments), np.nan, dtype=float)

    if len(gps_speed_kmh) >= 2:
        segment_speed = 0.5 * (gps_speed_kmh[:-1] + gps_speed_kmh[1:])

    finite_mask = np.isfinite(segment_speed)
    if np.any(finite_mask):
        speed_min = float(np.nanmin(segment_speed[finite_mask]))
        speed_max = float(np.nanmax(segment_speed[finite_mask]))
        if speed_max <= speed_min:
            speed_min -= 0.5
            speed_max += 0.5

        line = LineCollection(
            segments,
            cmap="turbo",
            norm=plt.Normalize(vmin=speed_min, vmax=speed_max),
            linewidths=1.8,
            alpha=0.95,
        )
        line.set_array(segment_speed)
        ax.add_collection(line)

        cbar = fig.colorbar(line, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(color="#D1D5DB", labelcolor="#D1D5DB", labelsize=8)
        cbar.set_label("Speed (km/h)", color="#D1D5DB", fontsize=9)
        cbar.outline.set_edgecolor("#374151")
    else:
        ax.plot(gps_x_km, gps_y_km, color="#93C5FD", linewidth=1.8, alpha=0.95)

    # Proxy legend entry to avoid point markers while keeping a clear legend.
    ax.plot([], [], color="#93C5FD", linewidth=1.8, label="GPS track")


def split_track_by_jump(
    x_km: np.ndarray,
    y_km: np.ndarray,
    max_jump_km: float = AIS_TRACK_CONNECT_MAX_JUMP_KM,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split a track into connected segments, breaking on large jumps."""
    if len(x_km) < 2:
        return []

    segs: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    jumps = np.hypot(np.diff(x_km), np.diff(y_km))

    for i, jump_km in enumerate(jumps, start=1):
        if not math.isfinite(float(jump_km)) or float(jump_km) > max_jump_km:
            if i - start >= 2:
                segs.append((x_km[start:i], y_km[start:i]))
            start = i

    if len(x_km) - start >= 2:
        segs.append((x_km[start:], y_km[start:]))
    return segs


def add_ais_tracks_with_points(
    ax: plt.Axes,
    ais_tracks: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    origin_lon: float,
    origin_lat: float,
    max_jump_km: float = AIS_TRACK_CONNECT_MAX_JUMP_KM,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Plot AIS per-MMSI points and segmented paths; return all projected points."""
    ais_x_parts: List[np.ndarray] = []
    ais_y_parts: List[np.ndarray] = []
    if not ais_tracks:
        return ais_x_parts, ais_y_parts

    mmsi_sorted = sorted(ais_tracks.keys())
    colors = plt.cm.plasma(np.linspace(0.15, 0.90, max(1, len(mmsi_sorted))))
    added_line_legend = False
    added_point_legend = False

    for idx, mmsi in enumerate(mmsi_sorted):
        lon, lat, _ = ais_tracks[mmsi]
        if len(lon) == 0:
            continue
        ais_x_km, ais_y_km = project_to_local_km(lon, lat, origin_lon, origin_lat)
        ais_x_parts.append(ais_x_km)
        ais_y_parts.append(ais_y_km)

        point_label = "AIS readings" if not added_point_legend else "_nolegend_"
        ax.scatter(
            ais_x_km,
            ais_y_km,
            color=colors[idx],
            s=11,
            alpha=0.35,
            edgecolors="none",
            label=point_label,
        )
        added_point_legend = True

        for seg_x, seg_y in split_track_by_jump(
            ais_x_km, ais_y_km, max_jump_km=max_jump_km
        ):
            line_label = "AIS ship tracks" if not added_line_legend else "_nolegend_"
            ax.plot(
                seg_x,
                seg_y,
                color=colors[idx],
                linewidth=1.2,
                alpha=0.72,
                label=line_label,
            )
            added_line_legend = True

    return ais_x_parts, ais_y_parts


def draw_frame_counts_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    frames: List[ParsedFrame],
    decoded_rows: List[Dict[str, object]],
) -> None:
    del fig, decoded_rows
    style_axis(ax, "Frame Counts")
    id_counts = Counter(frame.can_id for frame in frames if frame.can_id != "000")
    if not id_counts:
        annotate_no_data(ax)
        return

    ids_sorted = sorted(id_counts.keys(), key=lambda x: int(x, 16))
    vals = [id_counts[cid] for cid in ids_sorted]
    colors = plt.cm.viridis(np.linspace(0.2, 0.95, len(ids_sorted)))
    bars = ax.bar(ids_sorted, vals, color=colors, edgecolor="#E5E7EB", linewidth=0.4)
    ax.set_xlabel("CAN ID (hex)")
    ax.set_ylabel("Frames")
    max_val = max(vals) if vals else 0.0
    label_offset = max_val * 0.015
    y_top = max_val * 1.15 if max_val > 0 else 1.0
    ax.set_ylim(0.0, y_top)
    for bar, value in zip(bars, vals):
        y_label = min(bar.get_height() + label_offset, y_top * 0.985)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_label,
            f"{value}",
            ha="center",
            va="bottom",
            color="#F8FAFC",
            fontsize=8.5,
            rotation=90,
            clip_on=True,
        )


def draw_rudder_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    frames: List[ParsedFrame],
    decoded_rows: List[Dict[str, object]],
) -> None:
    del fig, frames
    style_axis(ax, "Rudder Actual vs Commanded")
    x_actual, y_actual = signal_series(decoded_rows, "actual_rudder_deg", "204")
    x_cmd, y_cmd = signal_series(decoded_rows, "commanded_rudder_deg", "204")
    if len(x_actual) > 0:
        ax.plot(x_actual, y_actual, color="#34D399", linewidth=1.5, label="Actual")
    if len(x_cmd) > 0:
        ax.plot(
            x_cmd, y_cmd, color="#F97316", linewidth=1.5, alpha=0.9, label="Commanded"
        )
    if len(x_actual) == 0 and len(x_cmd) == 0:
        annotate_no_data(ax)
        return
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Rudder Angle (deg)")
    ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#D1D5DB")


def draw_imu_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    frames: List[ParsedFrame],
    decoded_rows: List[Dict[str, object]],
) -> None:
    del fig, frames
    style_axis(ax, "IMU Roll, Pitch, and Heading")

    x_roll, y_roll = signal_series(decoded_rows, "imu_roll_deg", "204")
    x_pitch, y_pitch = signal_series(decoded_rows, "imu_pitch_deg", "204")
    x_heading, y_heading = signal_series(decoded_rows, "imu_heading_deg", "204")

    if len(x_roll) > 0:
        ax.plot(x_roll, y_roll, color="#60A5FA", linewidth=1.4, label="Roll", zorder=2)
    if len(x_pitch) > 0:
        ax.plot(x_pitch, y_pitch, color="#FBBF24", linewidth=1.4, label="Pitch", zorder=2)

    ax_heading = ax.twinx()
    style_twin_y_axis(ax_heading)
    # Keep heading trace on top of the primary axis artists.
    ax_heading.set_zorder(ax.get_zorder() + 1)
    ax_heading.patch.set_visible(False)
    if len(x_heading) > 0:
        ax_heading.plot(
            x_heading,
            y_heading,
            color="#F472B6",
            linewidth=1.3,
            alpha=0.95,
            label="Heading",
            zorder=8,
        )
        ax_heading.set_ylabel("Heading (deg)")

    if len(x_roll) == 0 and len(x_pitch) == 0 and len(x_heading) == 0:
        annotate_no_data(ax)
        ax_heading.set_yticks([])
        return

    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Roll/Pitch (deg)")
    lines = ax.get_lines() + ax_heading.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(
        lines, labels, facecolor="#111827", edgecolor="#374151", labelcolor="#D1D5DB"
    )


def draw_wind_angle_split_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    frames: List[ParsedFrame],
    decoded_rows: List[Dict[str, object]],
) -> None:
    del fig, frames
    style_axis(ax, "Wind Angle")

    x_040, y_040 = signal_series(decoded_rows, "wind_angle_deg", "040")
    x_041, y_041 = signal_series(decoded_rows, "wind_angle_deg", "041")

    has_data = False
    if len(x_040) > 0:
        x_040_plot, y_040_plot = break_wrapped_angle_series(x_040, y_040, wrap_deg=360.0)
        ax.plot(
            x_040_plot,
            y_040_plot,
            color="#22D3EE",
            linewidth=1.4,
            label="Sail Wind Sensor",
        )
        has_data = True
    if len(x_041) > 0:
        x_041_plot, y_041_plot = break_wrapped_angle_series(x_041, y_041, wrap_deg=360.0)
        ax.plot(
            x_041_plot,
            y_041_plot,
            color="#F59E0B",
            linewidth=1.4,
            label="Hull Wind Sensor",
        )
        has_data = True

    if not has_data:
        annotate_no_data(ax)
        return

    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Wind Angle (deg)")
    ax.legend(
        facecolor="#111827", edgecolor="#374151", labelcolor="#D1D5DB", loc="upper right"
    )


def draw_wind_speed_split_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    frames: List[ParsedFrame],
    decoded_rows: List[Dict[str, object]],
) -> None:
    del fig, frames
    style_axis(ax, "Wind Speed")

    x_040, y_040 = signal_series(decoded_rows, "wind_speed_knots", "040")
    x_041, y_041 = signal_series(decoded_rows, "wind_speed_knots", "041")

    has_data = False
    if len(x_040) > 0:
        ax.plot(x_040, y_040, color="#22D3EE", linewidth=1.4, label="Sail Wind Sensor")
        has_data = True
    if len(x_041) > 0:
        ax.plot(x_041, y_041, color="#F59E0B", linewidth=1.4, label="Hull Wind Sensor")
        has_data = True

    if not has_data:
        annotate_no_data(ax)
        return

    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Wind Speed (knots)")
    ax.legend(
        facecolor="#111827", edgecolor="#374151", labelcolor="#D1D5DB", loc="upper right"
    )


def _draw_geo_panel_common(
    fig: plt.Figure,
    ax: plt.Axes,
    frames: List[ParsedFrame],
    panel_title: str,
    gps_scaled_bounds: bool,
    use_basemap: bool,
) -> None:
    style_axis(ax, panel_title)
    gps_lon, gps_lat, gps_speed = extract_gps_points(frames)
    ais_tracks = extract_ais_tracks(frames)

    # Filter to nearby AIS contacts around the current GPS position.
    if len(gps_lon) > 0:
        gps_ref_lon = float(gps_lon[-1])
        gps_ref_lat = float(gps_lat[-1])
        ais_tracks = filter_ais_tracks_near_gps(ais_tracks, gps_ref_lon, gps_ref_lat)

    ais_lon_all, ais_lat_all = flatten_ais_track_points(ais_tracks)

    if len(gps_lon) > 0 and len(ais_lon_all) > 0:
        origin_lon = float(np.mean(np.concatenate([gps_lon, ais_lon_all])))
        origin_lat = float(np.mean(np.concatenate([gps_lat, ais_lat_all])))
    elif len(gps_lon) > 0:
        origin_lon = float(np.mean(gps_lon))
        origin_lat = float(np.mean(gps_lat))
    elif len(ais_lon_all) > 0:
        origin_lon = float(np.mean(ais_lon_all))
        origin_lat = float(np.mean(ais_lat_all))
    else:
        origin_lon = 0.0
        origin_lat = 0.0

    gps_x_km = np.array([])
    gps_y_km = np.array([])
    ais_x_parts: List[np.ndarray] = []
    ais_y_parts: List[np.ndarray] = []

    if len(gps_lon) > 0:
        gps_x_km, gps_y_km = project_to_local_km(gps_lon, gps_lat, origin_lon, origin_lat)
        add_gps_speed_line(fig, ax, gps_x_km, gps_y_km, gps_speed)

    ais_x_parts, ais_y_parts = add_ais_tracks_with_points(
        ax,
        ais_tracks,
        origin_lon,
        origin_lat,
        max_jump_km=AIS_TRACK_CONNECT_MAX_JUMP_KM,
    )

    has_ais_points = any(arr.size > 0 for arr in ais_x_parts)
    if len(gps_lon) == 0 and not has_ais_points:
        annotate_no_data(ax)
        return

    if gps_scaled_bounds:
        # Scale bounds to GPS track size (with margin). If GPS is unavailable,
        # fall back to all available points.
        if gps_x_km.size > 0:
            base_x = gps_x_km
            base_y = gps_y_km
        else:
            base_x = np.concatenate([arr for arr in ais_x_parts if arr.size > 0])
            base_y = np.concatenate([arr for arr in ais_y_parts if arr.size > 0])

        x_min = float(np.min(base_x))
        x_max = float(np.max(base_x))
        y_min = float(np.min(base_y))
        y_max = float(np.max(base_y))
        span_x = max(x_max - x_min, 1e-6)
        span_y = max(y_max - y_min, 1e-6)
        x_pad = 0.5 * span_x * GEO_MARGIN_FRAC
        y_pad = 0.5 * span_y * GEO_MARGIN_FRAC
        x_lo = x_min - x_pad
        x_hi = x_max + x_pad
        y_lo = y_min - y_pad
        y_hi = y_max + y_pad
    else:
        all_x_candidates = [gps_x_km] + ais_x_parts
        all_y_candidates = [gps_y_km] + ais_y_parts
        all_x = np.concatenate([arr for arr in all_x_candidates if arr.size > 0])
        all_y = np.concatenate([arr for arr in all_y_candidates if arr.size > 0])
        x_min = float(np.min(all_x))
        x_max = float(np.max(all_x))
        y_min = float(np.min(all_y))
        y_max = float(np.max(all_y))
        span_x = max(x_max - x_min, 1e-6)
        span_y = max(y_max - y_min, 1e-6)
        x_pad = 0.5 * span_x * GEO_MARGIN_FRAC
        y_pad = 0.5 * span_y * GEO_MARGIN_FRAC
        x_lo = x_min - x_pad
        x_hi = x_max + x_pad
        y_lo = y_min - y_pad
        y_hi = y_max + y_pad

        # Add explicit horizontal padding to avoid a narrow-looking geo panel,
        # but keep imagery panels tight so satellite tiles fill the full map area.
        if not use_basemap:
            extra_x = 0.5 * span_y * GEO_HORIZONTAL_PAD_FRAC
            x_lo -= extra_x
            x_hi += extra_x

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal", adjustable="datalim")
    # With adjustable="datalim", Matplotlib can further expand limits at draw-time
    # (often in X). Force that update now so basemap tile bounds match final view.
    ax.apply_aspect()
    x_lo_view, x_hi_view = ax.get_xlim()
    y_lo_view, y_hi_view = ax.get_ylim()

    if use_basemap and BASEMAP_SETTINGS.enabled:
        add_satellite_basemap(
            ax=ax,
            origin_lon_deg=origin_lon,
            origin_lat_deg=origin_lat,
            x_lo_km=x_lo_view,
            x_hi_km=x_hi_view,
            y_lo_km=y_lo_view,
            y_hi_km=y_hi_view,
            cache_dir=BASEMAP_SETTINGS.cache_dir,
            provider_key=BASEMAP_SETTINGS.provider_key,
            offline=BASEMAP_SETTINGS.offline,
            max_tiles=BASEMAP_SETTINGS.max_tiles,
            alpha=BASEMAP_SETTINGS.alpha,
        )

    x_lo_view, x_hi_view = ax.get_xlim()
    y_lo_view, y_hi_view = ax.get_ylim()
    apply_distance_axis_units(ax, x_lo_view, x_hi_view, y_lo_view, y_hi_view)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            facecolor="#111827",
            edgecolor="#374151",
            labelcolor="#D1D5DB",
        )


def draw_geo_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    frames: List[ParsedFrame],
    decoded_rows: List[Dict[str, object]],
) -> None:
    del decoded_rows
    _draw_geo_panel_common(
        fig,
        ax,
        frames,
        panel_title="GPS Track + AIS Ship Tracks",
        gps_scaled_bounds=False,
        use_basemap=False,
    )


def draw_geo_gps_scaled_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    frames: List[ParsedFrame],
    decoded_rows: List[Dict[str, object]],
) -> None:
    del decoded_rows
    _draw_geo_panel_common(
        fig,
        ax,
        frames,
        panel_title="GPS Track + AIS Ship Tracks (GPS-Scaled)",
        gps_scaled_bounds=True,
        use_basemap=False,
    )


def draw_geo_panel_imagery(
    fig: plt.Figure,
    ax: plt.Axes,
    frames: List[ParsedFrame],
    decoded_rows: List[Dict[str, object]],
) -> None:
    del decoded_rows
    _draw_geo_panel_common(
        fig,
        ax,
        frames,
        panel_title="GPS Track + AIS Ship Tracks (Historic Satallite Imagery)",
        gps_scaled_bounds=False,
        use_basemap=True,
    )


def draw_geo_gps_scaled_panel_imagery(
    fig: plt.Figure,
    ax: plt.Axes,
    frames: List[ParsedFrame],
    decoded_rows: List[Dict[str, object]],
) -> None:
    del decoded_rows
    _draw_geo_panel_common(
        fig,
        ax,
        frames,
        panel_title="GPS Track + AIS Ship Tracks (GPS-Scaled, Historic Satallite Imagery)",
        gps_scaled_bounds=True,
        use_basemap=True,
    )


def draw_pdb_voltages_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    frames: List[ParsedFrame],
    decoded_rows: List[Dict[str, object]],
) -> None:
    del fig, frames
    style_axis(ax, "PDB Cell Voltages")
    cell_signals = [
        ("cell_voltage_1_v", "#FCA5A5", "Filtered Cell 1"),
        ("cell_voltage_2_v", "#86EFAC", "Filtered Cell 2"),
        ("cell_voltage_3_v", "#93C5FD", "Filtered Cell 3"),
        ("cell_voltage_4_v", "#FDE68A", "Filtered Cell 4"),
    ]

    has_data = False
    for signal, color, display_label in cell_signals:
        xs, ys = signal_series(decoded_rows, signal, "206")
        if len(xs) == 0:
            continue
        ys_filtered = robust_rolling_mean(ys, window=5, outlier_sigma=3.0)
        ax.scatter(
            xs, ys, color=color, s=10, alpha=0.1, edgecolors="none", label="_nolegend_"
        )
        ax.plot(
            xs,
            ys_filtered,
            color=color,
            linewidth=1.5,
            marker="o",
            markersize=2.3,
            markerfacecolor=color,
            markeredgewidth=0.0,
            alpha=0.98,
            label=display_label,
        )
        has_data = True

    if not has_data:
        annotate_no_data(ax)
        return

    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Cell Voltage (V)")
    ax.legend(
        ax.get_lines(),
        [line.get_label() for line in ax.get_lines()],
        facecolor="#111827",
        edgecolor="#374151",
        labelcolor="#D1D5DB",
        fontsize=8,
        ncol=2,
        loc="upper right",
    )


def draw_can_utilization_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    frames: List[ParsedFrame],
    decoded_rows: List[Dict[str, object]],
) -> None:
    del fig, decoded_rows
    style_axis(ax, "Estimated CAN FD Utilization")
    series_specs = [(0.02, "#22EE29"), (0.1, "#38BDF8"), (0.5, "#8B5CF6")]

    has_data = False
    y_max = 0.0
    for window_s, color in series_specs:
        x, y = estimate_can_utilization_series(frames, window_s=window_s)
        if len(x) == 0:
            continue
        # Raw utilization points for this binning.
        ax.scatter(
            x, y, color=color, s=8, alpha=0.10, edgecolors="none", label="_nolegend_"
        )

        # Smoothed envelope: trailing max utilization over 5 seconds.
        window_points = max(1, int(round(UTILIZATION_MAX_WINDOW_S / window_s)))
        y_max_5s = trailing_rolling_max(y, window_points=window_points)
        ax.plot(
            x,
            y_max_5s,
            color=color,
            linewidth=1.6,
            alpha=0.98,
            label=f"{window_s:.2f}s bins ({UTILIZATION_MAX_WINDOW_S:.0f}s max)",
        )
        has_data = True
        y_max = max(y_max, float(np.max(y)))
        y_max = max(y_max, float(np.max(y_max_5s)))

    if not has_data:
        annotate_no_data(ax)
        return

    ax.set_ylim(0.0, max(1.0, y_max * 1.15))
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Estimated Utilization (%)")
    ax.legend(
        facecolor="#111827", edgecolor="#374151", labelcolor="#D1D5DB", loc="upper right"
    )


def draw_battery_temps_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    frames: List[ParsedFrame],
    decoded_rows: List[Dict[str, object]],
) -> None:
    del fig, frames
    style_axis(ax, "Battery Pack Temperatures")
    x_t1, y_t1 = signal_series(decoded_rows, "temp_pack2_port_c", "206")
    x_t3, y_t3 = signal_series(decoded_rows, "temp_pack1_starboard_c", "206")

    has_data = False
    if len(x_t1) > 0:
        y_t1_filtered = robust_rolling_mean(y_t1, window=5, outlier_sigma=3.0)
        ax.scatter(
            x_t1,
            y_t1,
            color="#FB7185",
            s=10,
            alpha=0.1,
            edgecolors="none",
            label="_nolegend_",
        )
        ax.plot(
            x_t1,
            y_t1_filtered,
            color="#FB7185",
            linewidth=1.5,
            marker="o",
            markersize=2.3,
            markerfacecolor="#FB7185",
            markeredgewidth=0.0,
            alpha=0.98,
            label="Port Pack",
        )
        has_data = True
    if len(x_t3) > 0:
        y_t3_filtered = robust_rolling_mean(y_t3, window=5, outlier_sigma=3.0)
        ax.scatter(
            x_t3,
            y_t3,
            color="#22D3EE",
            s=10,
            alpha=0.1,
            edgecolors="none",
            label="_nolegend_",
        )
        ax.plot(
            x_t3,
            y_t3_filtered,
            color="#22D3EE",
            linewidth=1.5,
            marker="o",
            markersize=2.3,
            markerfacecolor="#22D3EE",
            markeredgewidth=0.0,
            alpha=0.98,
            label="Starboard Pack",
        )
        has_data = True

    if not has_data:
        annotate_no_data(ax)
        return

    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Temperature (C)")
    ax.legend(
        facecolor="#111827", edgecolor="#374151", labelcolor="#D1D5DB", loc="upper right"
    )


def draw_sensor_temp_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    frames: List[ParsedFrame],
    decoded_rows: List[Dict[str, object]],
) -> None:
    del fig, frames
    style_axis(ax, "Data Sensor: Temperature")
    x_temp, y_temp = signal_series(decoded_rows, "temperature_c", "100")
    if len(x_temp) == 0:
        annotate_no_data(ax)
        return
    ax.plot(x_temp, y_temp, color="#F59E0B", linewidth=1.5, label="Temperature")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Temperature (C)")
    ax.legend(
        facecolor="#111827", edgecolor="#374151", labelcolor="#D1D5DB", loc="upper right"
    )


def draw_sensor_ph_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    frames: List[ParsedFrame],
    decoded_rows: List[Dict[str, object]],
) -> None:
    del fig, frames
    style_axis(ax, "Data Sensor: pH")
    x_ph, y_ph = signal_series(decoded_rows, "ph", "110")
    if len(x_ph) == 0:
        annotate_no_data(ax)
        return
    ax.plot(x_ph, y_ph, color="#60A5FA", linewidth=1.5, label="pH")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("pH")
    ax.legend(
        facecolor="#111827", edgecolor="#374151", labelcolor="#D1D5DB", loc="upper right"
    )


def draw_sensor_cond_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    frames: List[ParsedFrame],
    decoded_rows: List[Dict[str, object]],
) -> None:
    del fig, frames
    style_axis(ax, "Data Sensor: Conductivity")
    x_cond, y_cond = signal_series(decoded_rows, "conductivity_us_cm", "120")
    if len(x_cond) == 0:
        annotate_no_data(ax)
        return

    ax.plot(
        x_cond,
        y_cond,
        color="#34D399",
        linewidth=1.5,
        alpha=0.95,
        label="Conductivity",
    )
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Conductivity (uS/cm)")
    ax.legend(
        facecolor="#111827", edgecolor="#374151", labelcolor="#D1D5DB", loc="upper right"
    )


PANEL_DRAWERS = {
    "frame_counts": draw_frame_counts_panel,
    "can_utilization": draw_can_utilization_panel,
    "rudder": draw_rudder_panel,
    "imu": draw_imu_panel,
    "wind_angle_split": draw_wind_angle_split_panel,
    "wind_speed_split": draw_wind_speed_split_panel,
    "geo": draw_geo_panel,
    "geo_gps_scaled": draw_geo_gps_scaled_panel,
    "geo_imagery": draw_geo_panel_imagery,
    "geo_gps_scaled_imagery": draw_geo_gps_scaled_panel_imagery,
    "pdb_voltages": draw_pdb_voltages_panel,
    "battery_temps": draw_battery_temps_panel,
    "sensor_temp": draw_sensor_temp_panel,
    "sensor_ph": draw_sensor_ph_panel,
    "sensor_cond": draw_sensor_cond_panel,
}

TIME_AXIS_PANEL_KEYS = {
    "can_utilization",
    "rudder",
    "imu",
    "wind_angle_split",
    "wind_speed_split",
    "pdb_voltages",
    "battery_temps",
    "sensor_temp",
    "sensor_ph",
    "sensor_cond",
}


def elapsed_time_bounds(frames: List[ParsedFrame]) -> Optional[Tuple[float, float]]:
    times = [float(frame.elapsed_s) for frame in frames if math.isfinite(frame.elapsed_s)]
    if not times:
        return None
    t_min = float(min(times))
    t_max = float(max(times))
    if t_max <= t_min:
        t_max = t_min + 1e-6
    return t_min, t_max


def add_on_water_start_marker(ax: plt.Axes, start_s: float) -> None:
    x_lo, x_hi = ax.get_xlim()
    x_min = min(x_lo, x_hi)
    x_max = max(x_lo, x_hi)
    if start_s > x_max:
        return

    start_clamped = max(start_s, x_min)
    # Highlight only the on-water part of the horizontal axis (not a full vertical line).
    ax.plot(
        [start_clamped, x_max],
        [0.0, 0.0],
        transform=ax.get_xaxis_transform(),
        color="#F43F5E",
        linewidth=4.0,
        alpha=0.95,
        solid_capstyle="butt",
        clip_on=True,
        zorder=10,
    )


def create_dashboard(
    frames: List[ParsedFrame],
    decoded_rows: List[Dict[str, object]],
    output_path: Path,
    source_name: str,
    title: str,
    panels: List[str],
    on_water_start_s: Optional[float] = None,
    show_on_water_marker: bool = False,
    subtitle_extra: str = "",
    time_margin_frac: float = 0.0,
) -> None:
    panel_count = len(panels)
    if panel_count == 0:
        return

    ncols = 2 if panel_count > 1 else 1
    nrows = int(math.ceil(panel_count / ncols))

    # Fixed physical margins make title/subtitle spacing consistent across
    # dashboards with different row counts.
    fig_width_in = 18.0
    row_height_in = 4.3
    header_height_in = 1.0
    footer_height_in = 0.45
    left_margin_in = 0.9
    right_margin_in = 1.0
    fig_height_in = max(
        5.2, (row_height_in * nrows) + header_height_in + footer_height_in
    )

    fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=130)
    fig.patch.set_facecolor("#0B1020")
    grid = fig.add_gridspec(
        nrows,
        ncols,
        hspace=SUBPLOT_HSPACE,
        wspace=SUBPLOT_WSPACE,
        left=left_margin_in / fig_width_in,
        right=1.0 - (right_margin_in / fig_width_in),
        bottom=footer_height_in / fig_height_in,
        top=1.0 - (header_height_in / fig_height_in),
    )
    time_bounds = elapsed_time_bounds(frames)
    if time_bounds is not None:
        t_min, t_max = time_bounds
        span = max(t_max - t_min, 1e-6)
        pad = max(0.0, float(time_margin_frac)) * span
        time_bounds = (t_min - pad, t_max + pad)

    for idx, panel_key in enumerate(panels):
        row = idx // ncols
        col = idx % ncols
        ax = fig.add_subplot(grid[row, col])
        drawer = PANEL_DRAWERS.get(panel_key)
        if drawer is None:
            style_axis(ax, f"Unknown Panel: {panel_key}")
            annotate_no_data(ax, "Unknown panel key")
            continue
        drawer(fig, ax, frames, decoded_rows)
        if panel_key in TIME_AXIS_PANEL_KEYS and time_bounds is not None:
            ax.set_xlim(*time_bounds)
        if (
            show_on_water_marker
            and on_water_start_s is not None
            and panel_key in TIME_AXIS_PANEL_KEYS
        ):
            add_on_water_start_marker(ax, on_water_start_s)

    for idx in range(panel_count, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        ax = fig.add_subplot(grid[row, col])
        ax.axis("off")

    title_y = 1.0 - (0.30 / fig_height_in)
    subtitle_y = 1.0 - (0.60 / fig_height_in)
    fig.text(
        0.5,
        title_y,
        title,
        ha="center",
        va="center",
        color="#F8FAFC",
        fontsize=18,
        fontweight="bold",
    )
    subtitle = f"Source: {source_name} | Frames: {len(frames):,} | Decoded signals: {len(decoded_rows):,}"
    if subtitle_extra:
        subtitle = f"{subtitle} | {subtitle_extra}"
    fig.text(
        0.5, subtitle_y, subtitle, ha="center", va="center", color="#CBD5E1", fontsize=10
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)
