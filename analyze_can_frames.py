#!/usr/bin/env python3
"""Parse and visualize POLARIS CAN dump CSV logs.

The script expects CSV rows with the schema:
Timestamp,Elapsed_Time_s,CAN_Message

Example CAN_Message:
can0  204  [16]  A6 28 D6 44 DB 29 00 00 04 29 30 75 30 75 00 00

Output files:
1) parsed_frames.csv   (one row per CAN frame)
2) decoded_signals.csv (one row per decoded signal)
3) physical_dashboard.png   (physical/navigation visualization set)
4) electrical_dashboard.png (electrical/power visualization set)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


CAN_MESSAGE_RE = re.compile(
    r"^(?P<iface>\S+)\s+(?P<can_id>[0-9A-Fa-f]+)\s+\[(?P<dlc>\d+)\](?:\s+(?P<data>.*))?$"
)

FRAME_NAMES: Dict[str, str] = {
    "001": "MAIN_HEADING",
    "002": "MAIN_TRIMTAB",
    "003": "POWER_OFF",
    "040": "SAIL_WIND",
    "041": "DATA_WIND",
    "060": "SAIL_AIS",
    "070": "PATH_GPS",
    "100": "DATA_TEMP",
    "110": "DATA_PH",
    "120": "DATA_COND",
    "130": "PDB_HEARTBEAT",
    "131": "RUDDER_HEARTBEAT",
    "132": "SAIL_HEARTBEAT",
    "133": "SENSE_HEARTBEAT",
    "200": "RUDDER_PID_DEBUG",
    "201": "MAIN_TO_WINGSAIL_DEBUG",
    "202": "MAIN_TO_PDB_DEBUG",
    "204": "RUDDER_TO_MAIN_DEBUG",
    "205": "WINGSAIL_TO_MAIN_DEBUG",
    "206": "PDB_TO_MAIN_DEBUG",
}

EARTH_RADIUS_KM = 6371.0088
AIS_NEARBY_THRESHOLD_DEG = 0.1
GEO_MARGIN_FRAC = 0.05
GEO_HORIZONTAL_PAD_FRAC = 0.10
CAN_NOMINAL_BITRATE_BPS = 500_000.0
CAN_DATA_BITRATE_BPS = 1_000_000.0
UTILIZATION_WINDOW_S = 0.5
CANFD_STUFFING_FACTOR = 1.15
UTILIZATION_MAX_WINDOW_S = 5.0
SUBPLOT_HSPACE = 0.22
SUBPLOT_WSPACE = 0.16

# Easy-to-edit dashboard grouping configuration.
# Keys are output PNG filenames, each mapped to a title and panel list.
DASHBOARD_CONFIG: Dict[str, Dict[str, object]] = {
    "physical_dashboard.png": {
        "title": "POLARIS Physical Data Dashboard",
        "panels": [
            "rudder",
            "imu",
            "geo",
            "geo_gps_scaled",
        ],
    },
    "electrical_dashboard.png": {
        "title": "POLARIS Electrical Data Dashboard",
        "panels": [
            "frame_counts",
            "can_utilization",
            "pdb_voltages",
            "battery_temps",
        ],
    },
    "sensor_dashboard.png": {
        "title": "POLARIS Sensor Dashboard",
        "panels": [
            "wind_angle_split",
            "wind_speed_split",
            "sensor_temp",
            "sensor_ph",
            "sensor_cond",
        ],
    },
}


@dataclass
class ParsedFrame:
    timestamp: str
    elapsed_s: float
    interface: str
    can_id: str
    can_id_int: int
    frame_name: str
    dlc: int
    data: List[int]
    raw_message: str
    parse_warning: str = ""


def le_u16(data: List[int], offset: int) -> int:
    return data[offset] | (data[offset + 1] << 8)


def le_u24(data: List[int], offset: int) -> int:
    return data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16)


def le_u32(data: List[int], offset: int) -> int:
    return (
        data[offset]
        | (data[offset + 1] << 8)
        | (data[offset + 2] << 16)
        | (data[offset + 3] << 24)
    )


def default_input_csv() -> Path:
    candidates = sorted(Path("data").glob("*.csv"))
    if not candidates:
        raise FileNotFoundError("No CSV files found in ./data")
    return candidates[0]


def parse_can_message(raw_message: str) -> Tuple[str, str, int, List[int], str]:
    match = CAN_MESSAGE_RE.match(raw_message.strip())
    if not match:
        raise ValueError(f"Unrecognized CAN_Message format: {raw_message!r}")

    interface = match.group("iface")
    can_id_int = int(match.group("can_id"), 16)
    can_id = f"{can_id_int:03X}"
    dlc = int(match.group("dlc"))
    data_field = (match.group("data") or "").strip()
    data_tokens = data_field.split() if data_field else []

    parse_warning = ""
    try:
        data = [int(token, 16) for token in data_tokens]
    except ValueError as exc:
        raise ValueError(f"Invalid byte in payload: {raw_message!r}") from exc

    if len(data) != dlc:
        parse_warning = f"dlc_mismatch: dlc={dlc}, payload_bytes={len(data)}"

    return interface, can_id, dlc, data, parse_warning


def parse_csv(path: Path, max_rows: Optional[int] = None) -> List[ParsedFrame]:
    frames: List[ParsedFrame] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break

            timestamp = row.get("Timestamp", "").strip()
            elapsed_raw = row.get("Elapsed_Time_s", "").strip()
            raw_message = row.get("CAN_Message", "").strip()
            if not raw_message:
                continue

            try:
                elapsed_s = float(elapsed_raw)
            except ValueError:
                elapsed_s = math.nan

            interface = ""
            can_id = "000"
            can_id_int = 0
            dlc = 0
            data: List[int] = []
            warning = ""
            try:
                interface, can_id, dlc, data, warning = parse_can_message(raw_message)
                can_id_int = int(can_id, 16)
            except ValueError as exc:
                warning = str(exc)

            frames.append(
                ParsedFrame(
                    timestamp=timestamp,
                    elapsed_s=elapsed_s,
                    interface=interface,
                    can_id=can_id,
                    can_id_int=can_id_int,
                    frame_name=FRAME_NAMES.get(can_id, "UNKNOWN"),
                    dlc=dlc,
                    data=data,
                    raw_message=raw_message,
                    parse_warning=warning,
                )
            )

    return frames


def add_signal(
    decoded_rows: List[Dict[str, object]],
    frame: ParsedFrame,
    signal: str,
    value: Optional[float],
    unit: str,
) -> None:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return
    decoded_rows.append(
        {
            "timestamp": frame.timestamp,
            "elapsed_s": frame.elapsed_s,
            "can_id": frame.can_id,
            "frame_name": frame.frame_name,
            "signal": signal,
            "value": value,
            "unit": unit,
        }
    )


def decode_frame(frame: ParsedFrame) -> List[Dict[str, object]]:
    d = frame.data
    cid = frame.can_id
    out: List[Dict[str, object]] = []

    if cid == "001":
        if len(d) >= 4:
            target_raw = le_u32(d, 0)
            add_signal(out, frame, "target_raw", float(target_raw), "raw")
        if len(d) >= 5:
            status = d[4]
            steering_selection = (status >> 7) & 0x01
            steering_enable = (status >> 6) & 0x01
            add_signal(out, frame, "steering_selection_bit", float(steering_selection), "flag")
            add_signal(out, frame, "steering_enable_bit", float(steering_enable), "flag")
            if len(d) >= 4:
                target = le_u32(d, 0) / 1000.0
                if steering_selection == 0:
                    add_signal(out, frame, "desired_heading_deg", target, "deg")
                else:
                    add_signal(out, frame, "manual_rudder_deg", target - 90.0, "deg")

    elif cid == "002":
        if len(d) >= 4:
            trim_tab_deg = (le_u32(d, 0) / 1000.0) - 90.0
            add_signal(out, frame, "trim_tab_angle_deg", trim_tab_deg, "deg")

    elif cid in {"040", "041"}:
        if len(d) >= 2:
            add_signal(out, frame, "wind_angle_deg", float(le_u16(d, 0)), "deg")
        if len(d) >= 4:
            add_signal(out, frame, "wind_speed_knots", le_u16(d, 2) / 10.0, "knots")

    elif cid == "060":
        if len(d) >= 4:
            add_signal(out, frame, "mmsi", float(le_u32(d, 0)), "id")
        if len(d) >= 8:
            add_signal(out, frame, "latitude_deg", (le_u32(d, 4) / 1_000_000.0) - 90.0, "deg")
        if len(d) >= 12:
            add_signal(out, frame, "longitude_deg", (le_u32(d, 8) / 1_000_000.0) - 180.0, "deg")
        if len(d) >= 14:
            sog_raw = le_u16(d, 12)
            add_signal(out, frame, "sog_knots", None if sog_raw == 1023 else sog_raw / 10.0, "knots")
        if len(d) >= 16:
            cog_raw = le_u16(d, 14)
            add_signal(out, frame, "cog_deg", None if cog_raw == 3600 else cog_raw / 10.0, "deg")
        if len(d) >= 18:
            heading_raw = le_u16(d, 16)
            add_signal(out, frame, "true_heading_deg", None if heading_raw == 511 else float(heading_raw), "deg")
        if len(d) >= 19:
            rot = d[18] - 128
            add_signal(out, frame, "rot_raw", None if rot == -128 else float(rot), "rot_units")
        if len(d) >= 21:
            add_signal(out, frame, "ship_length_m", float(le_u16(d, 19)), "m")
        if len(d) >= 23:
            add_signal(out, frame, "ship_width_m", float(le_u16(d, 21)), "m")
        if len(d) >= 24:
            add_signal(out, frame, "ship_idx", float(d[23]), "index")
        if len(d) >= 25:
            add_signal(out, frame, "total_ships", float(d[24]), "count")

    elif cid == "070":
        if len(d) >= 4:
            add_signal(out, frame, "latitude_deg", (le_u32(d, 0) / 1_000_000.0) - 90.0, "deg")
        if len(d) >= 8:
            add_signal(out, frame, "longitude_deg", (le_u32(d, 4) / 1_000_000.0) - 180.0, "deg")
        if len(d) >= 12:
            add_signal(out, frame, "utc_seconds", le_u32(d, 8) / 1000.0, "s")
        if len(d) >= 13:
            add_signal(out, frame, "utc_minutes", float(d[12]), "min")
        if len(d) >= 14:
            add_signal(out, frame, "utc_hours", float(d[13]), "h")
        if len(d) >= 20:
            add_signal(out, frame, "speed_over_ground_kmh", le_u32(d, 16) / 1000.0, "km/h")

    elif cid == "100":
        if len(d) >= 4:
            temp_k = le_u32(d, 0) / 1000.0
            add_signal(out, frame, "temperature_k", temp_k, "K")
            add_signal(out, frame, "temperature_c", temp_k - 273.15, "C")
        elif len(d) == 3:
            temp_k = le_u24(d, 0) / 1000.0
            add_signal(out, frame, "temperature_k", temp_k, "K")
            add_signal(out, frame, "temperature_c", temp_k - 273.15, "C")

    elif cid == "110":
        if len(d) >= 2:
            add_signal(out, frame, "ph", le_u16(d, 0) / 1000.0, "pH")

    elif cid == "120":
        if len(d) >= 4:
            add_signal(out, frame, "conductivity_us_cm", le_u32(d, 0) / 1000.0, "uS/cm")

    elif cid in {"130", "131", "132", "133"}:
        if len(d) == 0:
            add_signal(out, frame, "heartbeat", 1.0, "flag")

    elif cid == "204":
        if len(d) >= 2:
            add_signal(out, frame, "actual_rudder_deg", (le_u16(d, 0) / 100.0) - 90.0, "deg")
        if len(d) >= 4:
            # User-confirmed correction: bytes [31:16] are pitch (not roll).
            add_signal(out, frame, "imu_pitch_deg", (le_u16(d, 2) / 100.0) - 180.0, "deg")
        if len(d) >= 6:
            # User-confirmed correction: bytes [47:32] are roll (not pitch).
            add_signal(out, frame, "imu_roll_deg", (le_u16(d, 4) / 100.0) - 180.0, "deg")
        if len(d) >= 8:
            add_signal(out, frame, "imu_heading_deg", le_u16(d, 6) / 100.0, "deg")
        if len(d) >= 10:
            add_signal(out, frame, "commanded_rudder_deg", (le_u16(d, 8) / 100.0) - 90.0, "deg")
        if len(d) >= 12:
            add_signal(out, frame, "rudder_integral_raw", float(le_u16(d, 10)), "raw")
        if len(d) >= 14:
            add_signal(out, frame, "rudder_derivative_raw", float(le_u16(d, 12)), "raw")
        if len(d) >= 16:
            add_signal(out, frame, "speed_over_ground_kmh", le_u16(d, 14) / 1000.0, "km/h")

    elif cid == "206":
        if len(d) >= 2:
            add_signal(out, frame, "cell_voltage_2_v", le_u16(d, 0) / 1000.0, "V")
        if len(d) >= 4:
            add_signal(out, frame, "temp_pack2_port_c", le_u16(d, 2) / 100.0, "C")
        if len(d) >= 6:
            add_signal(out, frame, "cell_voltage_3_v", le_u16(d, 4) / 1000.0, "V")
        if len(d) >= 8:
            add_signal(out, frame, "temp_buck_boost_c", le_u16(d, 6) / 100.0, "C")
        if len(d) >= 10:
            add_signal(out, frame, "temp_pack1_starboard_c", le_u16(d, 8) / 100.0, "C")
        if len(d) >= 12:
            add_signal(out, frame, "cell_voltage_4_v", le_u16(d, 10) / 1000.0, "V")
        if len(d) >= 14:
            add_signal(out, frame, "cell_voltage_1_v", le_u16(d, 12) / 1000.0, "V")
        if len(d) >= 16:
            add_signal(out, frame, "mppt_hull_port_a", le_u16(d, 14) / 1000.0, "A")
        if len(d) >= 18:
            add_signal(out, frame, "mppt_hull_starboard_a", le_u16(d, 16) / 1000.0, "A")
        if len(d) >= 20:
            add_signal(out, frame, "mppt_sail_port_a", le_u16(d, 18) / 1000.0, "A")
        if len(d) >= 22:
            add_signal(out, frame, "mppt_sail_starboard_a", le_u16(d, 20) / 1000.0, "A")
        if len(d) >= 24:
            add_signal(out, frame, "padding_raw", float(le_u16(d, 22)), "raw")

    return out


def decode_frames(frames: Iterable[ParsedFrame]) -> List[Dict[str, object]]:
    decoded_rows: List[Dict[str, object]] = []
    for frame in frames:
        decoded_rows.extend(decode_frame(frame))
    return decoded_rows


def write_parsed_frames_csv(frames: Iterable[ParsedFrame], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "timestamp",
        "elapsed_s",
        "interface",
        "can_id",
        "can_id_int",
        "frame_name",
        "dlc",
        "data_len",
        "data_hex",
        "parse_warning",
        "raw_message",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for frame in frames:
            writer.writerow(
                {
                    "timestamp": frame.timestamp,
                    "elapsed_s": frame.elapsed_s,
                    "interface": frame.interface,
                    "can_id": frame.can_id,
                    "can_id_int": frame.can_id_int,
                    "frame_name": frame.frame_name,
                    "dlc": frame.dlc,
                    "data_len": len(frame.data),
                    "data_hex": " ".join(f"{b:02X}" for b in frame.data),
                    "parse_warning": frame.parse_warning,
                    "raw_message": frame.raw_message,
                }
            )


def write_decoded_signals_csv(decoded_rows: Iterable[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["timestamp", "elapsed_s", "can_id", "frame_name", "signal", "value", "unit"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in decoded_rows:
            writer.writerow(row)


def signal_series(
    decoded_rows: Iterable[Dict[str, object]], signal: str, can_id: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    x: List[float] = []
    y: List[float] = []
    for row in decoded_rows:
        if row["signal"] != signal:
            continue
        if can_id is not None and row["can_id"] != can_id:
            continue
        try:
            xv = float(row["elapsed_s"])
            yv = float(row["value"])
        except (TypeError, ValueError):
            continue
        if math.isfinite(xv) and math.isfinite(yv):
            x.append(xv)
            y.append(yv)

    if not x:
        return np.array([]), np.array([])

    xs = np.array(x)
    ys = np.array(y)
    order = np.argsort(xs)
    return xs[order], ys[order]


def robust_rolling_mean(ys: np.ndarray, window: int = 5, outlier_sigma: float = 3.0) -> np.ndarray:
    """Return a rolling-mean trace with local outlier rejection.

    Outliers are identified with a Hampel-style rule using local median and MAD.
    Then a rolling mean is computed over inlier points (NaNs ignored).
    """
    n = len(ys)
    if n == 0:
        return ys

    window = max(1, int(window))
    half = window // 2
    cleaned = ys.astype(float).copy()

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        local = ys[lo:hi]
        if local.size < 3:
            continue

        median = float(np.median(local))
        mad = float(np.median(np.abs(local - median)))
        scale = 1.4826 * mad
        if scale <= 1e-12:
            std = float(np.std(local))
            scale = std if std > 1e-12 else 0.0

        if scale > 0.0 and abs(float(ys[i]) - median) > outlier_sigma * scale:
            cleaned[i] = np.nan

    filtered = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        local = cleaned[lo:hi]
        local = local[np.isfinite(local)]
        if local.size == 0:
            filtered[i] = filtered[i - 1] if i > 0 else float(ys[i])
        else:
            filtered[i] = float(np.mean(local))

    return filtered


def estimate_canfd_frame_time_s(
    payload_bytes: int,
    nominal_bitrate_bps: float = CAN_NOMINAL_BITRATE_BPS,
    data_bitrate_bps: float = CAN_DATA_BITRATE_BPS,
    stuffing_factor: float = CANFD_STUFFING_FACTOR,
) -> float:
    """Estimate on-bus CAN FD frame duration in seconds.

    Uses a standard 11-bit ID CAN FD frame model with bit-rate switching (BRS)
    and an average stuffing overhead factor.
    """
    n = max(0, int(payload_bytes))

    # Approximate base bits by phase (11-bit CAN FD, BRS enabled).
    # Nominal-rate phase includes arbitration + tail bits.
    nominal_bits = 31  # SOF, ID/control (to BRS), CRC delim, ACK, EOF, IFS
    crc_bits = 17 if n <= 16 else 21
    data_bits = 1 + 4 + (8 * n) + crc_bits  # ESI + DLC + payload + CRC sequence

    nominal_time_s = (nominal_bits * stuffing_factor) / nominal_bitrate_bps
    data_time_s = (data_bits * stuffing_factor) / data_bitrate_bps
    return nominal_time_s + data_time_s


def estimate_can_utilization_series(
    frames: List[ParsedFrame],
    window_s: float = UTILIZATION_WINDOW_S,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate CAN utilization (%) in fixed time windows."""
    ts: List[float] = []
    dur: List[float] = []

    for frame in frames:
        if frame.can_id == "000" or not math.isfinite(frame.elapsed_s):
            continue
        ts.append(float(frame.elapsed_s))
        dur.append(estimate_canfd_frame_time_s(frame.dlc))

    if not ts:
        return np.array([]), np.array([])

    t = np.array(ts, dtype=float)
    d = np.array(dur, dtype=float)
    order = np.argsort(t)
    t = t[order]
    d = d[order]

    start = float(np.min(t))
    end = float(np.max(t))
    if end <= start:
        return np.array([start]), np.array([0.0])

    edges = np.arange(start, end + window_s, window_s, dtype=float)
    if edges.size < 2:
        edges = np.array([start, start + window_s], dtype=float)

    occupancy_s, _ = np.histogram(t, bins=edges, weights=d)
    centers = edges[:-1] + (window_s / 2.0)
    utilization_pct = (occupancy_s / window_s) * 100.0
    return centers, utilization_pct


def trailing_rolling_max(values: np.ndarray, window_points: int) -> np.ndarray:
    """Trailing rolling max with fixed sample window length."""
    n = len(values)
    if n == 0:
        return values
    w = max(1, int(window_points))
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - w + 1)
        out[i] = float(np.max(values[lo : i + 1]))
    return out


def extract_gps_points(frames: Iterable[ParsedFrame]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lons: List[float] = []
    lats: List[float] = []
    speeds: List[float] = []
    for frame in frames:
        if frame.can_id != "070":
            continue
        d = frame.data
        if len(d) < 8:
            continue
        lat = (le_u32(d, 0) / 1_000_000.0) - 90.0
        lon = (le_u32(d, 4) / 1_000_000.0) - 180.0
        speed = (le_u32(d, 16) / 1000.0) if len(d) >= 20 else np.nan
        if math.isfinite(lat) and math.isfinite(lon):
            lats.append(lat)
            lons.append(lon)
            speeds.append(float(speed))
    return np.array(lons), np.array(lats), np.array(speeds)


def extract_ais_points(frames: Iterable[ParsedFrame]) -> Tuple[np.ndarray, np.ndarray]:
    lons: List[float] = []
    lats: List[float] = []
    for frame in frames:
        if frame.can_id != "060":
            continue
        d = frame.data
        if len(d) < 12:
            continue
        lat = (le_u32(d, 4) / 1_000_000.0) - 90.0
        lon = (le_u32(d, 8) / 1_000_000.0) - 180.0
        if math.isfinite(lat) and math.isfinite(lon):
            lats.append(lat)
            lons.append(lon)
    return np.array(lons), np.array(lats)


def project_to_local_km(
    lon_deg: np.ndarray, lat_deg: np.ndarray, origin_lon_deg: float, origin_lat_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Project lon/lat to local tangent-plane x/y in kilometers."""
    lon = np.asarray(lon_deg, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)
    origin_lon = float(origin_lon_deg)
    origin_lat = float(origin_lat_deg)
    lat0_rad = np.deg2rad(origin_lat)

    x_km = EARTH_RADIUS_KM * np.deg2rad(lon - origin_lon) * np.cos(lat0_rad)
    y_km = EARTH_RADIUS_KM * np.deg2rad(lat - origin_lat)
    return x_km, y_km


def filter_ais_near_gps(
    ais_lon: np.ndarray,
    ais_lat: np.ndarray,
    gps_ref_lon: float,
    gps_ref_lat: float,
    half_range_deg: float = AIS_NEARBY_THRESHOLD_DEG,
) -> Tuple[np.ndarray, np.ndarray]:
    if ais_lon.size == 0 or ais_lat.size == 0:
        return ais_lon, ais_lat
    mask = (np.abs(ais_lon - gps_ref_lon) <= half_range_deg) & (
        np.abs(ais_lat - gps_ref_lat) <= half_range_deg
    )
    return ais_lon[mask], ais_lat[mask]


def style_axis(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor("#111827")
    ax.set_title(title, color="#E5E7EB", fontsize=11, pad=8)
    ax.grid(color="#374151", alpha=0.35, linewidth=0.8)
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


def draw_frame_counts_panel(
    fig: plt.Figure, ax: plt.Axes, frames: List[ParsedFrame], decoded_rows: List[Dict[str, object]]
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
    y_top = max_val * 1.10 if max_val > 0 else 1.0
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
    fig: plt.Figure, ax: plt.Axes, frames: List[ParsedFrame], decoded_rows: List[Dict[str, object]]
) -> None:
    del fig, frames
    style_axis(ax, "Rudder Actual vs Commanded")
    x_actual, y_actual = signal_series(decoded_rows, "actual_rudder_deg", "204")
    x_cmd, y_cmd = signal_series(decoded_rows, "commanded_rudder_deg", "204")
    if len(x_actual) > 0:
        ax.plot(x_actual, y_actual, color="#34D399", linewidth=1.5, label="Actual")
    if len(x_cmd) > 0:
        ax.plot(x_cmd, y_cmd, color="#F97316", linewidth=1.5, alpha=0.9, label="Commanded")
    if len(x_actual) == 0 and len(x_cmd) == 0:
        annotate_no_data(ax)
        return
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Rudder Angle (deg)")
    ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#D1D5DB")


def draw_imu_panel(
    fig: plt.Figure, ax: plt.Axes, frames: List[ParsedFrame], decoded_rows: List[Dict[str, object]]
) -> None:
    del fig, frames
    style_axis(ax, "IMU Roll, Pitch, and Heading")

    x_roll, y_roll = signal_series(decoded_rows, "imu_roll_deg", "204")
    x_pitch, y_pitch = signal_series(decoded_rows, "imu_pitch_deg", "204")
    x_heading, y_heading = signal_series(decoded_rows, "imu_heading_deg", "204")

    if len(x_roll) > 0:
        ax.plot(x_roll, y_roll, color="#60A5FA", linewidth=1.4, label="Roll")
    if len(x_pitch) > 0:
        ax.plot(x_pitch, y_pitch, color="#FBBF24", linewidth=1.4, label="Pitch")

    ax_heading = ax.twinx()
    style_twin_y_axis(ax_heading)
    if len(x_heading) > 0:
        ax_heading.plot(x_heading, y_heading, color="#F472B6", linewidth=1.2, alpha=0.9, label="Heading")
        ax_heading.set_ylabel("Heading (deg)")

    if len(x_roll) == 0 and len(x_pitch) == 0 and len(x_heading) == 0:
        annotate_no_data(ax)
        ax_heading.set_yticks([])
        return

    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Roll/Pitch (deg)")
    lines = ax.get_lines() + ax_heading.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, facecolor="#111827", edgecolor="#374151", labelcolor="#D1D5DB")


def draw_wind_angle_split_panel(
    fig: plt.Figure, ax: plt.Axes, frames: List[ParsedFrame], decoded_rows: List[Dict[str, object]]
) -> None:
    del fig, frames
    style_axis(ax, "Wind Angle (0x040 vs 0x041)")

    x_040, y_040 = signal_series(decoded_rows, "wind_angle_deg", "040")
    x_041, y_041 = signal_series(decoded_rows, "wind_angle_deg", "041")

    has_data = False
    if len(x_040) > 0:
        ax.plot(x_040, y_040, color="#22D3EE", linewidth=1.4, label="0x040 Angle")
        has_data = True
    if len(x_041) > 0:
        ax.plot(x_041, y_041, color="#F59E0B", linewidth=1.4, label="0x041 Angle")
        has_data = True

    if not has_data:
        annotate_no_data(ax)
        return

    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Wind Angle (deg)")
    ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#D1D5DB", loc="upper right")


def draw_wind_speed_split_panel(
    fig: plt.Figure, ax: plt.Axes, frames: List[ParsedFrame], decoded_rows: List[Dict[str, object]]
) -> None:
    del fig, frames
    style_axis(ax, "Wind Speed (0x040 vs 0x041)")

    x_040, y_040 = signal_series(decoded_rows, "wind_speed_knots", "040")
    x_041, y_041 = signal_series(decoded_rows, "wind_speed_knots", "041")

    has_data = False
    if len(x_040) > 0:
        ax.plot(x_040, y_040, color="#22D3EE", linewidth=1.4, label="0x040 Speed")
        has_data = True
    if len(x_041) > 0:
        ax.plot(x_041, y_041, color="#F59E0B", linewidth=1.4, label="0x041 Speed")
        has_data = True

    if not has_data:
        annotate_no_data(ax)
        return

    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Wind Speed (knots)")
    ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#D1D5DB", loc="upper right")


def draw_geo_panel(
    fig: plt.Figure, ax: plt.Axes, frames: List[ParsedFrame], decoded_rows: List[Dict[str, object]]
) -> None:
    del decoded_rows
    style_axis(ax, "GPS Track + AIS Contacts (km)")
    gps_lon, gps_lat, gps_speed = extract_gps_points(frames)
    ais_lon, ais_lat = extract_ais_points(frames)

    # Filter to nearby AIS contacts around the GPS position to avoid distant outliers.
    if len(gps_lon) > 0:
        gps_ref_lon = float(gps_lon[-1])
        gps_ref_lat = float(gps_lat[-1])
        ais_lon, ais_lat = filter_ais_near_gps(ais_lon, ais_lat, gps_ref_lon, gps_ref_lat)

    if len(gps_lon) > 0 and len(ais_lon) > 0:
        origin_lon = float(np.mean(np.concatenate([gps_lon, ais_lon])))
        origin_lat = float(np.mean(np.concatenate([gps_lat, ais_lat])))
    elif len(gps_lon) > 0:
        origin_lon = float(np.mean(gps_lon))
        origin_lat = float(np.mean(gps_lat))
    elif len(ais_lon) > 0:
        origin_lon = float(np.mean(ais_lon))
        origin_lat = float(np.mean(ais_lat))
    else:
        origin_lon = 0.0
        origin_lat = 0.0

    gps_x_km = np.array([])
    gps_y_km = np.array([])
    ais_x_km = np.array([])
    ais_y_km = np.array([])

    if len(gps_lon) > 0:
        gps_x_km, gps_y_km = project_to_local_km(gps_lon, gps_lat, origin_lon, origin_lat)
        ax.plot(gps_x_km, gps_y_km, color="#93C5FD", linewidth=1.0, alpha=0.55)
        scatter = ax.scatter(
            gps_x_km,
            gps_y_km,
            c=gps_speed,
            cmap="turbo",
            s=15,
            alpha=0.9,
            edgecolors="none",
            label="GPS path",
        )
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(color="#D1D5DB", labelcolor="#D1D5DB", labelsize=8)
        cbar.set_label("Speed (km/h)", color="#D1D5DB", fontsize=9)
        cbar.outline.set_edgecolor("#374151")

    if len(ais_lon) > 0:
        ais_x_km, ais_y_km = project_to_local_km(ais_lon, ais_lat, origin_lon, origin_lat)
        ax.scatter(
            ais_x_km,
            ais_y_km,
            color="#F472B6",
            marker="x",
            s=18,
            linewidths=1.0,
            alpha=0.75,
            label="AIS contacts",
        )

    if len(gps_lon) == 0 and len(ais_lon) == 0:
        annotate_no_data(ax)
        return

    all_x = np.concatenate([arr for arr in [gps_x_km, ais_x_km] if arr.size > 0])
    all_y = np.concatenate([arr for arr in [gps_y_km, ais_y_km] if arr.size > 0])
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

    # Add explicit horizontal padding to avoid a narrow-looking geo panel.
    extra_x = 0.5 * span_y * GEO_HORIZONTAL_PAD_FRAC
    x_lo -= extra_x
    x_hi += extra_x

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal", adjustable="datalim")
    apply_distance_axis_units(ax, x_lo, x_hi, y_lo, y_hi)
    ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#D1D5DB")


def draw_geo_gps_scaled_panel(
    fig: plt.Figure, ax: plt.Axes, frames: List[ParsedFrame], decoded_rows: List[Dict[str, object]]
) -> None:
    del decoded_rows
    style_axis(ax, "GPS Track + AIS (GPS-Scaled, km)")
    gps_lon, gps_lat, gps_speed = extract_gps_points(frames)
    ais_lon, ais_lat = extract_ais_points(frames)

    # Filter to nearby AIS contacts around the GPS position to avoid distant outliers.
    if len(gps_lon) > 0:
        gps_ref_lon = float(gps_lon[-1])
        gps_ref_lat = float(gps_lat[-1])
        ais_lon, ais_lat = filter_ais_near_gps(ais_lon, ais_lat, gps_ref_lon, gps_ref_lat)

    if len(gps_lon) > 0 and len(ais_lon) > 0:
        origin_lon = float(np.mean(np.concatenate([gps_lon, ais_lon])))
        origin_lat = float(np.mean(np.concatenate([gps_lat, ais_lat])))
    elif len(gps_lon) > 0:
        origin_lon = float(np.mean(gps_lon))
        origin_lat = float(np.mean(gps_lat))
    elif len(ais_lon) > 0:
        origin_lon = float(np.mean(ais_lon))
        origin_lat = float(np.mean(ais_lat))
    else:
        origin_lon = 0.0
        origin_lat = 0.0

    gps_x_km = np.array([])
    gps_y_km = np.array([])
    ais_x_km = np.array([])
    ais_y_km = np.array([])

    if len(gps_lon) > 0:
        gps_x_km, gps_y_km = project_to_local_km(gps_lon, gps_lat, origin_lon, origin_lat)
        ax.plot(gps_x_km, gps_y_km, color="#93C5FD", linewidth=1.0, alpha=0.55)
        scatter = ax.scatter(
            gps_x_km,
            gps_y_km,
            c=gps_speed,
            cmap="turbo",
            s=15,
            alpha=0.9,
            edgecolors="none",
            label="GPS path",
        )
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(color="#D1D5DB", labelcolor="#D1D5DB", labelsize=8)
        cbar.set_label("Speed (km/h)", color="#D1D5DB", fontsize=9)
        cbar.outline.set_edgecolor("#374151")

    if len(ais_lon) > 0:
        ais_x_km, ais_y_km = project_to_local_km(ais_lon, ais_lat, origin_lon, origin_lat)
        ax.scatter(
            ais_x_km,
            ais_y_km,
            color="#F472B6",
            marker="x",
            s=18,
            linewidths=1.0,
            alpha=0.75,
            label="AIS contacts",
        )

    if len(gps_lon) == 0 and len(ais_lon) == 0:
        annotate_no_data(ax)
        return

    # Scale bounds to GPS track size (with margin). If GPS is unavailable,
    # fall back to all available points.
    base_x = gps_x_km if gps_x_km.size > 0 else np.concatenate([arr for arr in [gps_x_km, ais_x_km] if arr.size > 0])
    base_y = gps_y_km if gps_y_km.size > 0 else np.concatenate([arr for arr in [gps_y_km, ais_y_km] if arr.size > 0])
    x_min = float(np.min(base_x))
    x_max = float(np.max(base_x))
    y_min = float(np.min(base_y))
    y_max = float(np.max(base_y))

    span_x = max(x_max - x_min, 1e-6)
    span_y = max(y_max - y_min, 1e-6)
    x_pad = 0.5 * span_x * GEO_MARGIN_FRAC
    y_pad = 0.5 * span_y * GEO_MARGIN_FRAC

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("equal", adjustable="datalim")
    apply_distance_axis_units(ax, x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)
    ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#D1D5DB")


def draw_pdb_voltages_panel(
    fig: plt.Figure, ax: plt.Axes, frames: List[ParsedFrame], decoded_rows: List[Dict[str, object]]
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
        ax.scatter(xs, ys, color=color, s=10, alpha=0.1, edgecolors="none", label="_nolegend_")
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
    fig: plt.Figure, ax: plt.Axes, frames: List[ParsedFrame], decoded_rows: List[Dict[str, object]]
) -> None:
    del fig, decoded_rows
    style_axis(ax, "Estimated CAN FD Utilization (Raw + 5s Max)")
    series_specs = [
        (0.02, "#22EE29"),
        (0.1, "#38BDF8"),
        (0.5, "#8B5CF6")
    ]

    has_data = False
    y_max = 0.0
    for window_s, color in series_specs:
        x, y = estimate_can_utilization_series(frames, window_s=window_s)
        if len(x) == 0:
            continue
        # Raw utilization points for this binning.
        ax.scatter(x, y, color=color, s=8, alpha=0.10, edgecolors="none", label="_nolegend_")

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
    ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#D1D5DB", loc="upper right")


def draw_battery_temps_panel(
    fig: plt.Figure, ax: plt.Axes, frames: List[ParsedFrame], decoded_rows: List[Dict[str, object]]
) -> None:
    del fig, frames
    style_axis(ax, "Battery Pack Temperatures")
    x_t1, y_t1 = signal_series(decoded_rows, "temp_pack2_port_c", "206")
    x_t3, y_t3 = signal_series(decoded_rows, "temp_pack1_starboard_c", "206")

    has_data = False
    if len(x_t1) > 0:
        y_t1_filtered = robust_rolling_mean(y_t1, window=5, outlier_sigma=3.0)
        ax.scatter(x_t1, y_t1, color="#FB7185", s=10, alpha=0.1, edgecolors="none", label="_nolegend_")
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
        ax.scatter(x_t3, y_t3, color="#22D3EE", s=10, alpha=0.1, edgecolors="none", label="_nolegend_")
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
    ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#D1D5DB", loc="upper right")


def draw_sensor_temp_panel(
    fig: plt.Figure, ax: plt.Axes, frames: List[ParsedFrame], decoded_rows: List[Dict[str, object]]
) -> None:
    del fig, frames
    style_axis(ax, "Data Sensor: Temperature")
    x_temp, y_temp = signal_series(decoded_rows, "temperature_c", "100")
    if len(x_temp) == 0:
        annotate_no_data(ax)
        return
    ax.plot(x_temp, y_temp, color="#F59E0B", linewidth=1.5, label="Temperature (C)")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Temperature (C)")
    ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#D1D5DB", loc="upper right")


def draw_sensor_ph_panel(
    fig: plt.Figure, ax: plt.Axes, frames: List[ParsedFrame], decoded_rows: List[Dict[str, object]]
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
    ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#D1D5DB", loc="upper right")


def draw_sensor_cond_panel(
    fig: plt.Figure, ax: plt.Axes, frames: List[ParsedFrame], decoded_rows: List[Dict[str, object]]
) -> None:
    del fig, frames
    style_axis(ax, "Data Sensor: Conductivity")
    x_cond, y_cond = signal_series(decoded_rows, "conductivity_us_cm", "120")
    if len(x_cond) == 0:
        annotate_no_data(ax)
        return

    ax.plot(x_cond, y_cond, color="#34D399", linewidth=1.5, label="Conductivity (uS/cm)")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Conductivity (uS/cm)")
    ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#D1D5DB", loc="upper right")


PANEL_DRAWERS = {
    "frame_counts": draw_frame_counts_panel,
    "can_utilization": draw_can_utilization_panel,
    "rudder": draw_rudder_panel,
    "imu": draw_imu_panel,
    "wind_angle_split": draw_wind_angle_split_panel,
    "wind_speed_split": draw_wind_speed_split_panel,
    "geo": draw_geo_panel,
    "geo_gps_scaled": draw_geo_gps_scaled_panel,
    "pdb_voltages": draw_pdb_voltages_panel,
    "battery_temps": draw_battery_temps_panel,
    "sensor_temp": draw_sensor_temp_panel,
    "sensor_ph": draw_sensor_ph_panel,
    "sensor_cond": draw_sensor_cond_panel,
}


def create_dashboard(
    frames: List[ParsedFrame],
    decoded_rows: List[Dict[str, object]],
    output_path: Path,
    source_name: str,
    title: str,
    panels: List[str],
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
    fig_height_in = max(5.2, (row_height_in * nrows) + header_height_in + footer_height_in)

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
    subtitle = (
        f"Source: {source_name} | Frames: {len(frames):,} | Decoded signals: {len(decoded_rows):,}"
    )
    fig.text(0.5, subtitle_y, subtitle, ha="center", va="center", color="#CBD5E1", fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input if args.input is not None else default_input_csv()
    frames = parse_csv(input_csv, max_rows=args.max_rows)
    decoded_rows = decode_frames(frames)

    parsed_frames_path = args.outdir / "parsed_frames.csv"
    decoded_signals_path = args.outdir / "decoded_signals.csv"
    write_parsed_frames_csv(frames, parsed_frames_path)
    write_decoded_signals_csv(decoded_rows, decoded_signals_path)

    dashboard_paths: List[Path] = []
    if not args.skip_plot:
        for output_name, cfg in DASHBOARD_CONFIG.items():
            title = str(cfg.get("title", "POLARIS CAN Dashboard"))
            panels_raw = cfg.get("panels", [])
            panels = [str(item) for item in panels_raw] if isinstance(panels_raw, list) else []
            output_path = args.outdir / output_name
            create_dashboard(
                frames,
                decoded_rows,
                output_path,
                source_name=input_csv.name,
                title=title,
                panels=panels,
            )
            dashboard_paths.append(output_path)

    malformed = sum(1 for frame in frames if frame.parse_warning)
    print(f"Input file: {input_csv}")
    print(f"Total frames: {len(frames):,}")
    print(f"Frames with parse warnings: {malformed:,}")
    print(f"Decoded signal rows: {len(decoded_rows):,}")
    print(f"Wrote: {parsed_frames_path}")
    print(f"Wrote: {decoded_signals_path}")
    if args.skip_plot:
        print("Skipped dashboard plot (--skip-plot)")
    else:
        for dashboard_path in dashboard_paths:
            print(f"Wrote: {dashboard_path}")


if __name__ == "__main__":
    main()
