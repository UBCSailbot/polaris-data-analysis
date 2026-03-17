from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from polaris_can_analysis.config import (
    AIS_NEARBY_THRESHOLD_DEG,
    CAN_DATA_BITRATE_BPS,
    CAN_NOMINAL_BITRATE_BPS,
    CANFD_STUFFING_FACTOR,
    CONDUCTIVITY_FILTER_WINDOW,
    EARTH_RADIUS_KM,
    ON_WATER_MIN_CONSECUTIVE,
    ON_WATER_STEADY_TOP_QUANTILE,
    ON_WATER_THRESHOLD_FRAC,
    UTILIZATION_WINDOW_S,
)
from polaris_can_analysis.models import OnWaterDetection, ParsedFrame
from polaris_can_analysis.processing import le_u32, signal_series


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


def conductivity_series(
    decoded_rows: Iterable[Dict[str, object]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return conductivity raw + filtered series."""
    x_cond, y_cond = signal_series(decoded_rows, "conductivity_us_cm", "120")
    if len(x_cond) == 0:
        return x_cond, y_cond, y_cond
    y_filtered = robust_rolling_mean(y_cond, window=CONDUCTIVITY_FILTER_WINDOW, outlier_sigma=3.0)
    return x_cond, y_cond, y_filtered


def detect_on_water_start(decoded_rows: Iterable[Dict[str, object]]) -> OnWaterDetection:
    """Detect launch/on-water start from filtered conductivity."""
    x_cond, _, y_filt = conductivity_series(decoded_rows)
    if len(x_cond) < ON_WATER_MIN_CONSECUTIVE:
        return OnWaterDetection(start_s=None, steady_state_us_cm=None, threshold_us_cm=None)

    finite = np.isfinite(x_cond) & np.isfinite(y_filt)
    x_use = x_cond[finite]
    y_use = y_filt[finite]
    if len(x_use) < ON_WATER_MIN_CONSECUTIVE:
        return OnWaterDetection(start_s=None, steady_state_us_cm=None, threshold_us_cm=None)

    high_cut = float(np.quantile(y_use, ON_WATER_STEADY_TOP_QUANTILE))
    steady_candidates = y_use[y_use >= high_cut]
    if len(steady_candidates) == 0:
        return OnWaterDetection(start_s=None, steady_state_us_cm=None, threshold_us_cm=None)

    steady_state = float(np.median(steady_candidates))
    if steady_state <= 0.0:
        return OnWaterDetection(start_s=None, steady_state_us_cm=None, threshold_us_cm=None)

    threshold = steady_state * ON_WATER_THRESHOLD_FRAC
    run_len = ON_WATER_MIN_CONSECUTIVE

    for i in range(0, len(y_use) - run_len + 1):
        window = y_use[i : i + run_len]
        if np.all(np.isfinite(window)) and np.all(window >= threshold):
            return OnWaterDetection(
                start_s=float(x_use[i]),
                steady_state_us_cm=steady_state,
                threshold_us_cm=threshold,
            )

    return OnWaterDetection(start_s=None, steady_state_us_cm=steady_state, threshold_us_cm=threshold)


def filter_frames_by_start(frames: List[ParsedFrame], start_s: float) -> List[ParsedFrame]:
    return [frame for frame in frames if math.isfinite(frame.elapsed_s) and frame.elapsed_s >= start_s]


def filter_decoded_rows_by_start(
    decoded_rows: List[Dict[str, object]],
    start_s: float,
) -> List[Dict[str, object]]:
    filtered: List[Dict[str, object]] = []
    for row in decoded_rows:
        try:
            elapsed = float(row["elapsed_s"])
        except (TypeError, ValueError, KeyError):
            continue
        if math.isfinite(elapsed) and elapsed >= start_s:
            filtered.append(row)
    return filtered


def break_wrapped_angle_series(
    xs: np.ndarray,
    ys: np.ndarray,
    wrap_deg: float = 360.0,
    break_threshold_deg: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Insert NaN breaks so lines don't connect across angular wrap transitions."""
    if len(xs) <= 1:
        return xs, ys

    threshold = (wrap_deg * 0.5) if break_threshold_deg is None else float(break_threshold_deg)
    out_x: List[float] = [float(xs[0])]
    out_y: List[float] = [float(ys[0])]

    for i in range(1, len(xs)):
        prev = float(ys[i - 1])
        curr = float(ys[i])
        if math.isfinite(prev) and math.isfinite(curr) and abs(curr - prev) > threshold:
            out_x.append(float("nan"))
            out_y.append(float("nan"))
        out_x.append(float(xs[i]))
        out_y.append(curr)

    return np.array(out_x, dtype=float), np.array(out_y, dtype=float)


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


def extract_ais_tracks(frames: Iterable[ParsedFrame]) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    tracks: Dict[int, Dict[str, List[float]]] = {}
    for idx, frame in enumerate(frames):
        if frame.can_id != "060":
            continue
        d = frame.data
        if len(d) < 12:
            continue
        mmsi = int(le_u32(d, 0)) if len(d) >= 4 else -1
        lat = (le_u32(d, 4) / 1_000_000.0) - 90.0
        lon = (le_u32(d, 8) / 1_000_000.0) - 180.0
        t = float(frame.elapsed_s) if math.isfinite(frame.elapsed_s) else float(idx)
        if not math.isfinite(lat) or not math.isfinite(lon) or mmsi < 0:
            continue
        if mmsi not in tracks:
            tracks[mmsi] = {"lon": [], "lat": [], "t": []}
        tracks[mmsi]["lon"].append(lon)
        tracks[mmsi]["lat"].append(lat)
        tracks[mmsi]["t"].append(t)

    out: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for mmsi, data in tracks.items():
        lon_arr = np.array(data["lon"], dtype=float)
        lat_arr = np.array(data["lat"], dtype=float)
        t_arr = np.array(data["t"], dtype=float)
        order = np.argsort(t_arr)
        out[mmsi] = (lon_arr[order], lat_arr[order], t_arr[order])
    return out


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


def filter_ais_tracks_near_gps(
    ais_tracks: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    gps_ref_lon: float,
    gps_ref_lat: float,
    half_range_deg: float = AIS_NEARBY_THRESHOLD_DEG,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    filtered: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for mmsi, (lon, lat, t) in ais_tracks.items():
        mask = (np.abs(lon - gps_ref_lon) <= half_range_deg) & (np.abs(lat - gps_ref_lat) <= half_range_deg)
        if np.any(mask):
            filtered[mmsi] = (lon[mask], lat[mask], t[mask])
    return filtered
