from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from polaris_can_analysis.config import CAN_MESSAGE_RE, FRAME_NAMES
from polaris_can_analysis.models import ParsedFrame


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

