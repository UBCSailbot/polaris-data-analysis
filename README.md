# POLARIS CAN Analysis Tool

This repo now includes a basic parser + visualization script for CAN dumps:

- Script: `analyze_can_frames.py`
- Input CSV format: `Timestamp,Elapsed_Time_s,CAN_Message`
- Example input file: `data/candump_20260315_153322.csv`

The decoding logic is based on:

- `prjt22-CAN Frames-170326-202810.pdf`
- `prjt22-Debug CAN Frames-170326-202858.pdf`

It is intentionally tolerant of partial implementations and shorter payloads that appear in real logs.

## Quick Start

Run with the default CSV in `data/`:

```bash
python3 analyze_can_frames.py
```

Run on a specific file:

```bash
python3 analyze_can_frames.py --input data/candump_20260315_153322.csv
```

Write outputs to a custom directory:

```bash
python3 analyze_can_frames.py --outdir outputs/run_01
```

## Outputs

By default the script writes to `outputs/`:

- `parsed_frames.csv`: one row per frame, including CAN ID, DLC, payload bytes, and parse warnings.
- `decoded_signals.csv`: one row per decoded signal value.
- `physical_dashboard.png`: physical/navigation focused panels.
- `electrical_dashboard.png`: electrical/power focused panels.
- `sensor_dashboard.png`: wind + data sensor focused panels.

### Dashboard Configuration

Dashboard grouping is controlled in `analyze_can_frames.py` via `DASHBOARD_CONFIG`.

- Keys are output PNG names.
- `title` sets the figure title.
- `panels` is a list of panel keys (for example `frame_counts`, `can_utilization`, `rudder`, `geo`, `geo_gps_scaled`, `pdb_voltages`, `battery_temps`, `wind_angle_split`, `wind_speed_split`, `sensor_temp`, `sensor_ph`, `sensor_cond`).

Edit this map to quickly choose which graphs appear on which PNG.

## Currently Decoded IDs

- Main/control: `0x001`, `0x002`
- Wind: `0x040`, `0x041`
- AIS/GPS: `0x060`, `0x070`
- Sensors: `0x100`, `0x110`, `0x120`
- Heartbeats: `0x130`, `0x131`, `0x132`, `0x133`
- Debug: `0x204`, `0x206`

Unknown/undocumented IDs are still kept in `parsed_frames.csv` so nothing is discarded.
