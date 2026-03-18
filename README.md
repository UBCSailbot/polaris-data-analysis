# POLARIS CAN Analysis Tool

WARNING: This repo is nearly entirely AI generated.

## Setup (assuming you are running Ubuntu)

* Buy chatGPT plus
* Install vscode if you don't have it already
* Install the openAI CODEX extension
* In the base directory of the repo run:
  * Run `python3 -m venv venv`
  * Run `source venv/bin/activate`
  * Run `python3 -m pip install -e . -r requirements.txt`
* See quick start with how to run the script.

## Features

This repo now includes parser + visualization tooling for CAN dumps:

- CLI entrypoint: `analyze_can_frames.py` (wrapper)
- Package modules: `polaris_can_analysis/`
- Input CSV format: `Timestamp,Elapsed_Time_s,CAN_Message`
- Example input file: `data/candump_20260315_153322.csv`

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

Run with cached-only satellite imagery (no network):

```bash
python3 analyze_can_frames.py --basemap satellite --basemap-offline
```

## Outputs

By default the script writes to `outputs/`:

- `parsed_frames.csv`: one row per frame, including CAN ID, DLC, payload bytes, and parse warnings.
- `decoded_signals.csv`: one row per decoded signal value.
- `full/physical_dashboard_full.png`: full physical/navigation dashboard.
- `full/electrical_dashboard_full.png`: full electrical/power dashboard.
- `full/sensor_dashboard_full.png`: full wind + data sensor dashboard.
- `on_water/physical_dashboard_trimmed.png`: on-water-only physical/navigation dashboard.
- `on_water/electrical_dashboard_trimmed.png`: on-water-only electrical/power dashboard.
- `on_water/sensor_dashboard_trimmed.png`: on-water-only wind + data sensor dashboard.

### Basemap (Satellite Imagery)

- GPS/AIS panels use Esri World Imagery by default (no API key required).
- Tiles are cached locally under `data/tile_cache/`.
- Cached tiles can be committed or shared so teammates can render without internet.
- Use `--basemap-offline` to force cache-only behavior.
- Use `--basemap none` to disable imagery and keep the plain background.
- Physical dashboards now include four geo panels:
  - plain local-scale
  - plain GPS-scaled
  - imagery local-scale
  - imagery GPS-scaled

### Dashboard Configuration

Dashboard grouping is controlled in `polaris_can_analysis/config.py` via
`DASHBOARD_CONFIG`.

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

## Formatting

Black formatting is configured via `pyproject.toml` with a max line length of 90.
For VS Code users, `.vscode/settings.json` enables format-on-save using the Black
extension.

Install formatting tooling with:

```bash
pip install -e ".[dev]"
```
