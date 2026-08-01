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

- Top-level CLI scripts:
  - `analyze_can_frames.py` — parse and decode a single candump, write per-file dashboards
  - `decode_all_candumps.py` — decode each session into `outputs/<session>/decoded_signals.csv`
  - `combine_session_logs.py` — merge each session's candumps into `outputs/<session>/combined_can_frames.csv`
  - `build_physical_dashboard.py` — render each session's dashboards into `outputs/<session>/`
- Package modules: `polaris_can_analysis/`
- Input CSV format: `Timestamp,Elapsed_Time_s,CAN_Message`

It is intentionally tolerant of partial implementations and shorter payloads that appear in real logs.

### Data Layout

Candumps are stored per on-water test session:

```
data/
  25Nov8_owt/candump_20251108_155236.csv
  26Mar15_owt/candump_20260315_120418.csv    # + 13 more
  26May23_owt/candump_20260523_134405.csv    # + 11 more
  26Jun6_owt/candump_20260606_020740.csv     # + 51 more
  tile_cache/                                # basemap tiles, not candump data
```

`--data-dir` is searched **recursively**, so the default `data` sweeps every
session folder at once. Pass one session (`--data-dir data/26Jun6_owt`) to work
on a single test. Files are ordered by filename, which is chronological given
the `candump_YYYYMMDD_HHMMSS.csv` naming; `tile_cache/` is always skipped, and
empty session folders are simply ignored.

`decode_all_candumps.py`, `combine_session_logs.py`, and
`build_physical_dashboard.py` mirror this layout into `outputs/`, one folder per
session:

```
outputs/
  25Nov8_owt/decoded_signals.csv
  25Nov8_owt/combined_can_frames.csv
  25Nov8_owt/physical_dashboard.png
  25Nov8_owt/electrical_dashboard.png
  25Nov8_owt/sensor_dashboard.png
  26Mar15_owt/…
```

## Quick Start

Run on the earliest candump found under `data/`:

```bash
python3 analyze_can_frames.py
```

Run on a specific file:

```bash
python3 analyze_can_frames.py --input data/26Jun6_owt/candump_20260606_020740.csv
```

Write outputs to a custom directory:

```bash
python3 analyze_can_frames.py --outdir outputs/run_01
```

Run with cached-only satellite imagery (no network):

```bash
python3 analyze_can_frames.py --basemap satellite --basemap-offline
```

## Commands

All commands are run from the repo root inside the activated venv.

### `analyze_can_frames.py` — parse + decode + plot one candump

Single-file pipeline. Writes `parsed_frames.csv`, `decoded_signals.csv`, and the three dashboards (full + on-water trimmed when on-water start is detected from conductivity).

```bash
python3 analyze_can_frames.py [--input PATH] [--outdir DIR] [--max-rows N]
                              [--skip-plot]
                              [--basemap {satellite,none}] [--basemap-offline]
                              [--tile-cache-dir DIR]
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--input PATH` | earliest `candump_*.csv` under `data/` (any session), else first CSV in `data/` | candump CSV to process |
| `--outdir DIR` | `outputs` | output directory root |
| `--max-rows N` | unlimited | cap rows parsed (quick iteration) |
| `--skip-plot` | off | only write CSVs, no PNGs |
| `--basemap satellite\|none` | `satellite` | background imagery for geo panels |
| `--basemap-offline` | off | cache-only basemap tiles |
| `--tile-cache-dir DIR` | `data/tile_cache` | tile cache location |

### `decode_all_candumps.py` — one decoded CSV per session

Groups the candumps under `--data-dir` by session folder and streams each
session's files through the same parser/decoder used above, writing
`outputs/<session>/decoded_signals.csv`. Within a session file the absolute
`timestamp` column distinguishes sources; `elapsed_s` is per-file and repeats.

```bash
python3 decode_all_candumps.py [--data-dir DIR] [--output-dir DIR]
                               [--filename NAME] [--glob PATTERN]
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--data-dir DIR` | `data` | root to search recursively (e.g. `data/26Jun6_owt` for one session) |
| `--output-dir DIR` | `outputs` | root for the per-session output folders |
| `--filename NAME` | `decoded_signals.csv` | CSV name written inside each session folder |
| `--glob PATTERN` | `candump_*.csv` | filename glob, matched at any depth |

Decode every session:

```bash
python3 decode_all_candumps.py
# -> outputs/25Nov8_owt/decoded_signals.csv, outputs/26Mar15_owt/…, …
```

Decode one session:

```bash
python3 decode_all_candumps.py --data-dir data/26May23_owt
# -> outputs/26May23_owt/decoded_signals.csv
```

### `combine_session_logs.py` — one candump per session

A session is captured as many candump files, each a slice of the same test.
This merges all of them into a single `outputs/<session>/combined_can_frames.csv`
ordered by the absolute `Timestamp`.

The output keeps the candump schema verbatim — `Timestamp,Elapsed_Time_s,CAN_Message`
— so it feeds straight back into `analyze_can_frames.py --input`. AIS is carried
in the CAN frames themselves (ID `0x060`, `SAIL_AIS`) and so is included; the
separate `ais_values_*.csv` files are already-decoded ship reports, not CAN
frames, and are not merged.

```bash
python3 combine_session_logs.py [--data-dir DIR] [--output-dir DIR]
                                [--filename NAME] [--glob PATTERN]
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--data-dir DIR` | `data` | root to search recursively (e.g. `data/26Jun6_owt` for one session) |
| `--output-dir DIR` | `outputs` | root for the per-session output folders |
| `--filename NAME` | `combined_can_frames.csv` | CSV name written inside each session folder |
| `--glob PATTERN` | `candump_*.csv` | filename glob, matched at any depth |

Combine every session:

```bash
python3 combine_session_logs.py
# -> outputs/25Nov8_owt/combined_can_frames.csv, outputs/26Mar15_owt/…, …
```

Combine one session, then analyze it as a single candump:

```bash
python3 combine_session_logs.py --data-dir data/26May23_owt
python3 analyze_can_frames.py --input outputs/26May23_owt/combined_can_frames.csv
```

Because each capture file is itself written chronologically, the merge streams
the files rather than loading a session into memory, so multi-hundred-MB
sessions combine in constant memory.

Two caveats on the combined file:

- `Elapsed_Time_s` is measured **per capture file**, so it restarts partway
  through the combined output. `Timestamp` is the only session-wide ordering
  key. (`build_physical_dashboard.py` solves the same problem by rebasing onto
  one clock; this script deliberately passes the original column through
  untouched.)
- Truncated `CAN_Message` values that already exist in the source candumps are
  passed through verbatim rather than dropped, matching the repo's tolerance of
  partial real-world logs. The parser records them as `parse_warning`.

### `build_physical_dashboard.py` — dashboards per session

Groups the candumps under `--data-dir` by session folder, rebases each session's
files onto one clock derived from their absolute timestamps (so the per-file
timers don't overlap), and writes that session's dashboards to
`outputs/<session>/`. All three dashboards are rendered unless `--config-key`
narrows it.

```bash
python3 build_physical_dashboard.py [--data-dir DIR] [--output-dir DIR]
                                    [--config-key KEY ...] [--combined]
                                    [--glob PATTERN]
                                    [--basemap {satellite,none}] [--basemap-offline]
                                    [--tile-cache-dir DIR]
                                    [--timezone TZ]
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--data-dir DIR` | `data` | root to search recursively (e.g. `data/26Jun6_owt` for one session) |
| `--output-dir DIR` | `outputs` | root for the per-session output folders |
| `--config-key KEY ...` | all three | keys into `DASHBOARD_CONFIG`: `physical_dashboard.png`, `electrical_dashboard.png`, `sensor_dashboard.png` |
| `--combined` | off | render one set across every session into `<output-dir>/full/` instead |
| `--glob PATTERN` | `candump_*.csv` | filename glob, matched at any depth |
| `--basemap`, `--basemap-offline`, `--tile-cache-dir` | as in `analyze_can_frames.py` | basemap controls |
| `--timezone TZ` | (unset → elapsed `H:MM`) | IANA tz for a wall-clock x-axis, e.g. `America/Los_Angeles` for PDT/PST |

Render every session's dashboards:

```bash
python3 build_physical_dashboard.py
# -> outputs/25Nov8_owt/physical_dashboard.png, …/electrical_dashboard.png, …
```

One session, one dashboard:

```bash
python3 build_physical_dashboard.py \
  --data-dir data/26May23_owt \
  --config-key sensor_dashboard.png
# -> outputs/26May23_owt/sensor_dashboard.png
```

Everything on one global time axis (`_full` suffix, written to `outputs/full/`):

```bash
python3 build_physical_dashboard.py --combined
```

Note that `--combined` spans months once several sessions exist, so the time-series
panels stretch across the gaps between tests and their x-axis switches from `h:mm`
to elapsed days. The per-session default is the readable view.

## Outputs

Everything lands under `outputs/`.

`analyze_can_frames.py` (single candump) writes to the `--outdir` root:

- `parsed_frames.csv`: one row per frame, including CAN ID, DLC, payload bytes, and parse warnings.
- `decoded_signals.csv`: one row per decoded signal value.
- `full/physical_dashboard_full.png`: full physical/navigation dashboard.
- `full/electrical_dashboard_full.png`: full electrical/power dashboard.
- `full/sensor_dashboard_full.png`: full wind + data sensor dashboard.
- `on_water/physical_dashboard_trimmed.png`: on-water-only physical/navigation dashboard.
- `on_water/electrical_dashboard_trimmed.png`: on-water-only electrical/power dashboard.
- `on_water/sensor_dashboard_trimmed.png`: on-water-only wind + data sensor dashboard.

`decode_all_candumps.py`, `combine_session_logs.py`, and
`build_physical_dashboard.py` write one folder per session,
`outputs/<session>/`:

- `decoded_signals.csv`: every decoded signal from that session's candumps.
- `combined_can_frames.csv`: that session's candumps merged into one
  chronological CAN log, same three columns as the input candumps.
- `physical_dashboard.png`, `electrical_dashboard.png`, `sensor_dashboard.png`:
  that session's dashboards, all its candumps on one clock.

`build_physical_dashboard.py --combined` writes the all-sessions versions to
`outputs/full/` with a `_full` suffix.

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
- `panels` is a list of panel keys: `frame_counts`, `can_utilization`, `rudder`, `imu`, `geo`, `geo_gps_scaled`, `geo_imagery`, `geo_gps_scaled_imagery`, `pdb_voltages`, `battery_temps`, `wind_angle_split`, `wind_speed_split`, `sensor_temp`, `sensor_ph`, `sensor_cond`.

Edit this map to quickly choose which graphs appear on which PNG.

## Currently Decoded IDs

- Main/control: `0x001`, `0x002`
- Wind: `0x040`, `0x041`
- Rudder data: `0x050`
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
