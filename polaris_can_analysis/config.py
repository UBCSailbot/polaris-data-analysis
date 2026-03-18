from __future__ import annotations

import re
from typing import Dict


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
AIS_TRACK_CONNECT_MAX_JUMP_KM = 0.3
GEO_MARGIN_FRAC = 0.05
GEO_HORIZONTAL_PAD_FRAC = 0.10
CAN_NOMINAL_BITRATE_BPS = 500_000.0
CAN_DATA_BITRATE_BPS = 1_000_000.0
UTILIZATION_WINDOW_S = 0.5
CANFD_STUFFING_FACTOR = 1.15
UTILIZATION_MAX_WINDOW_S = 5.0
CONDUCTIVITY_FILTER_WINDOW = 60
ON_WATER_STEADY_TOP_QUANTILE = 0.95
ON_WATER_THRESHOLD_FRAC = 0.90
ON_WATER_MIN_CONSECUTIVE = 3
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
            "geo_imagery",
            "geo_gps_scaled_imagery",
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
