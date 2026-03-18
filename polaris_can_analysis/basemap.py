from __future__ import annotations

import math
import json
from dataclasses import dataclass
from datetime import date
from email.utils import parsedate_to_datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from polaris_can_analysis.analytics import project_to_local_km
from polaris_can_analysis.config import EARTH_RADIUS_KM

WEB_MERCATOR_MAX_LAT = 85.05112878
WEB_TILE_SIZE = 256
MIN_ZOOM = 2
MAX_ZOOM = 19


@dataclass(frozen=True)
class TileProvider:
    key: str
    url_template: str
    attribution: str
    extension: str


TILE_PROVIDERS: Dict[str, TileProvider] = {
    "esri_world_imagery": TileProvider(
        key="esri_world_imagery",
        url_template=(
            "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
            "MapServer/tile/{z}/{y}/{x}"
        ),
        attribution="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        extension="jpg",
    )
}

_NETWORK_FETCH_ALLOWED = True


def clamp_lat(lat_deg: float) -> float:
    return max(-WEB_MERCATOR_MAX_LAT, min(WEB_MERCATOR_MAX_LAT, float(lat_deg)))


def local_km_to_lonlat(
    x_km: float,
    y_km: float,
    origin_lon_deg: float,
    origin_lat_deg: float,
) -> Tuple[float, float]:
    lat0_rad = math.radians(origin_lat_deg)
    cos_lat0 = max(1e-12, abs(math.cos(lat0_rad)))

    lon = float(origin_lon_deg) + math.degrees(float(x_km) / (EARTH_RADIUS_KM * cos_lat0))
    lat = float(origin_lat_deg) + math.degrees(float(y_km) / EARTH_RADIUS_KM)
    return lon, lat


def lonlat_to_tile_xy(lon_deg: float, lat_deg: float, zoom: int) -> Tuple[int, int]:
    n = 2**zoom
    lat_clamped = clamp_lat(lat_deg)

    x_float = ((float(lon_deg) + 180.0) / 360.0) * n
    lat_rad = math.radians(lat_clamped)
    y_float = (
        (1.0 - (math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi))
        / 2.0
        * n
    )

    x = int(math.floor(x_float))
    y = int(math.floor(y_float))
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y


def tile_xy_to_lonlat(x: int, y: int, zoom: int) -> Tuple[float, float]:
    n = 2**zoom
    lon_deg = (float(x) / n) * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - (2.0 * float(y) / n))))
    lat_deg = math.degrees(lat_rad)
    return lon_deg, lat_deg


def tile_range_for_bbox(
    lon_min_deg: float,
    lat_min_deg: float,
    lon_max_deg: float,
    lat_max_deg: float,
    zoom: int,
) -> Tuple[int, int, int, int]:
    x0, y0 = lonlat_to_tile_xy(lon_min_deg, lat_max_deg, zoom)
    x1, y1 = lonlat_to_tile_xy(lon_max_deg, lat_min_deg, zoom)
    return min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)


def choose_zoom(
    lon_min_deg: float,
    lat_min_deg: float,
    lon_max_deg: float,
    lat_max_deg: float,
    max_tiles: int,
) -> int:
    for zoom in range(MAX_ZOOM, MIN_ZOOM - 1, -1):
        x0, x1, y0, y1 = tile_range_for_bbox(
            lon_min_deg,
            lat_min_deg,
            lon_max_deg,
            lat_max_deg,
            zoom,
        )
        tile_count = (x1 - x0 + 1) * (y1 - y0 + 1)
        if tile_count <= max_tiles:
            return zoom
    return MIN_ZOOM


def tile_cache_path(
    cache_dir: Path,
    provider: TileProvider,
    zoom: int,
    x: int,
    y: int,
) -> Path:
    return cache_dir / provider.key / str(zoom) / str(x) / f"{y}.{provider.extension}"


def tile_meta_path(
    cache_dir: Path,
    provider: TileProvider,
    zoom: int,
    x: int,
    y: int,
) -> Path:
    return cache_dir / provider.key / str(zoom) / str(x) / f"{y}.meta.json"


def parse_http_last_modified(http_value: Optional[str]) -> Optional[date]:
    if not http_value:
        return None
    try:
        dt = parsedate_to_datetime(http_value)
    except (TypeError, ValueError, OverflowError):
        return None
    return dt.date()


def read_tile_date(path: Path) -> Optional[date]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw = payload.get("last_modified_date")
    if not isinstance(raw, str):
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def write_tile_date(path: Path, tile_date: Optional[date]) -> None:
    if tile_date is None:
        return
    payload = {"last_modified_date": tile_date.isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        # Cache metadata should not break plotting.
        pass


def decode_tile_bytes(raw_bytes: bytes) -> Optional[np.ndarray]:
    try:
        with Image.open(BytesIO(raw_bytes)) as img:
            rgba = img.convert("RGBA")
            return np.asarray(rgba, dtype=np.uint8)
    except OSError:
        return None


def read_cached_tile(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        with Image.open(path) as img:
            rgba = img.convert("RGBA")
            return np.asarray(rgba, dtype=np.uint8)
    except OSError:
        return None


def fetch_tile(
    provider: TileProvider,
    zoom: int,
    x: int,
    y: int,
    cache_dir: Path,
    offline: bool = False,
    timeout_s: float = 8.0,
) -> Tuple[Optional[np.ndarray], Optional[date]]:
    cache_file = tile_cache_path(cache_dir, provider, zoom, x, y)
    meta_file = tile_meta_path(cache_dir, provider, zoom, x, y)
    cached = read_cached_tile(cache_file)
    if cached is not None:
        return cached, read_tile_date(meta_file)

    global _NETWORK_FETCH_ALLOWED
    if offline or not _NETWORK_FETCH_ALLOWED:
        return None, None

    url = provider.url_template.format(z=zoom, x=x, y=y)
    request = Request(url, headers={"User-Agent": "polaris-can-analysis/0.1"})

    try:
        with urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
            tile_date = parse_http_last_modified(response.headers.get("Last-Modified"))
    except (URLError, TimeoutError, OSError):
        _NETWORK_FETCH_ALLOWED = False
        return None, None

    tile = decode_tile_bytes(raw)
    if tile is None:
        return None, None

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache_file.write_bytes(raw)
    except OSError:
        # Plotting should still proceed even if writing cache fails.
        pass
    write_tile_date(meta_file, tile_date)

    return tile, tile_date


def build_tile_mosaic(
    provider: TileProvider,
    lon_min_deg: float,
    lat_min_deg: float,
    lon_max_deg: float,
    lat_max_deg: float,
    cache_dir: Path,
    offline: bool = False,
    max_tiles: int = 64,
) -> Optional[Tuple[np.ndarray, int, int, int, int, int, Optional[date]]]:
    zoom = choose_zoom(
        lon_min_deg,
        lat_min_deg,
        lon_max_deg,
        lat_max_deg,
        max_tiles=max_tiles,
    )
    x0, x1, y0, y1 = tile_range_for_bbox(
        lon_min_deg,
        lat_min_deg,
        lon_max_deg,
        lat_max_deg,
        zoom,
    )

    rows = y1 - y0 + 1
    cols = x1 - x0 + 1
    mosaic = np.zeros((rows * WEB_TILE_SIZE, cols * WEB_TILE_SIZE, 4), dtype=np.uint8)
    have_any = False
    oldest_tile_date: Optional[date] = None

    for row, tile_y in enumerate(range(y0, y1 + 1)):
        for col, tile_x in enumerate(range(x0, x1 + 1)):
            tile, tile_date = fetch_tile(
                provider=provider,
                zoom=zoom,
                x=tile_x,
                y=tile_y,
                cache_dir=cache_dir,
                offline=offline,
            )
            if tile is None:
                continue
            if tile_date is not None and (
                oldest_tile_date is None or tile_date < oldest_tile_date
            ):
                oldest_tile_date = tile_date

            if tile.shape[0] != WEB_TILE_SIZE or tile.shape[1] != WEB_TILE_SIZE:
                with Image.fromarray(tile) as img:
                    img_resized = img.resize((WEB_TILE_SIZE, WEB_TILE_SIZE), Image.Resampling.BILINEAR)
                    tile = np.asarray(img_resized, dtype=np.uint8)

            y_start = row * WEB_TILE_SIZE
            x_start = col * WEB_TILE_SIZE
            mosaic[y_start : y_start + WEB_TILE_SIZE, x_start : x_start + WEB_TILE_SIZE] = tile
            have_any = True

    if not have_any:
        return None

    return mosaic, zoom, x0, x1, y0, y1, oldest_tile_date


def add_satellite_basemap(
    ax,
    origin_lon_deg: float,
    origin_lat_deg: float,
    x_lo_km: float,
    x_hi_km: float,
    y_lo_km: float,
    y_hi_km: float,
    cache_dir: Path,
    provider_key: str = "esri_world_imagery",
    offline: bool = False,
    max_tiles: int = 64,
    alpha: float = 0.90,
) -> Tuple[bool, Optional[str]]:
    provider = TILE_PROVIDERS.get(provider_key)
    if provider is None:
        return False, None

    lon1, lat1 = local_km_to_lonlat(x_lo_km, y_lo_km, origin_lon_deg, origin_lat_deg)
    lon2, lat2 = local_km_to_lonlat(x_hi_km, y_hi_km, origin_lon_deg, origin_lat_deg)

    lon_min = min(lon1, lon2)
    lon_max = max(lon1, lon2)
    lat_min = clamp_lat(min(lat1, lat2))
    lat_max = clamp_lat(max(lat1, lat2))

    if lon_max <= lon_min or lat_max <= lat_min:
        return False, None

    # Expand the request bbox slightly so imagery fully covers the plotted extent.
    lon_span = max(lon_max - lon_min, 1e-9)
    lat_span = max(lat_max - lat_min, 1e-9)
    lon_pad = 0.02 * lon_span
    lat_pad = 0.02 * lat_span
    lon_min -= lon_pad
    lon_max += lon_pad
    lat_min = clamp_lat(lat_min - lat_pad)
    lat_max = clamp_lat(lat_max + lat_pad)

    built = build_tile_mosaic(
        provider=provider,
        lon_min_deg=lon_min,
        lat_min_deg=lat_min,
        lon_max_deg=lon_max,
        lat_max_deg=lat_max,
        cache_dir=cache_dir,
        offline=offline,
        max_tiles=max_tiles,
    )
    if built is None:
        return False, None

    mosaic, zoom, x0, x1, y0, y1, oldest_tile_date = built

    west_lon, north_lat = tile_xy_to_lonlat(x0, y0, zoom)
    east_lon, south_lat = tile_xy_to_lonlat(x1 + 1, y1 + 1, zoom)

    x_west, y_north = project_to_local_km(
        np.array([west_lon]),
        np.array([north_lat]),
        origin_lon_deg,
        origin_lat_deg,
    )
    x_east, y_south = project_to_local_km(
        np.array([east_lon]),
        np.array([south_lat]),
        origin_lon_deg,
        origin_lat_deg,
    )

    extent = (
        float(x_west[0]),
        float(x_east[0]),
        float(y_south[0]),
        float(y_north[0]),
    )

    xlim_before = ax.get_xlim()
    ylim_before = ax.get_ylim()

    ax.imshow(
        mosaic,
        extent=extent,
        origin="upper",
        interpolation="bilinear",
        alpha=float(alpha),
        zorder=-20,
    )
    ax.set_xlim(*xlim_before)
    ax.set_ylim(*ylim_before)

    ax.text(
        0.01,
        0.01,
        f"Imagery © {provider.attribution}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6,
        color="#E5E7EB",
        bbox={"facecolor": "#111827", "alpha": 0.45, "pad": 1.5, "edgecolor": "none"},
        zorder=15,
    )

    oldest_tile_date_text = oldest_tile_date.isoformat() if oldest_tile_date else None
    return True, oldest_tile_date_text
