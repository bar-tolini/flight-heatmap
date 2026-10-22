#!/usr/bin/env python3
"""
Privacy-Preserving Multi-Zoom Flight Density Heatmap Generator
==============================================================
Generates a standalone, lightweight, privacy-preserving rasterized multi-zoom
density map (ImageOverlay) from flight trajectory data with interactive
multi-criteria logical AND filtering:
  - Altitude Tier Layers (6 tiers + Combined)
  - Ground Noise Exposure Threshold (AND single-select: All, 50+ dB(A), 60+ dB(A), 70+ dB(A))
  - Aircraft Group Category (AND single-select: All, Airliners, Helis)

100% Privacy-Preserving & Public-Ready: Zero raw GPS points, zero timestamps,
and zero aircraft identifiers exposed in the client-side HTML.
"""

import base64
import gc
import io
import json
import os
import sys
from datetime import UTC, datetime
from typing import Any

import folium
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from folium.plugins import Fullscreen, MousePosition
from folium.raster_layers import ImageOverlay
from PIL import Image, ImageFilter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PARQUET = os.path.join(SCRIPT_DIR, "processed_flight_paths.parquet")
DEFAULT_OUTPUT_HTML = os.path.join(SCRIPT_DIR, "opensky_flight_density_map.html")

# Default Bounding Box (Zurich TMA / Lake Zurich area)
DEFAULT_LAT_MIN = 47.1102
DEFAULT_LAT_MAX = 47.4855
DEFAULT_LON_MIN = 8.3720
DEFAULT_LON_MAX = 9.0478

# Ground Noise Exposure Single-Select Thresholds (None = No noise filter / All traffic)
NOISE_CONFIGS = [
    {"id": "none", "label": "All", "val": None, "tag": "All Noise Levels"},
    {"id": "50", "label": "50+ dB", "val": 50, "tag": "≥50 dB(A)"},
    {"id": "60", "label": "60+ dB", "val": 60, "tag": "≥60 dB(A)"},
    {"id": "70", "label": "70+ dB", "val": 70, "tag": "≥70 dB(A)"},
]

# Aircraft Group Type Single-Select Filters
AIRCRAFT_CONFIGS = [
    {"id": "all", "label": "✈️ All", "groups": None},
    {
        "id": "airliners",
        "label": "🛫 Airliners",
        "groups": ["Commercial Medium Jet", "Commercial Heavy Jet"],
    },
    {"id": "helis", "label": "🚁 Helis", "groups": ["Helicopter / Rotorcraft"]},
]

# Multi-Zoom Adaptive Resolution Tiers
ZOOM_TIERS = {
    "low": {
        "grid_h": 600,
        "grid_w": 800,
        "blur": 4.8,
        "p_norm": 96.8,
        "gamma": 0.47,
    },  # z8 - z10
    "med": {
        "grid_h": 900,
        "grid_w": 1200,
        "blur": 3.5,
        "p_norm": 97.2,
        "gamma": 0.50,
    },  # z11 - z12
    "high": {
        "grid_h": 1200,
        "grid_w": 1600,
        "blur": 2.4,
        "p_norm": 97.8,
        "gamma": 0.54,
    },  # z13 - z14
}

TRANSPARENT_PIXEL_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAKey=="


def render_single_zoom_raster(
    pts_lat: np.ndarray,
    pts_lon: np.ndarray,
    weights: np.ndarray | None,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    lut_256: np.ndarray,
    zoom_cfg: dict[str, Any],
) -> str:
    """
    Renders an ultra-strong, vibrant RGBA PNG raster for a specific zoom level.
    Uses memory-efficient 256-color uint8 lookup table (LUT) and fast PNG encoding.
    Returns base64 data URI string or empty string if no valid points.
    """
    if len(pts_lat) == 0:
        return ""

    grid_h = zoom_cfg["grid_h"]
    grid_w = zoom_cfg["grid_w"]
    blur_r = zoom_cfg["blur"]
    p_norm = zoom_cfg["p_norm"]
    gamma = zoom_cfg["gamma"]

    # 2D Histogram
    hist, _, _ = np.histogram2d(
        pts_lat,
        pts_lon,
        bins=[grid_h, grid_w],
        range=[[lat_min, lat_max], [lon_min, lon_max]],
        weights=weights,
    )

    if hist.max() == 0:
        return ""

    # Logarithmic compression to prevent runway hotspots from drowning out cruising lanes
    log_hist = np.log1p(hist)
    m = log_hist.max()
    if m == 0:
        return ""

    norm_raw = (log_hist / m * 255.0).astype(np.uint8)

    # Fast Gaussian Blur via PIL
    img_gray = Image.fromarray(norm_raw[::-1], mode="L")
    blurred_img = img_gray.filter(ImageFilter.GaussianBlur(radius=blur_r))
    blurred = np.array(blurred_img, dtype=np.float32) / 255.0

    # Strong Percentile Normalization
    pos = blurred[blurred > 0.003]
    p_val = float(np.percentile(pos, p_norm)) if len(pos) > 0 else 1.0
    if p_val <= 0:
        p_val = 1.0

    norm = np.clip(blurred / p_val, 0.0, 1.0)
    norm = np.power(norm, gamma)  # Strong gamma lift for mid-densities

    # Memory-Optimized Color Mapping via precomputed 256-color uint8 LUT
    idx = (norm * 255.0).astype(np.uint8)
    rgba = lut_256[idx].copy()

    # Active flight corridors baseline opacity (~46%) up to ~95% opacity
    active = blurred > 0.001
    alpha = np.zeros((grid_h, grid_w), dtype=np.uint8)
    alpha[active] = np.clip(
        (0.46 + 0.49 * (norm[active] ** 0.70)) * 255.0, 0.0, 242.0
    ).astype(np.uint8)
    rgba[..., 3] = alpha

    # Encode to fast in-memory PNG
    out_img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    out_img.save(buf, format="PNG", optimize=False)
    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
    buf.close()

    del (
        hist,
        log_hist,
        norm_raw,
        img_gray,
        blurred_img,
        blurred,
        pos,
        norm,
        idx,
        alpha,
        rgba,
        out_img,
    )
    return f"data:image/png;base64,{b64_str}"


def build_public_density_map(
    df: pd.DataFrame | str,
    lat_min: float = DEFAULT_LAT_MIN,
    lat_max: float = DEFAULT_LAT_MAX,
    lon_min: float = DEFAULT_LON_MIN,
    lon_max: float = DEFAULT_LON_MAX,
    region_name: str | None = None,
    output_html_path: str = DEFAULT_OUTPUT_HTML,
) -> folium.Map:
    """
    Builds the standalone, privacy-preserving, multi-zoom adaptive density Folium map.
    Pre-computes rasters for all single-select logical AND filter combinations:
      - Ground Noise Exposure: All (No filter), 50+ dB(A), 60+ dB(A), 70+ dB(A)
      - Aircraft Group: All, Airliners, Helis
      - Altitude Tiers: Red (0-500m), Orange (500-1000m), Amber (1000-2000m),
                        Yellow (2000-3000m), Teal (3000-5000m), Green (5000m+),
                        and Combined All-Altitudes.
    """
    loaded_from_file = False
    if isinstance(df, str):
        parquet_file = df
        if not os.path.exists(parquet_file):
            raise FileNotFoundError(f"Input Parquet file not found: {parquet_file}")
        print(f"📂 Loading trajectory points for density map from: {parquet_file}")
        desired_cols = [
            "latitude",
            "longitude",
            "altitude_tier",
            "ground_noise_dba",
            "aircraft_type_group",
            "timestamp",
        ]
        try:
            df = pd.read_parquet(parquet_file, columns=desired_cols)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ Column projection read note: {e}. Attempting full read...")
            df = pd.read_parquet(parquet_file)
        if "latitude" in df.columns:
            df["latitude"] = df["latitude"].astype(np.float32)
        if "longitude" in df.columns:
            df["longitude"] = df["longitude"].astype(np.float32)
        if "altitude_tier" in df.columns:
            df["altitude_tier"] = df["altitude_tier"].astype("category")
        if "aircraft_type_group" in df.columns:
            df["aircraft_type_group"] = df["aircraft_type_group"].astype("category")
        loaded_from_file = True
        print(
            f"  • Ingested {len(df):,} waypoints ({df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB in RAM)"
        )

    # Bounding box filter
    mask_bbox = (
        (df["latitude"] >= lat_min)
        & (df["latitude"] <= lat_max)
        & (df["longitude"] >= lon_min)
        & (df["longitude"] <= lon_max)
    )
    df = df[mask_bbox].copy()
    print(
        f"  • Filtered to bbox [{lat_min:.4f}, {lat_max:.4f}] N, [{lon_min:.4f}, {lon_max:.4f}] E: {len(df):,} records"
    )

    if region_name is None:
        region_name = f"Map Region [{lat_min:.2f}° to {lat_max:.2f}° N, {lon_min:.2f}° to {lon_max:.2f}° E]"

    center_lat = (lat_min + lat_max) / 2.0
    center_lon = (lon_min + lon_max) / 2.0

    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=10, tiles=None, control_scale=True
    )

    # 1. Base Map Tile Layers (Zero-Key Low-Contrast Alternatives)
    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        name="☀️ Muted Grayscale (OSM Mono)",
        className="tiles-muted-mono",
        max_zoom=19,
        show=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        name="🌑 Midnight Slate (Dark Canvas)",
        className="tiles-midnight-dark",
        max_zoom=19,
        show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, HOT',
        name="🌸 Soft Pastel (OSM Humanitarian)",
        max_zoom=19,
        show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg",
        attr='&copy; <a href="https://www.swisstopo.admin.ch/">swisstopo</a>',
        name="🇨🇭 Swisstopo Grau (Federal Grey)",
        max_zoom=18,
        show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="Map data: &copy; OpenStreetMap contributors, SRTM | Map style: &copy; OpenTopoMap (CC-BY-SA)",
        name="⛰️ OpenTopoMap (Topographic)",
        max_zoom=17,
        show=False,
    ).add_to(m)

    m.get_root().header.add_child(
        folium.Element("""
    <style>
    /* Clean Low-Contrast Map Tile Styling (Zero API Key) */
    .leaflet-tile-pane .tiles-muted-mono {
        filter: grayscale(100%) contrast(72%) brightness(108%) !important;
    }
    .leaflet-tile-pane .tiles-midnight-dark {
        filter: grayscale(100%) invert(92%) contrast(85%) brightness(95%) !important;
    }
    </style>
    """)
    )

    # 2. Bounding Box Outline
    folium.Rectangle(
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        color="#38bdf8",
        weight=2,
        dash_array="5, 5",
        fill=False,
        tooltip=f"<b>Analysis Region:</b> {region_name}<br>Lat: [{lat_min:.4f}, {lat_max:.4f}]<br>Lon: [{lon_min:.4f}, {lon_max:.4f}]",
    ).add_to(m)

    tiers_config = [
        {
            "id": "tier_red",
            "key": "0 - 500m (Red)",
            "name": "🔴 0 – 500 m (Surface / Climb)",
            "colors": ["#fca5a5", "#f87171", "#ef4444", "#dc2626", "#800000"],
            "show": False,
        },
        {
            "id": "tier_orange",
            "key": "500 - 1000m (Orange)",
            "name": "🟧 500 – 1,000 m (Low Approach / Climb)",
            "colors": ["#fed7aa", "#fb923c", "#f97316", "#ea580c", "#a63000"],
            "show": False,
        },
        {
            "id": "tier_amber",
            "key": "1000 - 2000m (Amber)",
            "name": "🟠 1,000 – 2,000 m (Lower TMA)",
            "colors": ["#fef08a", "#facc15", "#eab308", "#d97706", "#9a3412"],
            "show": False,
        },
        {
            "id": "tier_yellow",
            "key": "2000 - 3000m (Yellow)",
            "name": "🟡 2,000 – 3,000 m (Intermediate TMA)",
            "colors": ["#fef9c3", "#fde047", "#eab308", "#ca8a04", "#854d0e"],
            "show": False,
        },
        {
            "id": "tier_teal",
            "key": "3000 - 5000m (Teal)",
            "name": "🩵 3,000 – 5,000 m (Transitions)",
            "colors": ["#a5f3fc", "#5eead4", "#14b8a6", "#0d9488", "#115e59"],
            "show": False,
        },
        {
            "id": "tier_green",
            "key": "5000m+ (Green)",
            "name": "🟢 5,000+ m (Cruising Airways)",
            "colors": ["#bbf7d0", "#86efac", "#22c55e", "#16a34a", "#087f5b"],
            "show": False,
        },
        {
            "id": "tier_combined",
            "key": None,  # All altitudes combined
            "name": "🔥 All-Altitudes Combined Density",
            "colors": [
                "#bfdbfe",
                "#38bdf8",
                "#34d399",
                "#fbbf24",
                "#f97316",
                "#ef4444",
            ],
            "show": True,
        },
    ]

    bounds = [[lat_min, lon_min], [lat_max, lon_max]]

    # Precompute 256-color LUTs for each layer
    luts: dict[str, np.ndarray] = {}
    for cfg in tiers_config:
        cmap = mcolors.LinearSegmentedColormap.from_list(cfg["id"], cfg["colors"])
        luts[cfg["id"]] = (cmap(np.linspace(0, 1, 256)) * 255.0).astype(np.uint8)

    # Multi-dimensional rasters store:
    # zoom_rasters_json[layer_id][f"{noise_id}_{aircraft_group}"][zoom_tier] = b64_png_uri
    zoom_rasters_json: dict[str, dict[str, dict[str, str]]] = {
        cfg["id"]: {} for cfg in tiers_config
    }
    filter_counts_json: dict[str, int] = {}
    overlay_var_names: dict[str, str] = {}

    print("🎨 Generating multi-filter logical AND density rasters...")
    print(f"  • Noise options : {[c['id'] for c in NOISE_CONFIGS]}")
    print(f"  • Aircraft types: {[c['id'] for c in AIRCRAFT_CONFIGS]}")
    print(f"  • Zoom tiers    : {list(ZOOM_TIERS.keys())}")

    # Process all (noise, aircraft) single-select combinations
    for noise_cfg in NOISE_CONFIGS:
        noise_id = noise_cfg["id"]
        noise_val = noise_cfg["val"]

        if noise_val is None or "ground_noise_dba" not in df.columns:
            df_noise = df
        else:
            df_noise = df[df["ground_noise_dba"] >= noise_val]

        for ac_cfg in AIRCRAFT_CONFIGS:
            ac_id = ac_cfg["id"]
            ac_groups = ac_cfg["groups"]
            if ac_groups is None or "aircraft_type_group" not in df.columns:
                df_curr = df_noise
            else:
                df_curr = df_noise[df_noise["aircraft_type_group"].isin(ac_groups)]

            filter_key = f"{noise_id}_{ac_id}"
            filter_counts_json[filter_key] = len(df_curr)

            # Render for each altitude tier + combined layer
            for cfg in tiers_config:
                lid = cfg["id"]
                tkey = cfg["key"]

                if tkey is None:
                    tier_pts = df_curr
                else:
                    tier_pts = (
                        df_curr[df_curr["altitude_tier"] == tkey]
                        if "altitude_tier" in df_curr.columns
                        else pd.DataFrame()
                    )

                if tier_pts.empty:
                    zoom_rasters_json[lid][filter_key] = {
                        ztier: "" for ztier in ZOOM_TIERS
                    }
                else:
                    lats = tier_pts["latitude"].to_numpy(dtype=np.float32, copy=False)
                    lons = tier_pts["longitude"].to_numpy(dtype=np.float32, copy=False)

                    tier_zoom_dict: dict[str, str] = {}
                    for ztier_name, z_cfg in ZOOM_TIERS.items():
                        uri = render_single_zoom_raster(
                            lats,
                            lons,
                            None,
                            lat_min,
                            lat_max,
                            lon_min,
                            lon_max,
                            luts[lid],
                            z_cfg,
                        )
                        tier_zoom_dict[ztier_name] = uri

                    zoom_rasters_json[lid][filter_key] = tier_zoom_dict

            noise_desc = f"Noise: {noise_cfg['label']}"
            print(
                f"  ✓ Processed Filter [{noise_desc} ∩ Aircraft: {ac_cfg['label']}]: {len(df_curr):,} waypoints"
            )

    # Add ImageOverlays to Folium map with default initial state (Noise: none (All), Aircraft: all, Zoom: low)
    DEFAULT_INITIAL_FILTER = "none_all"
    DEFAULT_INITIAL_ZOOM_TIER = "low"

    for cfg in tiers_config:
        lid = cfg["id"]
        initial_uri = (
            zoom_rasters_json[lid]
            .get(DEFAULT_INITIAL_FILTER, {})
            .get(DEFAULT_INITIAL_ZOOM_TIER)
            or TRANSPARENT_PIXEL_URI
        )
        fg = folium.FeatureGroup(name=cfg["name"], show=cfg["show"])
        io = ImageOverlay(
            image=initial_uri, bounds=bounds, opacity=0.94, interactive=False
        )
        io.add_to(fg)
        fg.add_to(m)
        overlay_var_names[lid] = io.get_name()

    # 4. Metadata Calculations
    t_min = (
        int(df["timestamp"].min())
        if ("timestamp" in df.columns and not df.empty)
        else 0
    )
    t_max = (
        int(df["timestamp"].max())
        if ("timestamp" in df.columns and not df.empty)
        else 0
    )
    start_str = (
        datetime.fromtimestamp(t_min, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
        if t_min
        else "N/A"
    )
    end_str = (
        datetime.fromtimestamp(t_max, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
        if t_max
        else "N/A"
    )
    span_hrs = (t_max - t_min) / 3600.0 if (t_max and t_min) else 0.0

    map_var_name = m.get_name()
    total_unfiltered_pts = len(df)

    # 5. UI Elements: Header Banner, Filter Selectors, Legend & Dynamic Adapter Script
    ui_and_zoom_script = f"""
    <style>
    /* Segmented Control Styling */
    .filter-ctrl-box {{
        margin-top: 8px;
        padding-top: 6px;
        border-top: 1px solid #334155;
    }}
    .filter-ctrl-title {{
        font-size: 11px;
        font-weight: 600;
        color: #38bdf8;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    .filter-ctrl-tag {{
        font-size: 9px;
        color: #94a3b8;
        font-weight: normal;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .segmented-control {{
        display: flex;
        gap: 4px;
        background: rgba(30, 41, 59, 0.85);
        padding: 3px;
        border-radius: 6px;
        border: 1px solid #334155;
    }}
    .btn-seg {{
        flex: 1;
        background: transparent;
        color: #cbd5e1;
        border: none;
        border-radius: 4px;
        padding: 5px 4px;
        font-size: 11px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.15s ease-in-out;
        text-align: center;
        outline: none;
        user-select: none;
        white-space: nowrap;
    }}
    .btn-seg:hover {{
        background: rgba(56, 189, 248, 0.15);
        color: #f8fafc;
    }}
    .btn-seg.active {{
        background: #0284c7;
        color: #ffffff;
        font-weight: 700;
        box-shadow: 0 1px 4px rgba(0,0,0,0.4);
    }}
    </style>

    <!-- Top Header Banner & Controller Card -->
    <div style="
        position: fixed; 
        top: 15px; 
        left: 60px; 
        background: rgba(15, 23, 42, 0.96); 
        color: #f8fafc; 
        z-index: 9999; 
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        border-radius: 8px; 
        padding: 12px 16px; 
        box-shadow: 0 6px 24px rgba(0,0,0,0.55); 
        border: 1px solid rgba(56, 189, 248, 0.35);
        backdrop-filter: blur(10px);
        width: 370px;
        max-width: 92vw;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 4px;">
            <span style="font-weight: 700; font-size: 13px; color: #38bdf8;">
                ✈️ OpenSky Flight Density Map &mdash; {region_name}
            </span>
            <span style="font-size: 9px; background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); border-radius: 4px; padding: 2px 6px; white-space: nowrap; font-weight: 600;">
                🛡️ Public Safe
            </span>
        </div>
        
        <div style="color: #cbd5e1; font-size: 10.5px; font-family: monospace; margin-top: 3px;">
            ⏰ <strong>Timespan:</strong> <span style="color: #38bdf8;">{start_str}</span> &rarr; <span style="color: #38bdf8;">{end_str}</span> ({span_hrs:.1f}h)
        </div>

        <!-- Dynamic Filter Status Badge -->
        <div id="filter_status_badge" style="margin-top: 6px; padding: 4px 8px; background: rgba(30, 41, 59, 0.9); border: 1px solid #475569; border-radius: 4px; font-size: 10.5px; color: #e2e8f0; display: flex; align-items: center; justify-content: space-between;">
            <span>🛩️ <b>All Aircraft</b> &bull; 🔊 <b>All Noise Levels</b></span>
            <span id="badge_count" style="color: #38bdf8; font-weight: 600; font-family: monospace;">{total_unfiltered_pts:,} pts</span>
        </div>

        <!-- 1. Ground Noise Exposure (AND) Single Select -->
        <div class="filter-ctrl-box">
            <div class="filter-ctrl-title">
                <span>🔊 Ground Noise Exposure (AND):</span>
                <span class="filter-ctrl-tag">Single Select</span>
            </div>
            <div class="segmented-control" id="noise-selector">
                <button type="button" class="btn-seg active" data-val="none" onclick="setNoiseFilter('none', this)">All</button>
                <button type="button" class="btn-seg" data-val="50" onclick="setNoiseFilter('50', this)">50+ dB</button>
                <button type="button" class="btn-seg" data-val="60" onclick="setNoiseFilter('60', this)">60+ dB</button>
                <button type="button" class="btn-seg" data-val="70" onclick="setNoiseFilter('70', this)">70+ dB</button>
            </div>
        </div>

        <!-- 2. Aircraft Group Types (AND) Single Select -->
        <div class="filter-ctrl-box">
            <div class="filter-ctrl-title">
                <span>🛩️ Aircraft Group (AND):</span>
                <span class="filter-ctrl-tag">Single Select</span>
            </div>
            <div class="segmented-control" id="aircraft-selector">
                <button type="button" class="btn-seg active" data-val="all" onclick="setAircraftFilter('all', this)">✈️ All</button>
                <button type="button" class="btn-seg" data-val="airliners" onclick="setAircraftFilter('airliners', this)">🛫 Airliners</button>
                <button type="button" class="btn-seg" data-val="helis" onclick="setAircraftFilter('helis', this)">🚁 Helis</button>
            </div>
        </div>

        <!-- Heatmap Opacity / Strength Slider -->
        <div style="margin-top: 8px; padding-top: 6px; border-top: 1px solid #334155;">
            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 10px; margin-bottom: 2px;">
                <span style="color: #38bdf8; font-weight: 600;">✨ Heat Intensity / Opacity:</span>
                <span id="lbl_opacity_val" style="color: #f8fafc; font-family: monospace;">95%</span>
            </div>
            <input type="range" id="slider_heat_opacity" min="30" max="100" value="95" step="5" oninput="setHeatmapOpacity(this.value)" style="width: 100%; accent-color: #38bdf8; cursor: pointer; margin: 2px 0;">
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; color: #94a3b8; font-size: 9.5px; margin-top: 4px;">
            <span id="zoom_indicator_badge" style="color: #38bdf8; font-family: monospace;">
                🔍 Zoom: z10 (Tier: Low)
            </span>
            <span style="color: #64748b;">Logical AND Active</span>
        </div>
        
        <!-- Attribution & Rights Block -->
        <div style="margin-top: 8px; padding-top: 6px; border-top: 1px solid rgba(56, 189, 248, 0.2); font-size: 9.5px; color: #94a3b8; line-height: 1.4;">
            <div>🛰️ Flight data: <a href="https://opensky-network.org" target="_blank" rel="noopener noreferrer" style="color: #38bdf8; text-decoration: underline;" title="Citation: Schäfer et al., Bringing Up OpenSky: A Large-scale ADS-B Sensor Network for Research, IPSN 2014">The OpenSky Network</a> (CC BY-NC-SA / research)</div>
            <div>🗺️ Map tiles &copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer" style="color: #cbd5e1; text-decoration: underline;">OpenStreetMap</a> &bull; &copy; <a href="https://carto.com/attributions" target="_blank" rel="noopener noreferrer" style="color: #cbd5e1; text-decoration: underline;">CARTO</a></div>
            <div>⚠️ <a href="https://github.com/bar-tolini/flight-heatmap#disclaimer" target="_blank" rel="noopener noreferrer" style="color: #38bdf8; text-decoration: underline;" title="Read the project usage and safety disclaimer in the README">Usage Disclaimer</a> (informational &amp; research only)</div>
        </div>
    </div>

    <!-- Altitude Tier Legend -->
    <div style="
        position: fixed; 
        bottom: 30px; 
        right: 30px; 
        background: rgba(15, 23, 42, 0.95); 
        color: #f8fafc; 
        z-index: 9999; 
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        border-radius: 8px; 
        padding: 12px 16px; 
        box-shadow: 0 4px 20px rgba(0,0,0,0.5); 
        border: 1px solid rgba(56, 189, 248, 0.25);
        backdrop-filter: blur(8px);
        min-width: 220px;
    ">
        <div style="font-weight: 700; font-size: 12px; color: #38bdf8; margin-bottom: 8px; border-bottom: 1px solid #334155; padding-bottom: 4px;">
            📊 Altitude Density Tiers
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px; font-size: 11px;">
            <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: #ff1a1a; margin-right: 8px; box-shadow: 0 0 6px rgba(255,26,26,0.6);"></span>
            <span>🔴 <strong>0 – 500 m</strong> (Surface / Climb)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px; font-size: 11px;">
            <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: #ff6a00; margin-right: 8px; box-shadow: 0 0 6px rgba(255,106,0,0.6);"></span>
            <span>🟧 <strong>500 – 1,000 m</strong> (Low TMA / Approach)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px; font-size: 11px;">
            <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: #ffb300; margin-right: 8px; border-radius: 2px; box-shadow: 0 0 6px rgba(255,179,0,0.6);"></span>
            <span>🟠 <strong>1,000 – 2,000 m</strong> (Lower TMA)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px; font-size: 11px;">
            <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: #fcc419; margin-right: 8px; box-shadow: 0 0 6px rgba(252,196,25,0.6);"></span>
            <span>🟡 <strong>2,000 – 3,000 m</strong> (Intermediate TMA)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px; font-size: 11px;">
            <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: #2dd4bf; margin-right: 8px; box-shadow: 0 0 6px rgba(45,212,191,0.6);"></span>
            <span>🩵 <strong>3,000 – 5,000 m</strong> (Transitions)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 7px; font-size: 11px;">
            <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: #38d9a9; margin-right: 8px; box-shadow: 0 0 6px rgba(56,217,169,0.6);"></span>
            <span>🟢 <strong>5,000+ m</strong> (Cruising Airways)</span>
        </div>
        <div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid #334155; font-size: 11px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 3px;">
                <span>🔥 <strong>Combined Density:</strong></span>
                <span style="font-size: 9px; color: #94a3b8;">Low &rarr; High</span>
            </div>
            <div style="height: 6px; border-radius: 3px; background: linear-gradient(to right, #bfdbfe, #38bdf8, #34d399, #fbbf24, #f97316, #ef4444); box-shadow: 0 0 4px rgba(249,115,22,0.4);"></div>
        </div>
        <div style="font-size: 9px; color: #64748b; border-top: 1px solid #334155; padding-top: 5px; line-height: 1.3;">
            🔍 <strong>Multi-Zoom Adaptive Raster</strong><br>
            Filter: <em>Altitude ∩ Noise Exposure ∩ Aircraft Group</em>
        </div>
    </div>

    <!-- Multi-Zoom Dynamic Switching & Logical AND Filter Script -->
    <script>
    window.addEventListener("DOMContentLoaded", function() {{
        setTimeout(function() {{
            var zoomRasters = {json.dumps(zoom_rasters_json)};
            var filterCounts = {json.dumps(filter_counts_json)};
            var overlayVars = {json.dumps(overlay_var_names)};
            var leafletMap = window['{map_var_name}'];

            var selectedNoise = 'none';
            var selectedAircraft = 'all';
            var TRANSPARENT_1PX = "{TRANSPARENT_PIXEL_URI}";

            if (!leafletMap) {{
                console.warn("Leaflet map instance '{map_var_name}' not yet available");
                return;
            }}

            window.setNoiseFilter = function(noiseVal, elem) {{
                selectedNoise = String(noiseVal);
                var container = document.getElementById("noise-selector");
                if (container) {{
                    var btns = container.querySelectorAll(".btn-seg");
                    btns.forEach(function(b) {{ b.classList.remove("active"); }});
                }}
                if (elem) elem.classList.add("active");
                updateMapRasters();
            }};

            window.setAircraftFilter = function(acVal, elem) {{
                selectedAircraft = acVal;
                var container = document.getElementById("aircraft-selector");
                if (container) {{
                    var btns = container.querySelectorAll(".btn-seg");
                    btns.forEach(function(b) {{ b.classList.remove("active"); }});
                }}
                if (elem) elem.classList.add("active");
                updateMapRasters();
            }};

            window.setHeatmapOpacity = function(val) {{
                var opacity = parseInt(val) / 100.0;
                var lbl = document.getElementById("lbl_opacity_val");
                if (lbl) lbl.textContent = val + "%";
                for (var key in overlayVars) {{
                    var varName = overlayVars[key];
                    var overlay = window[varName];
                    if (overlay && typeof overlay.setOpacity === "function") {{
                        overlay.setOpacity(opacity);
                    }}
                }}
            }};

            function getZoomTier(zoom) {{
                if (zoom <= 10) return "low";
                if (zoom <= 12) return "med";
                return "high";
            }}

            function updateMapRasters() {{
                var currentZoom = leafletMap.getZoom();
                var zoomTier = getZoomTier(currentZoom);
                var filterKey = selectedNoise + "_" + selectedAircraft;

                // Update zoom indicator badge
                var badgeZoom = document.getElementById("zoom_indicator_badge");
                if (badgeZoom) {{
                    var tierLabel = zoomTier === "low" ? "Low" : (zoomTier === "med" ? "Med" : "High");
                    badgeZoom.textContent = "🔍 Zoom: z" + currentZoom + " (Tier: " + tierLabel + ")";
                }}

                // Update filter status badge
                var badgeStatus = document.getElementById("filter_status_badge");
                var badgeCount = document.getElementById("badge_count");
                if (badgeStatus && badgeCount) {{
                    var acLabel = selectedAircraft === "all" ? "All Aircraft" : (selectedAircraft === "airliners" ? "Airliners" : "Helicopters");
                    var noiseTag = (selectedNoise === "none") ? "All Noise Levels" : ("&ge;" + selectedNoise + " dB(A)");
                    badgeStatus.querySelector("span").innerHTML = "🛩️ <b>" + acLabel + "</b> &bull; 🔊 <b>" + noiseTag + "</b>";
                    var count = filterCounts[filterKey] || 0;
                    badgeCount.textContent = count.toLocaleString() + " pts";
                }}

                // Update active rasters for all altitude layers + combined layer
                for (var layerKey in overlayVars) {{
                    var varName = overlayVars[layerKey];
                    var overlay = window[varName];
                    if (overlay && zoomRasters[layerKey]) {{
                        var filterMap = zoomRasters[layerKey][filterKey];
                        var newUrl = (filterMap && filterMap[zoomTier] && filterMap[zoomTier].length > 0) ? filterMap[zoomTier] : TRANSPARENT_1PX;
                        overlay.setUrl(newUrl);
                    }}
                }}
            }}

            leafletMap.on('zoomend', updateMapRasters);
            updateMapRasters();
        }}, 300);
    }});
    </script>
    """

    m.get_root().html.add_child(folium.Element(ui_and_zoom_script))

    Fullscreen(
        position="topright",
        title="Toggle Fullscreen",
        title_cancel="Exit Fullscreen",
        force_separate_button=True,
    ).add_to(m)
    MousePosition(
        position="bottomleft",
        separator=" | Lat: ",
        empty_string="NaN",
        lng_first=False,
        prefix="Coordinates: Lon ",
    ).add_to(m)
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    m.save(output_html_path)
    file_size_kb = os.path.getsize(output_html_path) / 1024.0
    print(
        f"💾 Multi-Filter Density Heatmap exported to: {output_html_path} ({file_size_kb:.1f} KB / {file_size_kb / 1024.0:.2f} MB)"
    )
    if loaded_from_file:
        del df
        gc.collect()
    return m


if __name__ == "__main__":
    parquet_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PARQUET
    output_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_HTML

    if not os.path.exists(parquet_path):
        sys.exit(f"❌ Input parquet file not found: {parquet_path}")

    print(f"📂 Loading trajectory points from: {parquet_path}")
    desired_cols = [
        "latitude",
        "longitude",
        "altitude_tier",
        "ground_noise_dba",
        "aircraft_type_group",
        "timestamp",
    ]
    try:
        df_data = pd.read_parquet(parquet_path, columns=desired_cols)
    except Exception:  # noqa: BLE001
        df_data = pd.read_parquet(parquet_path)
    if "latitude" in df_data.columns:
        df_data["latitude"] = df_data["latitude"].astype(np.float32)
    if "longitude" in df_data.columns:
        df_data["longitude"] = df_data["longitude"].astype(np.float32)
    if "altitude_tier" in df_data.columns:
        df_data["altitude_tier"] = df_data["altitude_tier"].astype("category")
    if "aircraft_type_group" in df_data.columns:
        df_data["aircraft_type_group"] = df_data["aircraft_type_group"].astype(
            "category"
        )
    print(
        f"📊 Loaded {len(df_data):,} records ({df_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB in RAM)"
    )

    build_public_density_map(
        df=df_data,
        lat_min=DEFAULT_LAT_MIN,
        lat_max=DEFAULT_LAT_MAX,
        lon_min=DEFAULT_LON_MIN,
        lon_max=DEFAULT_LON_MAX,
        output_html_path=output_path,
    )
    print("🎉 Done! Successfully generated multi-filter density map.")
