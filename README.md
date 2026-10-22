# OpenSky Flight Density & Heatmap Pipeline ✈️🗺️

A high-performance pipeline for fetching, caching, interpolating, and visualizing flight trajectories from the [OpenSky Network](https://opensky-network.org/).

🌐 **Live Demo**: [OpenSky Flight Density Map on GitHub Pages](https://bar-tolini.github.io/flight-heatmap/)

> [!NOTE]
> **Primary Component**: The **Google Colab / Jupyter Notebook** ([`opensky_flight_heatmap.ipynb`](./opensky_flight_heatmap.ipynb)) is the **main part of this project**. It contains the comprehensive, end-to-end analytical pipeline with trajectory interpolation, simplified acoustic ground noise estimation (inspired by ECAC Doc 29 / FAA AEDT principles), interactive time window sliders, and multi-criteria aircraft filtering.
>
> The standalone raster map (`generate_density_heatmap.py` / HTML export) is a **simplified version only**, designed to export lightweight, privacy-preserving, pre-rendered multi-zoom image overlays for quick browser viewing without needing the full notebook environment.

---

## 🌟 Key Features

- **Interactive Colab / Jupyter Notebook (Main Component)**: Complete analytical environment featuring automated bounding box cache discovery, trajectory waypoint smoothing, 6-tier altitude classification, acoustic ground noise impact modeling ($L_\text{ground}$ in dB(A)), and rich Folium/ipywidgets controls.
- **Privacy-Preserving Raster Maps (Simplified Version)**: Generates self-contained, lightweight HTML maps using pre-rendered multi-zoom `ImageOverlay` raster tiles. Zero raw GPS coordinates, timestamps, or aircraft identifiers are exposed in the client HTML.
- **Automated Data Ingestion**: Historical state vector collection via OpenSky Network `/states/all` with automatic OAuth2 token renewal and rate limit management.
- **Efficient Disk Caching**: Single-file per bounding box Parquet cache with predicate pushdown and query deduplication.
- **Interactive Multi-Criteria Filtering**: Dynamic logical AND filtering:
  - 🛫 **Altitude Tiers**: 0–500m, 500–1,000m, 1,000–2,000m, 2,000–3,000m, 3,000–5,000m, and 5,000m+.
  - 🔊 **Noise Exposure**: Filter by estimated ground noise (≥50 dB, ≥60 dB, ≥70 dB).
  - 🚁 **Aircraft Categories**: Airliners, Helicopters / Rotorcraft, General Aviation, etc.

---

## 📁 Repository Structure

```text
.
├── opensky_flight_heatmap.ipynb            # ★ Main Component: End-to-end interactive Colab / Jupyter notebook
├── generate_density_heatmap.py             # Simplified Version: Standalone raster density map generator
├── fetch_flight_data.py                    # OpenSky API fetcher with Parquet cache manager
├── run_periodic_fetch.sh                   # Automation daemon / cron wrapper
├── credentials.json.example                # Example template for OpenSky API credentials
├── .env.example                            # Example environment variables template
├── requirements.txt                        # Python dependencies
├── example_opensky_flight_density_map.html # Sample rendered multi-zoom density map
└── timelapse_flight_heatmap.html           # Sample animated flight timelapse map
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Python 3.9+ is recommended. Clone this repository and create a virtual environment:

```bash
git clone <repo-url> flight_heatmap
cd flight_heatmap

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials

An OpenSky Network account is required to authenticate and obtain higher API request quotas (still free).

You can configure credentials using either a `credentials.json` file or environment variables:

#### Option A: `credentials.json`
Copy the example file and fill in your credentials:
```bash
cp credentials.json.example credentials.json
```
Edit `credentials.json`:
```json
{
  "clientId": "your-client-id-here",
  "clientSecret": "your-client-secret-here"
}
```

#### Option B: Environment Variables
```bash
export OPENSKY_CLIENT_ID="your-client-id-here"
export OPENSKY_CLIENT_SECRET="your-client-secret-here"
```

*(If no credentials are provided, the fetcher will run in Anonymous Mode with basic daily request limits).*

---

## 🛠️ Usage

### Fetching Flight Data

Run a manual fetch cycle for a designated bounding box:

```bash
python fetch_flight_data.py \
  --lat-min 47.1102 --lat-max 47.4855 \
  --lon-min 8.3720  --lon-max 9.0478 \
  --lookback-min 50 \
  --step-sec 20
```

#### CLI Options:
- `--lat-min`, `--lat-max`, `--lon-min`, `--lon-max`: Coordinates for the target bounding box.
- `--lookback-min`: Fetch window duration in minutes (default: `50`).
- `--step-sec`: Sampling resolution in seconds (default: `20`).
- `--active-start-hour`, `--active-end-hour`: Local operating hours to fetch (e.g. `5` to `23` for 5 AM to 11 PM).
- `--cache-dir`: Directory for caching `.parquet` files (default: `./flight_data_cache`).

### Periodic Fetching (Cron or Daemon)

Use `run_periodic_fetch.sh` to run continuous background ingestion or hook it into `cron`:

```bash
# Run continuously in background (fetches every 15 minutes)
./run_periodic_fetch.sh

# Run once (ideal for cron jobs)
./run_periodic_fetch.sh --once
```

Example `crontab` entry (runs every 15 minutes):
```cron
*/15 * * * * /path/to/flight_heatmap/run_periodic_fetch.sh --once >> /path/to/flight_heatmap/cron_fetch.log 2>&1
```

### Interactive Colab / Jupyter Notebook (Main Component)

The **Google Colab / Jupyter Notebook** is the **main part of this project**, providing the complete end-to-end analytical and visualization pipeline:

```bash
jupyter notebook opensky_flight_heatmap.ipynb
```
*(You can also run or import `opensky_flight_heatmap.ipynb` directly in [Google Colab](https://colab.research.google.com/)).*

The notebook guides you through:
1. **Cache Loading & Auto-Discovery**: Inspecting bounding box coverage, time spans, and aircraft counts.
2. **Trajectory Interpolation & Smoothing**: Batched interpolation bridging discrete ADS-B state vector samples.
3. **Acoustic Noise Modeling**: Perceived ground sound level estimation ($L_\text{ground}$ in dB(A)) based on airframe type, vertical rate, and altitude.
4. **Interactive Multi-Filter Visualizations**: Dynamic Leaflet maps with altitude tiers, noise thresholds, and aircraft group selectors.

### Generating the Density Heatmap (Simplified Version)

The standalone raster map generator (`generate_density_heatmap.py`) is a **simplified version only**, intended to export pre-rendered multi-zoom raster images into a standalone, lightweight HTML file:

```bash
python generate_density_heatmap.py \
  --input processed_flight_paths.parquet \
  --output opensky_flight_density_map.html \
  --region-name "Zurich TMA"
```

Open `opensky_flight_density_map.html` in any modern web browser to view the interactive map without requiring a notebook environment.

---

## 🔒 Security & Privacy

- `credentials.json`, `.env`, and all credentials files are excluded by `.gitignore`.
- Raw trajectory data, database files (`*.parquet`, `*.csv`, `*.db`), and cache directories are excluded by `.gitignore`.
- Rendered maps utilize client-side raster image overlays to prevent leaking PII, registration numbers, or discrete point paths.

---

<a id="disclaimer"></a>
## ⚠️ Disclaimer

### General Use & Aviation Safety
This project and its generated outputs are provided for informational, visualization, and research purposes only. The author assumes no responsibility or liability for the correctness, completeness, or accuracy of the data, nor for any decisions affected by or actions taken based on it. Under no circumstances should this data or visualization be relied upon for flight planning, navigation, operational decision-making, or safety-critical applications.

### Acoustic Noise Modeling & Computation Method
The ground noise footprint calculations ($L_\text{ground}$ in dB(A)) implemented in the analytical notebook are based on a **simplified engineering approximation model** inspired by standard aviation acoustic principles (such as ECAC Doc 29 / FAA AEDT), but do **not** constitute certified acoustic simulations. Please note the following key limitations and assumptions:

- **Simplified Source Emission Model**: Sound levels at the source ($L_0$) are estimated using generalized airframe categories and typical engine types rather than full, aircraft-specific certified Noise-Power-Distance (NPD) tables or measured acoustic signatures.
- **Empirical Operational Adjustments**: Engine thrust corrections are heuristically inferred from kinematic vertical climb/descent rates ($\text{m/s}$), and speed corrections are approximated from ADS-B ground speed rather than actual engine pressure ratios (EPR), fan speeds ($N_1$), flap/gear deployment, or true airspeed.
- **Propagation Assumptions**: Sound propagation assumes unobstructed line-of-sight spherical geometric spreading loss ($20 \log_{10}(d / d_0)$) and a constant linear atmospheric absorption coefficient ($0.005 \cdot d$). It does not model atmospheric stratification, meteorological gradients (wind shear, temperature inversions), humidity profiles, or turbulence scattering.
- **Terrain & Directivity Omissions**: Calculations assume flat terrain without digital elevation models (DEM), ground reflections/impedance, shielding by buildings or topography, or lateral directivity patterns and airframe installation shielding (such as bank angle effects).
- **Informational & Comparative Purpose Only**: These computed values are intended solely for relative comparative analysis, trajectory filtering, and qualitative visualization. They **must not** be used for regulatory compliance, official Environmental Impact Assessments (EIA), airport noise contour certification, municipal zoning, or legal proceedings.

---

## 📄 License

This project is open-source under the MIT License.

