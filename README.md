---
title: Analog Year Weather Maps
emoji: 🌏
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Analog Year Weather Maps

ERA5 reanalysis-based tool to find historical years with similar atmospheric patterns ("analog years") and generate composite weather maps.

## Features

- **Composite maps** for any date range using ERA5 / ERA5T data (~5-day lag)
- **Analog year finder** — ranks past years by area-weighted pattern correlation
- Variables: Z500, SLP, T2m, T850, T500, U200, U500, precipitation, ω500, SST, 10 m wind speed, OLR
- Regions: Northern Hemisphere, Global, North Pacific, North Atlantic, Asia/Japan, and more
- Anomaly and standardized anomaly maps (selectable climatology period)
- Wind vector overlay for 10 m wind speed maps

## Quick Start

### 1. Clone

```bash
git clone https://github.com/WeatherNote/public.git
cd public
```

### 2. Install dependencies

Requires Python 3.10+. Using a virtual environment is recommended.

```bash
pip install -r requirements.txt
```

> **Note (macOS/Linux):** If `cartopy` fails to install via pip, use conda:
> ```bash
> conda install -c conda-forge cartopy
> pip install -r requirements.txt
> ```

### 3. Set up CDS API credentials

Register at [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/) and obtain your API key.

Create `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: <your-api-key>
```

Or copy the example file:

```bash
cp .cdsapirc.example ~/.cdsapirc
# then edit ~/.cdsapirc and fill in your key
```

### 4. Run

```bash
python main.py
```

Open **http://localhost:8000** in your browser.

## Notes

- ERA5 data is cached in `./cache/` after the first download.
- First-time downloads can take several minutes depending on the date range and variable.
- Data availability: ERA5 covers 1940–present with ~5-day lag for the most recent period (ERA5T).

## Data Source

[ERA5 global reanalysis](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) — Copernicus Climate Change Service (C3S)
