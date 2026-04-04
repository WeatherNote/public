# Analog Year Weather Maps

ERA5-based weather composite and analog-year map tool.
Inspired by [NOAA PSL Composites](https://psl.noaa.gov/cgi-bin/data/composites/printpage.pl), but using [ERA5 reanalysis](https://cds.climate.copernicus.eu/) data that is updated within **~5 days** of real-time (vs. ~2 weeks for NOAA PSL).

---

## Features

| Feature | Details |
|---------|----------|
| **Variables** | Z500, SLP, T2m, T850, T500, U200, U500, Total Precipitation, \u03c9500 |
| **Map types** | Mean / Anomaly / Standardized Anomaly |
| **Regions** | Northern Hemisphere, Global, N. Pacific, N. Atlantic, Asia/Japan, Tropics, \u2026 |
| **Composite Map** | Plot the mean field for any date range |
| **Analog Year Finder** | Find the N historical years whose large-scale pattern best matches the target period (area-weighted pattern correlation) |
| **Climatology** | Selectable: 1951\u201380, 1961\u201390, 1971\u20132000, 1981\u20132010, 1991\u20132020 |
| **Caching** | Downloaded ERA5 files are cached locally so repeat requests are instant |

---

## Quick Start

### 1. Set up CDS API credentials

Register at <https://cds.climate.copernicus.eu/> and copy your API key:

```bash
cp .cdsapirc.example ~/.cdsapirc
# Edit ~/.cdsapirc and paste your API key
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** Cartopy requires system libraries.
> macOS: `brew install proj geos`
> Ubuntu/Debian: `sudo apt install libproj-dev libgeos-dev`

### 3. Run the app

```bash
streamlit run app.py
```

Open <http://localhost:8501> in your browser.

---

## How it Works

### Composite Map tab

1. Select a variable, region, and date range.
2. The app fetches ERA5 data:
   - **Recent periods (< 60 days ago) or short periods**: uses 6-hourly ERA5T daily data.
   - **Older / longer periods**: uses ERA5 monthly means (faster).
3. The mean field is plotted. If an anomaly type is selected, the climatological mean is subtracted.

### Analog Year Finder tab

1. Select a target period (e.g., the past 30 days).
2. The app downloads **all ERA5 monthly means from 1950 to present in a single batch request** and computes the area-weighted pattern correlation between the target period and the same calendar months in every historical year.
3. The top-N analog years are ranked by correlation and displayed with their composite maps.

---

## Data Source

- **ERA5** and **ERA5T (preliminary back extension)**: ECMWF via the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).
- ERA5 monthly means: latency ~2 months.
- ERA5T daily data: latency ~5 days.

---

## Project Structure

```
analog-year-weather-maps/
\u251c\u2500\u2500 app.py                  # Streamlit application entry point
\u251c\u2500\u2500 src/
\u2502   \u251c\u2500\u2500 era5_fetcher.py     # CDS API data download & caching
\u2502   \u251c\u2500\u2500 plotter.py          # Cartopy map generation
\u2502   \u2514\u2500\u2500 analog_finder.py    # Pattern-correlation analog search
\u251c\u2500\u2500 cache/                  # Auto-created; stores downloaded .nc files
\u251c\u2500\u2500 requirements.txt
\u251c\u2500\u2500 .cdsapirc.example
\u2514\u2500\u2500 .gitignore
```

---

## License

MIT
