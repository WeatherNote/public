"""
ERA5 data fetching via CDS API with local disk caching.

Requires ~/.cdsapirc with valid credentials.
See: https://cds.climate.copernicus.eu/how-to-api
"""

import hashlib
import json
from datetime import date
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

CACHE_DIR = Path("cache")

# ---------------------------------------------------------------------------
# Variable catalogue
# ---------------------------------------------------------------------------
VARIABLES: dict[str, dict] = {
    "z500": {
        "label": "500 hPa Geopotential Height",
        "era5_name": "geopotential",
        "era5_short": "z",
        "level": 500,
        "dataset_daily": "reanalysis-era5-pressure-levels",
        "dataset_monthly": "reanalysis-era5-pressure-levels-monthly-means",
        "product_type_monthly": "monthly_averaged_reanalysis",
        "scale": 1 / 9.80665,  # m²/s² → m
        "offset": 0.0,
        "units": "m",
        "cmap_mean": "RdBu_r",
        "cmap_anom": "RdBu_r",
        "contour_interval": 60,
        "clim_range": (-200, 200),
        "typical_range": (4800, 6000),
    },
    "slp": {
        "label": "Sea Level Pressure",
        "era5_name": "mean_sea_level_pressure",
        "era5_short": "msl",
        "level": None,
        "dataset_daily": "reanalysis-era5-single-levels",
        "dataset_monthly": "reanalysis-era5-single-levels-monthly-means",
        "product_type_monthly": "monthly_averaged_reanalysis",
        "scale": 0.01,  # Pa → hPa
        "offset": 0.0,
        "units": "hPa",
        "cmap_mean": "RdBu_r",
        "cmap_anom": "RdBu_r",
        "contour_interval": 4,
        "clim_range": (-10, 10),
        "typical_range": (970, 1040),
    },
    "t2m": {
        "label": "2 m Temperature",
        "era5_name": "2m_temperature",
        "era5_short": "t2m",
        "level": None,
        "dataset_daily": "reanalysis-era5-single-levels",
        "dataset_monthly": "reanalysis-era5-single-levels-monthly-means",
        "product_type_monthly": "monthly_averaged_reanalysis",
        "scale": 1.0,
        "offset": -273.15,  # K → °C
        "units": "°C",
        "cmap_mean": "coolwarm",
        "cmap_anom": "RdBu_r",
        "contour_interval": 4,
        "clim_range": (-6, 6),
        "typical_range": (-40, 40),
    },
    "t850": {
        "label": "850 hPa Temperature",
        "era5_name": "temperature",
        "era5_short": "t",
        "level": 850,
        "dataset_daily": "reanalysis-era5-pressure-levels",
        "dataset_monthly": "reanalysis-era5-pressure-levels-monthly-means",
        "product_type_monthly": "monthly_averaged_reanalysis",
        "scale": 1.0,
        "offset": -273.15,
        "units": "°C",
        "cmap_mean": "coolwarm",
        "cmap_anom": "RdBu_r",
        "contour_interval": 4,
        "clim_range": (-6, 6),
        "typical_range": (-40, 30),
    },
    "t500": {
        "label": "500 hPa Temperature",
        "era5_name": "temperature",
        "era5_short": "t",
        "level": 500,
        "dataset_daily": "reanalysis-era5-pressure-levels",
        "dataset_monthly": "reanalysis-era5-pressure-levels-monthly-means",
        "product_type_monthly": "monthly_averaged_reanalysis",
        "scale": 1.0,
        "offset": -273.15,
        "units": "°C",
        "cmap_mean": "coolwarm",
        "cmap_anom": "RdBu_r",
        "contour_interval": 4,
        "clim_range": (-6, 6),
        "typical_range": (-40, 10),
    },
    "u200": {
        "label": "200 hPa Zonal Wind",
        "era5_name": "u_component_of_wind",
        "era5_short": "u",
        "level": 200,
        "dataset_daily": "reanalysis-era5-pressure-levels",
        "dataset_monthly": "reanalysis-era5-pressure-levels-monthly-means",
        "product_type_monthly": "monthly_averaged_reanalysis",
        "scale": 1.0,
        "offset": 0.0,
        "units": "m/s",
        "cmap_mean": "YlOrRd",
        "cmap_anom": "RdBu_r",
        "contour_interval": 10,
        "clim_range": (-15, 15),
        "typical_range": (0, 80),
    },
    "u500": {
        "label": "500 hPa Zonal Wind",
        "era5_name": "u_component_of_wind",
        "era5_short": "u",
        "level": 500,
        "dataset_daily": "reanalysis-era5-pressure-levels",
        "dataset_monthly": "reanalysis-era5-pressure-levels-monthly-means",
        "product_type_monthly": "monthly_averaged_reanalysis",
        "scale": 1.0,
        "offset": 0.0,
        "units": "m/s",
        "cmap_mean": "RdBu_r",
        "cmap_anom": "RdBu_r",
        "contour_interval": 5,
        "clim_range": (-10, 10),
        "typical_range": (-30, 50),
    },
    "tp": {
        "label": "Total Precipitation",
        "era5_name": "total_precipitation",
        "era5_short": "tp",
        "level": None,
        "dataset_daily": "reanalysis-era5-single-levels",
        "dataset_monthly": "reanalysis-era5-single-levels-monthly-means",
        "product_type_monthly": "monthly_averaged_reanalysis",
        "scale": 1000.0,  # m → mm (monthly total)
        "offset": 0.0,
        "units": "mm/month",
        "cmap_mean": "BuGn",
        "cmap_anom": "BrBG",
        "contour_interval": 20,
        "clim_range": (-60, 60),
        "typical_range": (0, 300),
    },
    "w500": {
        "label": "500 hPa Vertical Velocity (ω)",
        "era5_name": "vertical_velocity",
        "era5_short": "w",
        "level": 500,
        "dataset_daily": "reanalysis-era5-pressure-levels",
        "dataset_monthly": "reanalysis-era5-pressure-levels-monthly-means",
        "product_type_monthly": "monthly_averaged_reanalysis",
        "scale": 1.0,
        "offset": 0.0,
        "units": "Pa/s",
        "cmap_mean": "RdBu",
        "cmap_anom": "RdBu",
        "contour_interval": 0.05,
        "clim_range": (-0.15, 0.15),
        "typical_range": (-0.5, 0.5),
    },
    "sst": {
        "label": "Sea Surface Temperature",
        "era5_name": "sea_surface_temperature",
        "era5_short": "sst",
        "level": None,
        "dataset_daily": "reanalysis-era5-single-levels",
        "dataset_monthly": "reanalysis-era5-single-levels-monthly-means",
        "product_type_monthly": "monthly_averaged_reanalysis",
        "scale": 1.0,
        "offset": -273.15,  # K → °C
        "units": "°C",
        "cmap_mean": "coolwarm",
        "cmap_anom": "RdBu_r",
        "contour_interval": 2,
        "clim_range": (-3, 3),
        "typical_range": (-2, 32),
    },
}

# Short name lookup built automatically
_ERA5_SHORT_TO_VAR = {v["era5_short"]: v["era5_name"] for v in VARIABLES.values()}


class ERA5Fetcher:
    """Fetch ERA5 data from CDS API with local caching."""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client: cdsapi.Client | None = None

    # ------------------------------------------------------------------
    # CDS client (lazy init)
    # ------------------------------------------------------------------
    @property
    def client(self) -> cdsapi.Client:
        if self._client is None:
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "cds" in st.secrets:
                    self._client = cdsapi.Client(
                        url=st.secrets["cds"]["url"],
                        key=st.secrets["cds"]["key"],
                        quiet=True,
                    )
                    return self._client
            except Exception:
                pass
            self._client = cdsapi.Client(quiet=True)
        return self._client

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _cache_key(self, **kwargs) -> str:
        s = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()[:14]

    def _cache_path(self, tag: str, **kwargs) -> Path:
        return self.cache_dir / f"{tag}_{self._cache_key(**kwargs)}.nc"

    @staticmethod
    def _normalize_time(ds: xr.Dataset) -> xr.Dataset:
        """Rename valid_time → time (new CDS API uses valid_time)."""
        if "valid_time" in ds.dims and "time" not in ds.dims:
            ds = ds.rename({"valid_time": "time"})
        elif "valid_time" in ds.coords and "time" not in ds.coords:
            ds = ds.rename({"valid_time": "time"})
        return ds

    # ------------------------------------------------------------------
    # Monthly means  (efficient batch download)
    # ------------------------------------------------------------------
    def fetch_monthly(
        self,
        var_key: str,
        years: list[int],
        months: list[int],
    ) -> xr.Dataset:
        """Return monthly-mean Dataset for requested years × months."""
        var_info = VARIABLES[var_key]
        cache_path = self._cache_path(
            f"{var_key}_mon", years=sorted(years), months=sorted(months)
        )

        if not cache_path.exists():
            req: dict = {
                "product_type": var_info["product_type_monthly"],
                "variable": var_info["era5_name"],
                "year": [str(y) for y in sorted(set(years))],
                "month": [f"{m:02d}" for m in sorted(set(months))],
                "time": "00:00",
                "format": "netcdf",
                "grid": "2.5/2.5",
            }
            if var_info["level"] is not None:
                req["pressure_level"] = str(var_info["level"])

            self.client.retrieve(var_info["dataset_monthly"], req, str(cache_path))

        return self._normalize_time(xr.open_dataset(cache_path))

    # ------------------------------------------------------------------
    # Daily means  (for recent / sub-monthly periods)
    # ------------------------------------------------------------------
    def fetch_daily(
        self,
        var_key: str,
        start_date: date,
        end_date: date,
    ) -> xr.Dataset:
        """
        Download 6-hourly ERA5(T) data and resample to daily means.
        Automatically includes ERA5T preliminary back-extension for recent dates.
        """
        var_info = VARIABLES[var_key]
        date_range = pd.date_range(str(start_date), str(end_date))
        years = sorted({int(d.year) for d in date_range})
        months = sorted({int(d.month) for d in date_range})
        days = sorted({int(d.day) for d in date_range})

        cache_path = self._cache_path(
            f"{var_key}_6h", years=years, months=months, days=days
        )

        if not cache_path.exists():
            req: dict = {
                "product_type": "reanalysis",
                "variable": var_info["era5_name"],
                "year": [str(y) for y in years],
                "month": [f"{m:02d}" for m in months],
                "day": [f"{d:02d}" for d in days],
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "format": "netcdf",
                "grid": "2.5/2.5",
            }
            if var_info["level"] is not None:
                req["pressure_level"] = str(var_info["level"])

            self.client.retrieve(var_info["dataset_daily"], req, str(cache_path))

        ds = self._normalize_time(xr.open_dataset(cache_path))
        # Trim to the exact requested dates and compute daily mean
        ds = ds.sel(time=slice(str(start_date), str(end_date)))
        ds = ds.resample(time="1D").mean()
        return ds

    # ------------------------------------------------------------------
    # Climatology helpers
    # ------------------------------------------------------------------
    def get_climatology(
        self,
        var_key: str,
        months: list[int],
        clim_years: tuple[int, int] = (1991, 2020),
    ) -> xr.DataArray:
        """Climatological mean over clim_years for the requested months."""
        years = list(range(clim_years[0], clim_years[1] + 1))
        ds = self.fetch_monthly(var_key, years, months)
        return self.extract(ds, var_key).mean(dim="time")

    def get_climatology_std(
        self,
        var_key: str,
        months: list[int],
        clim_years: tuple[int, int] = (1991, 2020),
    ) -> xr.DataArray:
        """Climatological std-dev over clim_years for the requested months."""
        years = list(range(clim_years[0], clim_years[1] + 1))
        ds = self.fetch_monthly(var_key, years, months)
        return self.extract(ds, var_key).std(dim="time")

    # ------------------------------------------------------------------
    # Variable extraction & unit conversion
    # ------------------------------------------------------------------
    def extract(self, ds: xr.Dataset, var_key: str) -> xr.DataArray:
        """Extract the target variable from a Dataset and apply unit conversion."""
        var_info = VARIABLES[var_key]
        era5_short = var_info["era5_short"]
        era5_long = var_info["era5_name"]

        # ERA5 files may use either the short name or the long name
        candidates = [era5_short, era5_long] + list(ds.data_vars)
        da: xr.DataArray | None = None
        for name in candidates:
            if name in ds:
                da = ds[name]
                break
        if da is None:
            raise KeyError(f"Cannot find variable for {var_key} in dataset. "
                           f"Available: {list(ds.data_vars)}")

        da = da * var_info["scale"] + var_info["offset"]

        # Drop pressure-level dimension if present
        for dim_name in ("level", "pressure_level"):
            if dim_name in da.dims and var_info["level"] is not None:
                da = da.sel({dim_name: var_info["level"]})
                break

        # Squeeze any leftover size-1 dimensions
        # Squeeze only non-time size-1 dimensions (e.g. leftover level dim)
        squeeze_dims = [d for d in da.dims if d != "time" and da.sizes[d] == 1]
        if squeeze_dims:
            da = da.squeeze(squeeze_dims, drop=True)
        return da
