"""
Analog-year detection.

Strategy
--------
1. Fetch ERA5 monthly means for ALL years (1950-present) in one batch request.
2. Compute the area-weighted pattern correlation between the target-period mean
   and the same calendar months in every historical year.
3. Return the N best-matching years ranked by correlation.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr

from .era5_fetcher import ERA5Fetcher, VARIABLES


class AnalogFinder:
    def __init__(self, fetcher: ERA5Fetcher):
        self.fetcher = fetcher

    def find_analogs(
        self,
        var_key: str,
        target_start: date,
        target_end: date,
        region: dict,
        n_analogs: int = 5,
        search_years: tuple[int, int] = (1950, 2023),
        clim_years: tuple[int, int] = (1991, 2020),
        use_anomaly: bool = True,
    ) -> list[dict]:
        """
        Find historical analog years for the given target period.

        Returns list of dicts: year, correlation, rmse, start_date, end_date
        """
        months = self._months_in_range(target_start, target_end)
        all_years = list(range(search_years[0], search_years[1] + 1))

        # Fetch all monthly data in one request
        ds_all = self.fetcher.fetch_monthly(var_key, all_years, months)
        da_all = self.fetcher.extract(ds_all, var_key)

        # Climatology
        if use_anomaly:
            clim_years_list = list(range(clim_years[0], clim_years[1] + 1))
            ds_clim = self.fetcher.fetch_monthly(var_key, clim_years_list, months)
            da_clim = self.fetcher.extract(ds_clim, var_key)
            climatology = da_clim.mean(dim="time")
        else:
            climatology = None

        # Target field
        target_mean = self._target_mean(var_key, target_start, target_end, months, da_all)
        target_field = target_mean - climatology if climatology is not None else target_mean
        target_crop = self._crop(target_field, region)
        target_vec, _ = self._flatten_weighted(target_crop)

        # Score each historical year
        scores: list[dict] = []
        time_coords = pd.DatetimeIndex(da_all.coords["time"].values)

        for year in all_years:
            if target_start.year <= year <= target_end.year:
                continue

            year_mask = time_coords.year == year
            if year_mask.sum() == 0:
                continue

            analog_mean = da_all.isel(time=year_mask).mean(dim="time")
            analog_field = analog_mean - climatology if climatology is not None else analog_mean
            analog_crop = self._crop(analog_field, region)
            analog_vec, _ = self._flatten_weighted(analog_crop)

            valid = np.isfinite(target_vec) & np.isfinite(analog_vec)
            if valid.sum() < 20:
                continue

            r, _ = pearsonr(target_vec[valid], analog_vec[valid])
            rmse = float(np.sqrt(np.mean((target_vec[valid] - analog_vec[valid]) ** 2)))

            try:
                y_start = target_start.replace(year=year)
                y_end = target_end.replace(year=year)
            except ValueError:
                y_start = target_start.replace(year=year, day=28)
                y_end = target_end.replace(year=year, day=28)

            scores.append({
                "year": year,
                "correlation": round(float(r), 4),
                "rmse": round(rmse, 2),
                "start_date": y_start,
                "end_date": y_end,
            })

        scores.sort(key=lambda x: -x["correlation"])
        return scores[:n_analogs]

    @staticmethod
    def _months_in_range(start: date, end: date) -> list[int]:
        return sorted({d.month for d in pd.date_range(str(start), str(end), freq="D")})

    def _target_mean(
        self,
        var_key: str,
        start: date,
        end: date,
        months: list[int],
        da_all: xr.DataArray,
    ) -> xr.DataArray:
        days_ago = (date.today() - end).days
        period_days = (end - start).days

        if days_ago < 60 or period_days < 20:
            try:
                ds_daily = self.fetcher.fetch_daily(var_key, start, end)
                return self.fetcher.extract(ds_daily, var_key).mean(dim="time")
            except Exception:
                pass

        time_coords = pd.DatetimeIndex(da_all.coords["time"].values)
        mask = (
            (time_coords.year >= start.year)
            & (time_coords.year <= end.year)
            & (time_coords.month.isin(months))
        )
        if mask.sum() == 0:
            raise ValueError(f"No monthly data found for {start}-{end}.")
        return da_all.isel(time=mask).mean(dim="time")

    @staticmethod
    def _crop(da: xr.DataArray, region: dict) -> xr.DataArray:
        lat_name = "latitude" if "latitude" in da.coords else "lat"
        lon_name = "longitude" if "longitude" in da.coords else "lon"
        lats = da.coords[lat_name]
        lons = da.coords[lon_name]
        lat_min, lat_max = sorted(region["lat"])
        lon_min, lon_max = sorted(region["lon"])
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        lon_mask = (lons >= lon_min) & (lons <= lon_max)
        return da.isel({lat_name: lat_mask.values, lon_name: lon_mask.values})

    @staticmethod
    def _flatten_weighted(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
        lat_name = "latitude" if "latitude" in da.coords else "lat"
        lats = da.coords[lat_name].values
        weights_1d = np.cos(np.deg2rad(lats))
        weights_2d = np.outer(weights_1d, np.ones(da.shape[-1]))
        weights_flat = weights_2d.flatten()
        weights_flat /= weights_flat.sum()
        vals = da.values
        if vals.shape != weights_2d.shape:
            vals = vals.T
        w_sqrt = np.sqrt(weights_flat)
        return vals.flatten() * w_sqrt, weights_flat
