"""
Analog-year detection.

Strategy
--------
1. Fetch ERA5 monthly means for ALL years (1950–present) in one batch request
   → much faster than year-by-year downloads.
2. Compute the area-weighted pattern correlation between the target-period mean
   and the same calendar months in every historical year.
3. Return the N best-matching years ranked by correlation.

For sub-monthly ("custom") periods, daily ERA5 data is used for the target
while monthly means are used for the analog search (good enough for ranking).
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
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

        Parameters
        ----------
        var_key : variable key (e.g. 'z500')
        target_start / target_end : date bounds of the target period
        region : {'lat': (lat_min, lat_max), 'lon': (lon_min, lon_max)}
        n_analogs : how many analogs to return
        search_years : (first_year, last_year) to search through
        clim_years : climatology period used when use_anomaly=True
        use_anomaly : compare anomaly fields (recommended; removes seasonal cycle)

        Returns
        -------
        list of dicts with keys: year, correlation, rmse, start_date, end_date
        """
        months = self._months_in_range(target_start, target_end)
        all_years = list(range(search_years[0], search_years[1] + 1))

        # ── 1. Fetch all monthly data in one request ──────────────────
        ds_all = self.fetcher.fetch_monthly(var_key, all_years, months)
        da_all = self.fetcher.extract(ds_all, var_key)  # (time, lat, lon)

        # ── 2. Compute climatology ─────────────────────────────────────
        if use_anomaly:
            clim_years_list = list(range(clim_years[0], clim_years[1] + 1))
            ds_clim = self.fetcher.fetch_monthly(var_key, clim_years_list, months)
            da_clim = self.fetcher.extract(ds_clim, var_key)
            climatology = da_clim.mean(dim="time")
        else:
            climatology = None

        # ── 3. Target mean ────────────────────────────────────────────
        target_mean = self._target_mean(
            var_key, target_start, target_end, months, da_all
        )
        if climatology is not None:
            target_field = target_mean - climatology
        else:
            target_field = target_mean

        target_crop = self._crop(target_field, region)
        target_vec, weights = self._flatten_weighted(target_crop)

        # ── 4. Score each historical year ─────────────────────────────
        scores: list[dict] = []
        time_coords = pd.DatetimeIndex(da_all.coords["time"].values)

        for year in all_years:
            # skip the year(s) that contain the target period
            if target_start.year <= year <= target_end.year:
                continue

            # average the requested months for this year
            year_mask = time_coords.year == year
            if year_mask.sum() == 0:
                continue

            analog_mean = da_all.isel(time=year_mask).mean(dim="time")
            if climatology is not None:
                analog_field = analog_mean - climatology
            else:
                analog_field = analog_mean

            analog_crop = self._crop(analog_field, region)
            analog_vec, _ = self._flatten_weighted(analog_crop)

            valid = np.isfinite(target_vec) & np.isfinite(analog_vec)
            if valid.sum() < 20:
                continue

            r, _ = pearsonr(target_vec[valid], analog_vec[valid])
            rmse = float(
                np.sqrt(np.mean((target_vec[valid] - analog_vec[valid]) ** 2))
            )

            try:
                y_start = target_start.replace(year=year)
                y_end = target_end.replace(year=year)
            except ValueError:  # Feb 29 in non-leap year
                y_start = target_start.replace(year=year, day=28)
                y_end = target_end.replace(year=year, day=28)

            scores.append(
                {
                    "year": year,
                    "correlation": round(float(r), 4),
                    "rmse": round(rmse, 2),
                    "start_date": y_start,
                    "end_date": y_end,
                }
            )

        scores.sort(key=lambda x: -x["correlation"])
        return scores[:n_analogs]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _months_in_range(start: date, end: date) -> list[int]:
        return sorted(
            {d.month for d in pd.date_range(str(start), str(end), freq="D")}
        )

    def _target_mean(
        self,
        var_key: str,
        start: date,
        end: date,
        months: list[int],
        da_all: xr.DataArray,
    ) -> xr.DataArray:
        """
        Compute the mean field for the target period.
        Uses daily data when the target period is recent (< 60 days ago);
        falls back to monthly means otherwise.
        """
        days_ago = (date.today() - end).days
        period_days = (end - start).days

        if days_ago < 60 or period_days < 20:
            # Use daily ERA5(T) for recent / sub-monthly periods
            try:
                ds_daily = self.fetcher.fetch_daily(var_key, start, end)
                return self.fetcher.extract(ds_daily, var_key).mean(dim="time")
            except Exception:
                pass  # fall back to monthly below

        # Use monthly means
        time_coords = pd.DatetimeIndex(da_all.coords["time"].values)
        mask = (
            (time_coords.year >= start.year)
            & (time_coords.year <= end.year)
            & (time_coords.month.isin(months))
        )
        if mask.sum() == 0:
            # Target year may not be in the search-year batch; fetch separately
            ds = self.fetcher.fetch_monthly(
                var_key, list(range(start.year, end.year + 1)), months
            )
            return self.fetcher.extract(ds, var_key).mean(dim="time")
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

        # Broadcast weights to 2-D (lat × lon)
        weights_2d = np.outer(weights_1d, np.ones(da.shape[-1]))
        weights_flat = weights_2d.flatten()
        weights_flat /= weights_flat.sum()

        # Apply sqrt(weight) so that pearsonr gives weighted correlation
        vals = da.values
        if vals.shape != weights_2d.shape:
            vals = vals.T
        w_sqrt = np.sqrt(weights_flat)
        return vals.flatten() * w_sqrt, weights_flat
