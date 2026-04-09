"""
Map plotting using Matplotlib + Cartopy.
"""

from __future__ import annotations

import io
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr

from .era5_fetcher import VARIABLES

# ---------------------------------------------------------------------------
# Region presets
# ---------------------------------------------------------------------------
REGIONS: dict[str, dict] = {
    "Northern Hemisphere": {
        "lat": (20, 90),
        "lon": (-180, 180),
        "projection": "stereo_north",
    },
    "Global": {
        "lat": (-90, 90),
        "lon": (-180, 180),
        "projection": "robinson",
    },
    "North Pacific / North America": {
        "lat": (15, 75),
        "lon": (120, 300),
        "projection": "platecarree",
    },
    "North Atlantic / Europe": {
        "lat": (20, 80),
        "lon": (-90, 50),
        "projection": "platecarree",
    },
    "Asia / Japan": {
        "lat": (10, 65),
        "lon": (80, 165),
        "projection": "platecarree",
    },
    "Japan (close-up)": {
        "lat": (22, 48),
        "lon": (120, 150),
        "projection": "platecarree",
    },
    "Tropics": {
        "lat": (-30, 30),
        "lon": (-180, 180),
        "projection": "platecarree",
    },
}


def _make_projection(region: dict) -> ccrs.Projection:
    proj_type = region.get("projection", "platecarree")
    lat = region["lat"]
    lon = region["lon"]
    central_lon = (lon[0] + lon[1]) / 2 % 360
    if central_lon > 180:
        central_lon -= 360

    if proj_type == "stereo_north":
        return ccrs.NorthPolarStereo(central_longitude=central_lon)
    elif proj_type == "stereo_south":
        return ccrs.SouthPolarStereo(central_longitude=central_lon)
    elif proj_type == "robinson":
        return ccrs.Robinson(central_longitude=central_lon)
    else:
        return ccrs.PlateCarree(central_longitude=0)


class WeatherMapPlotter:
    """Generate weather composite maps."""

    def plot_map(
        self,
        data: xr.DataArray,
        var_key: str,
        region: dict,
        title: str = "",
        plot_type: str = "Mean",
        figsize: tuple[float, float] = (11, 6.5),
        wind_uv: tuple[xr.DataArray, xr.DataArray] | None = None,
    ) -> plt.Figure:
        """
        Parameters
        ----------
        data : DataArray with dims (latitude/lat, longitude/lon)
        var_key : key in VARIABLES dict
        region : dict with keys 'lat', 'lon', optionally 'projection'
        plot_type : one of 'Mean', 'Anomaly', 'Standardized Anomaly'
        """
        var_info = VARIABLES[var_key]
        is_anomaly = "Anomaly" in plot_type

        cmap = var_info["cmap_anom"] if is_anomaly else var_info["cmap_mean"]
        vmin, vmax = (
            var_info["clim_range"] if is_anomaly else var_info["typical_range"]
        )

        proj = _make_projection(region)
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})

        # ------------------------------------------------------------------
        # Normalise coordinate names
        # ------------------------------------------------------------------
        lat_name = "latitude" if "latitude" in data.coords else "lat"
        lon_name = "longitude" if "longitude" in data.coords else "lon"

        lats = data.coords[lat_name].values
        lons = data.coords[lon_name].values
        vals = data.values

        # Make lons run -180..180
        if lons.max() > 180:
            shift = lons > 180
            lons[shift] -= 360
            # Re-sort along lon axis
            sort_idx = np.argsort(lons)
            lons = lons[sort_idx]
            lat_ax = 0 if vals.shape[0] == len(lats) else 1
            if lat_ax == 0:
                vals = vals[:, sort_idx]
            else:
                vals = vals[sort_idx, :]

        # Subset to region
        lat_min, lat_max = sorted(region["lat"])
        lon_min, lon_max = sorted(region["lon"])
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        lon_mask = (lons >= lon_min) & (lons <= lon_max)

        lats_sub = lats[lat_mask]
        lons_sub = lons[lon_mask]

        if vals.shape == (len(lats), len(lons)):
            vals_sub = vals[np.ix_(lat_mask, lon_mask)]
        elif vals.shape == (len(lons), len(lats)):
            vals_sub = vals[np.ix_(lon_mask, lat_mask)].T
        else:
            vals_sub = vals

        # ------------------------------------------------------------------
        # Filled contour
        # ------------------------------------------------------------------
        levels_fill = np.linspace(vmin, vmax, 21)
        cf = ax.contourf(
            lons_sub,
            lats_sub,
            vals_sub,
            levels=levels_fill,
            cmap=cmap,
            extend="both",
            transform=ccrs.PlateCarree(),
        )

        # Contour lines (absolute values only; skip for anomalies for clarity)
        if not is_anomaly:
            ci = var_info["contour_interval"]
            cl_levels = np.arange(
                np.floor(vmin / ci) * ci,
                np.ceil(vmax / ci) * ci + ci,
                ci,
            )
            cl = ax.contour(
                lons_sub,
                lats_sub,
                vals_sub,
                levels=cl_levels,
                colors="k",
                linewidths=0.6,
                transform=ccrs.PlateCarree(),
            )
            ax.clabel(cl, fmt="%g", fontsize=7, inline=True)
        else:
            # Zero contour for anomaly maps
            ax.contour(
                lons_sub,
                lats_sub,
                vals_sub,
                levels=[0],
                colors="k",
                linewidths=1.0,
                linestyles="--",
                transform=ccrs.PlateCarree(),
            )

        # ------------------------------------------------------------------
        # Wind vectors (quiver) — only when wind_uv is supplied
        # ------------------------------------------------------------------
        if wind_uv is not None:
            u_da, v_da = wind_uv
            u_lat = "latitude" if "latitude" in u_da.coords else "lat"
            u_lon = "longitude" if "longitude" in u_da.coords else "lon"
            u_lats = u_da.coords[u_lat].values
            u_lons = u_da.coords[u_lon].values
            u_vals = u_da.values
            v_vals = v_da.values

            # Make lons -180..180
            if u_lons.max() > 180:
                shift = u_lons > 180
                u_lons = u_lons.copy()
                u_lons[shift] -= 360
                sort_idx = np.argsort(u_lons)
                u_lons = u_lons[sort_idx]
                if u_vals.shape == (len(u_lats), len(u_lons)):
                    u_vals = u_vals[:, sort_idx]
                    v_vals = v_vals[:, sort_idx]

            # Subset to region
            u_lat_mask = (u_lats >= lat_min) & (u_lats <= lat_max)
            u_lon_mask = (u_lons >= lon_min) & (u_lons <= lon_max)
            u_lats_s = u_lats[u_lat_mask]
            u_lons_s = u_lons[u_lon_mask]
            if u_vals.shape == (len(u_lats), len(u_lons)):
                u_sub = u_vals[np.ix_(u_lat_mask, u_lon_mask)]
                v_sub = v_vals[np.ix_(u_lat_mask, u_lon_mask)]
            else:
                u_sub, v_sub = u_vals, v_vals

            # Subsample to avoid clutter (every ~5° → stride ≈ 2 for 2.5° grid)
            stride = max(1, len(u_lats_s) // 20)
            ax.quiver(
                u_lons_s[::stride],
                u_lats_s[::stride],
                u_sub[::stride, ::stride],
                v_sub[::stride, ::stride],
                transform=ccrs.PlateCarree(),
                scale=200,
                width=0.003,
                color="k",
                alpha=0.7,
                zorder=6,
            )

        # ------------------------------------------------------------------
        # Map features
        # ------------------------------------------------------------------
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=5)
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical", "land", "110m",
                facecolor="none", edgecolor="none"
            ),
            zorder=4,
        )

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.4,
            alpha=0.6,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False

        # Extent
        try:
            ax.set_extent(
                [lon_min, lon_max, lat_min, lat_max],
                crs=ccrs.PlateCarree(),
            )
        except Exception:
            pass

        # Colorbar
        fig.colorbar(
            cf,
            ax=ax,
            orientation="horizontal",
            pad=0.04,
            shrink=0.75,
            label=f"{var_info['label']}  [{var_info['units']}]",
        )
        ax.set_title(title, fontsize=11, pad=8)
        fig.tight_layout()
        return fig

    def fig_to_bytes(self, fig: plt.Figure, dpi: int = 150) -> bytes:
        """Render figure to PNG bytes."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf.read()
