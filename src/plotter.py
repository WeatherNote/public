"""
Map plotting using Matplotlib + Cartopy.
"""

from __future__ import annotations

import io
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

try:
    from cartopy.util import add_cyclic_point
    _HAS_CYCLIC = True
except ImportError:
    _HAS_CYCLIC = False

from .era5_fetcher import VARIABLES

# ---------------------------------------------------------------------------
# Region presets
# ---------------------------------------------------------------------------
REGIONS: dict[str, dict] = {
    "Northern Hemisphere": {
        "lat": (20, 90),
        "lon": (-180, 180),
        "projection": "stereo_north",
        "central_longitude": 0,
    },
    "Northern Hemisphere (Japan)": {
        "lat": (20, 90),
        "lon": (-180, 180),
        "projection": "stereo_north",
        "central_longitude": 140,
    },
    "Northern Hemisphere (N. America)": {
        "lat": (20, 90),
        "lon": (-180, 180),
        "projection": "stereo_north",
        "central_longitude": -100,
    },
    "Global (0° center)": {
        "lat": (-90, 90),
        "lon": (-180, 180),
        "projection": "robinson",
        "central_longitude": 0,
    },
    "Global (180° center)": {
        "lat": (-90, 90),
        "lon": (-180, 180),
        "projection": "robinson",
        "central_longitude": 180,
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

COLORMAPS = [
    "RdBu_r", "RdBu", "coolwarm", "bwr",
    "RdYlBu_r", "RdYlBu", "seismic",
    "BrBG", "PRGn", "PiYG",
    "viridis", "plasma", "inferno", "magma",
    "YlOrRd", "OrRd", "BuGn", "BuPu",
    "jet", "turbo",
]


def _make_projection(region: dict) -> ccrs.Projection:
    proj_type = region.get("projection", "platecarree")

    # Use explicit central_longitude if supplied; otherwise derive from lon range
    if "central_longitude" in region:
        central_lon = region["central_longitude"]
    else:
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
        wind_uv: Optional[Tuple[xr.DataArray, xr.DataArray]] = None,
        # Display setting overrides
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        contour_interval: Optional[float] = None,
        cmap: Optional[str] = None,
        draw_labels: bool = True,
        # Mean overlay on anomaly maps
        mean_overlay: Optional[xr.DataArray] = None,
    ) -> plt.Figure:
        var_info = VARIABLES[var_key]
        is_anomaly = "Anomaly" in plot_type

        _cmap = cmap or (var_info["cmap_anom"] if is_anomaly else var_info["cmap_mean"])
        _vmin, _vmax = var_info["clim_range"] if is_anomaly else var_info["typical_range"]
        if vmin is not None:
            _vmin = vmin
        if vmax is not None:
            _vmax = vmax
        _ci = contour_interval or var_info["contour_interval"]

        proj = _make_projection(region)
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})

        # ------------------------------------------------------------------
        # Normalise coordinate names
        # ------------------------------------------------------------------
        lat_name = "latitude" if "latitude" in data.coords else "lat"
        lon_name = "longitude" if "longitude" in data.coords else "lon"

        lats = data.coords[lat_name].values.copy()
        lons = data.coords[lon_name].values.copy()
        vals = data.values.copy()

        # Make lons run -180..180
        if lons.max() > 180:
            shift = lons > 180
            lons[shift] -= 360
            sort_idx = np.argsort(lons)
            lons = lons[sort_idx]
            if vals.shape == (len(lats), len(lons)):
                vals = vals[:, sort_idx]
            else:
                vals = vals[sort_idx, :]

        # Determine if this is a full-globe plot (skip set_extent)
        lat_min, lat_max = sorted(region["lat"])
        lon_min, lon_max = sorted(region["lon"])
        full_globe = (lon_max - lon_min) >= 359 and (lat_max - lat_min) >= 179

        # Subset to region (skip for full-globe: keep all data for cyclic fix)
        if not full_globe:
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
        else:
            lats_sub = lats
            lons_sub = lons
            if vals.shape == (len(lons), len(lats)):
                vals_sub = vals.T
            else:
                vals_sub = vals

        # Add cyclic point to close the date-line gap
        if _HAS_CYCLIC and full_globe:
            try:
                vals_sub, lons_sub = add_cyclic_point(vals_sub, coord=lons_sub)
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Filled contour
        # ------------------------------------------------------------------
        levels_fill = np.linspace(_vmin, _vmax, 21)
        cf = ax.contourf(
            lons_sub,
            lats_sub,
            vals_sub,
            levels=levels_fill,
            cmap=_cmap,
            extend="both",
            transform=ccrs.PlateCarree(),
        )

        # Contour lines
        if not is_anomaly:
            cl_levels = np.arange(
                np.floor(_vmin / _ci) * _ci,
                np.ceil(_vmax / _ci) * _ci + _ci,
                _ci,
            )
            cl = ax.contour(
                lons_sub, lats_sub, vals_sub,
                levels=cl_levels,
                colors="k", linewidths=0.6,
                transform=ccrs.PlateCarree(),
            )
            ax.clabel(cl, fmt="%g", fontsize=7, inline=True)
        else:
            ax.contour(
                lons_sub, lats_sub, vals_sub,
                levels=[0],
                colors="k", linewidths=1.0, linestyles="--",
                transform=ccrs.PlateCarree(),
            )

        # ------------------------------------------------------------------
        # Mean overlay on anomaly maps (optional)
        # ------------------------------------------------------------------
        if is_anomaly and mean_overlay is not None:
            m_lat = "latitude" if "latitude" in mean_overlay.coords else "lat"
            m_lon = "longitude" if "longitude" in mean_overlay.coords else "lon"
            m_lats = mean_overlay.coords[m_lat].values.copy()
            m_lons = mean_overlay.coords[m_lon].values.copy()
            m_vals = mean_overlay.values.copy()
            if m_lons.max() > 180:
                shift = m_lons > 180
                m_lons[shift] -= 360
                sidx = np.argsort(m_lons)
                m_lons = m_lons[sidx]
                if m_vals.shape == (len(m_lats), len(m_lons)):
                    m_vals = m_vals[:, sidx]
            if not full_globe:
                mlm = (m_lats >= lat_min) & (m_lats <= lat_max)
                mll = (m_lons >= lon_min) & (m_lons <= lon_max)
                m_lats = m_lats[mlm]; m_lons = m_lons[mll]
                if m_vals.shape == (len(m_lats[~mlm]) + mlm.sum(), len(m_lons[~mll]) + mll.sum()):
                    m_vals = m_vals[np.ix_(mlm, mll)]
            if _HAS_CYCLIC and full_globe:
                try:
                    m_vals, m_lons = add_cyclic_point(m_vals, coord=m_lons)
                except Exception:
                    pass
            m_ci = _ci
            m_levels = np.arange(
                np.floor(m_vals.min() / m_ci) * m_ci,
                np.ceil(m_vals.max() / m_ci) * m_ci + m_ci,
                m_ci,
            )
            cl_m = ax.contour(
                m_lons, m_lats, m_vals,
                levels=m_levels,
                colors="k", linewidths=0.5, alpha=0.6,
                transform=ccrs.PlateCarree(),
            )
            ax.clabel(cl_m, fmt="%g", fontsize=6, inline=True)

        # ------------------------------------------------------------------
        # Wind vectors (quiver)
        # ------------------------------------------------------------------
        if wind_uv is not None:
            u_da, v_da = wind_uv
            u_lat = "latitude" if "latitude" in u_da.coords else "lat"
            u_lon = "longitude" if "longitude" in u_da.coords else "lon"
            u_lats = u_da.coords[u_lat].values.copy()
            u_lons = u_da.coords[u_lon].values.copy()
            u_vals = u_da.values.copy()
            v_vals = v_da.values.copy()

            if u_lons.max() > 180:
                shift = u_lons > 180
                u_lons[shift] -= 360
                sort_idx = np.argsort(u_lons)
                u_lons = u_lons[sort_idx]
                if u_vals.shape == (len(u_lats), len(u_lons)):
                    u_vals = u_vals[:, sort_idx]
                    v_vals = v_vals[:, sort_idx]

            u_lat_mask = (u_lats >= lat_min) & (u_lats <= lat_max)
            u_lon_mask = (u_lons >= lon_min) & (u_lons <= lon_max)
            u_lats_s = u_lats[u_lat_mask]
            u_lons_s = u_lons[u_lon_mask]
            if u_vals.shape == (len(u_lats), len(u_lons)):
                u_sub = u_vals[np.ix_(u_lat_mask, u_lon_mask)]
                v_sub = v_vals[np.ix_(u_lat_mask, u_lon_mask)]
            else:
                u_sub, v_sub = u_vals, v_vals

            stride = max(1, len(u_lats_s) // 20)
            ax.quiver(
                u_lons_s[::stride], u_lats_s[::stride],
                u_sub[::stride, ::stride], v_sub[::stride, ::stride],
                transform=ccrs.PlateCarree(),
                scale=200, width=0.003, color="k", alpha=0.7, zorder=6,
            )

        # ------------------------------------------------------------------
        # Map features
        # ------------------------------------------------------------------
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=5)

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=draw_labels,
            linewidth=0.4,
            alpha=0.6,
            linestyle="--",
        )
        if draw_labels:
            gl.top_labels = False
            gl.right_labels = False

        # Extent (skip for full-globe Robinson projections)
        if not full_globe:
            try:
                ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            except Exception:
                pass

        fig.colorbar(
            cf, ax=ax,
            orientation="horizontal",
            pad=0.04, shrink=0.75,
            label=f"{var_info['label']}  [{var_info['units']}]",
        )
        ax.set_title(title, fontsize=11, pad=8)
        fig.tight_layout()
        return fig

    def fig_to_bytes(self, fig: plt.Figure, dpi: int = 150) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf.read()
