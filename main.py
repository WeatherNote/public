"""
Analog Year Weather Maps — FastAPI backend
==========================================
Run:
    python main.py
Then open http://localhost:8000
"""
from __future__ import annotations

import base64
import json
from datetime import date, timedelta
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.era5_fetcher import ERA5Fetcher, VARIABLES
from src.plotter import WeatherMapPlotter, REGIONS, COLORMAPS
from src.analog_finder import AnalogFinder

app = FastAPI(title="Analog Year Weather Maps")
fetcher = ERA5Fetcher()
plotter = WeatherMapPlotter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_b64(fig) -> str:
    img = plotter.fig_to_bytes(fig, dpi=130)
    return "data:image/png;base64," + base64.b64encode(img).decode()


def _get_wind_uv(ds, var_key: str):
    if var_key != "ws10":
        return None
    try:
        u, v = fetcher.extract_wind_components(ds)
        if "time" in u.dims:
            u = u.mean(dim="time")
            v = v.mean(dim="time")
        return u, v
    except Exception:
        return None


def _apply_anomaly(field, var_key, months, clim_start, clim_end, plot_type):
    clim = fetcher.get_climatology(var_key, months, (clim_start, clim_end))
    anom = field - clim
    if "Standardized" in plot_type:
        std = fetcher.get_climatology_std(var_key, months, (clim_start, clim_end))
        anom = anom / std.where(std > 0)
    return anom


def _make_map(field, wind_uv, req_dict: dict, region: dict, title: str):
    """Render a map using display settings from request dict."""
    return plotter.plot_map(
        data=field,
        var_key=req_dict["var_key"],
        region=region,
        title=title,
        plot_type=req_dict["plot_type"],
        wind_uv=wind_uv,
        vmin=req_dict.get("disp_vmin"),
        vmax=req_dict.get("disp_vmax"),
        contour_interval=req_dict.get("disp_ci") or None,
        cmap=req_dict.get("disp_cmap") or None,
        draw_labels=req_dict.get("disp_labels", True),
    )


# ---------------------------------------------------------------------------
# Metadata endpoints
# ---------------------------------------------------------------------------

@app.get("/api/variables")
def get_variables():
    return {
        k: {"label": v["label"], "units": v["units"]}
        for k, v in VARIABLES.items()
    }


@app.get("/api/regions")
def get_regions():
    return list(REGIONS.keys())


@app.get("/api/colormaps")
def get_colormaps():
    return COLORMAPS


@app.get("/api/status")
def data_status():
    """Return estimated latest available ERA5 date."""
    latest = date.today() - timedelta(days=5)
    return {
        "latest_available": str(latest),
        "lag_days": 5,
        "message": f"ERA5 data available up to approx. {latest} (~5-day lag)",
    }


@app.post("/api/fetch_latest")
def fetch_latest():
    """Pre-fetch the latest available daily data for a common variable (z500)."""
    try:
        latest = date.today() - timedelta(days=5)
        start = latest - timedelta(days=6)
        fetcher.fetch_daily("z500", start, latest)
        return {"ok": True, "fetched_through": str(latest)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Display settings model (shared by composite + analog)
# ---------------------------------------------------------------------------

class DisplaySettings(BaseModel):
    disp_vmin: Optional[float] = None
    disp_vmax: Optional[float] = None
    disp_ci: Optional[float] = None
    disp_cmap: Optional[str] = None
    disp_labels: bool = True


# ---------------------------------------------------------------------------
# Composite map  (single period OR multi-year)
# ---------------------------------------------------------------------------

class CompositeRequest(DisplaySettings):
    var_key: str
    region_name: str
    plot_type: str
    clim_start: int = 1991
    clim_end: int = 2020
    # Single-period mode
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    # Multi-year mode
    multi_years: Optional[List[int]] = None
    multi_months: Optional[List[int]] = None  # 1-12


@app.post("/api/composite")
def composite_map(req: CompositeRequest):
    try:
        region = REGIONS[req.region_name]
        req_d = req.model_dump()

        # ── Multi-year mode ──────────────────────────────────────────
        if req.multi_years and req.multi_months:
            months = sorted(req.multi_months)
            ds = fetcher.fetch_monthly(req.var_key, req.multi_years, months)
            field = fetcher.extract(ds, req.var_key).mean(dim="time")
            wind_uv = _get_wind_uv(ds, req.var_key)
            if "Anomaly" in req.plot_type:
                field = _apply_anomaly(
                    field, req.var_key, months, req.clim_start, req.clim_end, req.plot_type
                )
            yrs_str = ", ".join(str(y) for y in sorted(req.multi_years))
            mon_str = ", ".join(str(m) for m in months)
            title = (
                f"{VARIABLES[req.var_key]['label']}\n"
                f"Years: {yrs_str}  Months: {mon_str}  •  {req.plot_type}"
            )

        # ── Single-period mode ───────────────────────────────────────
        else:
            if not req.start_date or not req.end_date:
                raise ValueError("Provide start_date/end_date or multi_years/multi_months")
            start = date.fromisoformat(req.start_date)
            end = date.fromisoformat(req.end_date)
            months = sorted({d.month for d in pd.date_range(str(start), str(end))})
            days_ago = (date.today() - end).days
            period_days = (end - start).days

            if days_ago < 60 or period_days < 20:
                ds = fetcher.fetch_daily(req.var_key, start, end)
            else:
                year_list = list(range(start.year, end.year + 1))
                ds = fetcher.fetch_monthly(req.var_key, year_list, months)

            field = fetcher.extract(ds, req.var_key).mean(dim="time")
            wind_uv = _get_wind_uv(ds, req.var_key)
            if "Anomaly" in req.plot_type:
                field = _apply_anomaly(
                    field, req.var_key, months, req.clim_start, req.clim_end, req.plot_type
                )
            title = (
                f"{VARIABLES[req.var_key]['label']}\n"
                f"{start} – {end}  •  {req.plot_type}"
            )

        fig = _make_map(field, wind_uv, req_d, region, title)
        return {"image": _fig_b64(fig)}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Analog year finder  (Server-Sent Events)
# ---------------------------------------------------------------------------

class AnalogRequest(DisplaySettings):
    var_key: str
    start_date: str
    end_date: str
    region_name: str
    n_analogs: int = 5
    search_start_year: int = 1950
    clim_start: int = 1991
    clim_end: int = 2020
    use_anomaly: bool = True
    plot_type: str = "Mean"


@app.post("/api/analog")
def find_analogs(req: AnalogRequest):
    def generate():
        def send(obj: dict) -> str:
            return f"data: {json.dumps(obj)}\n\n"

        try:
            start = date.fromisoformat(req.start_date)
            end = date.fromisoformat(req.end_date)
            region = REGIONS[req.region_name]
            months = sorted({d.month for d in pd.date_range(str(start), str(end))})
            req_d = req.model_dump()

            yield send({"status": "Searching for analog years…"})

            finder = AnalogFinder(fetcher)
            analogs = finder.find_analogs(
                var_key=req.var_key,
                target_start=start,
                target_end=end,
                region=region,
                n_analogs=req.n_analogs,
                search_years=(req.search_start_year, date.today().year - 1),
                clim_years=(req.clim_start, req.clim_end),
                use_anomaly=req.use_anomaly,
            )

            analogs_json = [
                {**a, "start_date": str(a["start_date"]), "end_date": str(a["end_date"])}
                for a in analogs
            ]
            yield send({"status": "Plotting maps…", "analogs": analogs_json})

            # Target map
            try:
                target_ds = fetcher.fetch_daily(req.var_key, start, end)
                target_field = fetcher.extract(target_ds, req.var_key).mean(dim="time")
            except Exception:
                year_list = list(range(start.year, end.year + 1))
                target_ds = fetcher.fetch_monthly(req.var_key, year_list, months)
                target_field = fetcher.extract(target_ds, req.var_key).mean(dim="time")

            if "Anomaly" in req.plot_type:
                target_field = _apply_anomaly(
                    target_field, req.var_key, months,
                    req.clim_start, req.clim_end, req.plot_type
                )
            wind_uv = _get_wind_uv(target_ds, req.var_key)
            fig = _make_map(
                target_field, wind_uv, req_d, region,
                f"TARGET  {start} – {end}  •  {VARIABLES[req.var_key]['label']}",
            )
            yield send({"target_image": _fig_b64(fig)})

            # Analog maps
            for analog in analogs:
                yr = analog["year"]
                r = analog["correlation"]
                y_start = analog["start_date"]
                y_end = analog["end_date"]
                year_ds = fetcher.fetch_monthly(req.var_key, [yr], months)
                year_field = fetcher.extract(year_ds, req.var_key).mean(dim="time")
                if "Anomaly" in req.plot_type:
                    year_field = _apply_anomaly(
                        year_field, req.var_key, months,
                        req.clim_start, req.clim_end, req.plot_type
                    )
                wind_uv = _get_wind_uv(year_ds, req.var_key)
                fig = _make_map(
                    year_field, wind_uv, req_d, region,
                    f"ANALOG {yr}  ({y_start} – {y_end})  r = {r:.3f}",
                )
                yield send({"analog_image": {"year": yr, "r": r, "image": _fig_b64(fig)}})

            yield send({"done": True})

        except Exception as exc:
            yield send({"error": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
