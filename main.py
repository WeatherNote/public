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

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.era5_fetcher import ERA5Fetcher, VARIABLES
from src.plotter import WeatherMapPlotter, REGIONS
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


# ---------------------------------------------------------------------------
# Composite map
# ---------------------------------------------------------------------------

class CompositeRequest(BaseModel):
    var_key: str
    start_date: str
    end_date: str
    region_name: str
    plot_type: str
    clim_start: int = 1991
    clim_end: int = 2020


@app.post("/api/composite")
def composite_map(req: CompositeRequest):
    try:
        start = date.fromisoformat(req.start_date)
        end = date.fromisoformat(req.end_date)
        region = REGIONS[req.region_name]
        months = sorted({d.month for d in pd.date_range(str(start), str(end))})
        days_ago = (date.today() - end).days
        period_days = (end - start).days

        if days_ago < 60 or period_days < 20:
            ds = fetcher.fetch_daily(req.var_key, start, end)
        else:
            year_list = list(range(start.year, end.year + 1))
            ds = fetcher.fetch_monthly(req.var_key, year_list, months)

        field = fetcher.extract(ds, req.var_key).mean(dim="time")

        if "Anomaly" in req.plot_type:
            field = _apply_anomaly(
                field, req.var_key, months, req.clim_start, req.clim_end, req.plot_type
            )

        wind_uv = _get_wind_uv(ds, req.var_key)
        title = (
            f"{VARIABLES[req.var_key]['label']}\n"
            f"{start} – {end}  •  {req.plot_type}"
        )
        fig = plotter.plot_map(
            data=field,
            var_key=req.var_key,
            region=region,
            title=title,
            plot_type=req.plot_type,
            wind_uv=wind_uv,
        )
        return {"image": _fig_b64(fig)}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Analog year finder  (SSE — streams progress to client)
# ---------------------------------------------------------------------------

class AnalogRequest(BaseModel):
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
    """Server-Sent Events stream — yields JSON lines prefixed with 'data: '."""

    def generate():
        def send(obj: dict) -> str:
            return f"data: {json.dumps(obj)}\n\n"

        try:
            start = date.fromisoformat(req.start_date)
            end = date.fromisoformat(req.end_date)
            region = REGIONS[req.region_name]
            months = sorted({d.month for d in pd.date_range(str(start), str(end))})

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

            # Serialise analogs (dates → strings)
            analogs_json = [
                {
                    **a,
                    "start_date": str(a["start_date"]),
                    "end_date": str(a["end_date"]),
                }
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
            fig = plotter.plot_map(
                data=target_field,
                var_key=req.var_key,
                region=region,
                title=f"TARGET  {start} – {end}  •  {VARIABLES[req.var_key]['label']}",
                plot_type=req.plot_type,
                wind_uv=wind_uv,
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
                fig = plotter.plot_map(
                    data=year_field,
                    var_key=req.var_key,
                    region=region,
                    title=f"ANALOG {yr}  ({y_start} – {y_end})  r = {r:.3f}",
                    plot_type=req.plot_type,
                    wind_uv=wind_uv,
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
