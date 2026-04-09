"""
Analog Year Weather Maps  —  Streamlit application
===================================================
ERA5-based composite and analog-year map tool.

Usage
-----
    streamlit run app.py

Requires ~/.cdsapirc with valid Copernicus CDS credentials.
See: https://cds.climate.copernicus.eu/how-to-api
"""

from __future__ import annotations

from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.era5_fetcher import ERA5Fetcher, VARIABLES
from src.plotter import WeatherMapPlotter, REGIONS
from src.analog_finder import AnalogFinder

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Analog Year Weather Maps",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — modern dark-card theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Main area ── */
[data-testid="stAppViewContainer"] > .main { background: #0f172a; }
.block-container { padding-top: 1.5rem !important; }

/* ── Headings ── */
h1 { font-size: 1.9rem !important; font-weight: 800 !important; color: #f8fafc !important; letter-spacing: -0.02em; }
h2, h3 { color: #e2e8f0 !important; }
p, li, label, .stMarkdown { color: #cbd5e1 !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    color: #94a3b8 !important;
    font-weight: 600;
    border-radius: 8px 8px 0 0;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 0.55rem 2.2rem !important;
    transition: opacity 0.15s ease;
}
[data-testid="stButton"] > button:hover { opacity: 0.85 !important; }

/* ── Input widgets ── */
[data-testid="stDateInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
}

/* ── Expanders / cards ── */
[data-testid="stExpander"] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Divider ── */
hr { border-color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🌏 Analog Year Weather Maps")
st.caption(
    "ERA5 reanalysis  •  ~5-day lag via Copernicus CDS  •  "
    "Inspired by [NOAA PSL Composites](https://psl.noaa.gov/cgi-bin/data/composites/printpage.pl)"
)

# ---------------------------------------------------------------------------
# Sidebar — shared settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    var_key = st.selectbox(
        "Variable",
        options=list(VARIABLES.keys()),
        format_func=lambda k: VARIABLES[k]["label"],
        index=0,
    )

    region_name = st.selectbox("Region", options=list(REGIONS.keys()), index=0)
    region = REGIONS[region_name]

    plot_type = st.radio(
        "Map type",
        options=["Mean", "Anomaly (vs 1991–2020)", "Standardized Anomaly"],
    )

    clim_label = st.select_slider(
        "Climatology period",
        options=["1951–1980", "1961–1990", "1971–2000", "1981–2010", "1991–2020"],
        value="1991–2020",
    )
    clim_start, clim_end = int(clim_label[:4]), int(clim_label[5:])

    st.divider()
    st.caption(
        "Data: [ERA5 / Copernicus CDS](https://cds.climate.copernicus.eu/)"
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
plotter = WeatherMapPlotter()
fetcher = ERA5Fetcher()


def _get_wind_uv(ds, time_mean: bool = True):
    """Return (u, v) for ws10 maps, or None for other variables."""
    if var_key != "ws10":
        return None
    try:
        u, v = fetcher.extract_wind_components(ds)
        if time_mean and "time" in u.dims:
            u = u.mean(dim="time")
            v = v.mean(dim="time")
        return u, v
    except Exception:
        return None


def render_map(data, title: str, wind_uv=None) -> None:
    fig = plotter.plot_map(
        data=data,
        var_key=var_key,
        region=region,
        title=title,
        plot_type=plot_type,
        wind_uv=wind_uv,
    )
    img_bytes = plotter.fig_to_bytes(fig, dpi=150)
    st.image(img_bytes, use_container_width=True)


def apply_anomaly(mean_field, months: list[int]):
    """Subtract climatology (and optionally normalise by std)."""
    clim = fetcher.get_climatology(var_key, months, (clim_start, clim_end))
    anom = mean_field - clim
    if "Standardized" in plot_type:
        std = fetcher.get_climatology_std(var_key, months, (clim_start, clim_end))
        anom = anom / std.where(std > 0)
    return anom


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_composite, tab_analog = st.tabs(
    ["📊 Composite Map", "🔍 Analog Year Finder"]
)

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 1 — Composite Map                                      ║
# ╚══════════════════════════════════════════════════════════════╝
with tab_composite:
    st.markdown("### Composite Map")
    st.markdown(
        "Choose a date range. Recent periods (within ~60 days) use ERA5T daily data; "
        "longer periods use monthly means."
    )

    col1, col2 = st.columns(2)
    with col1:
        c_start = st.date_input(
            "Start date",
            value=date.today() - timedelta(days=30),
            max_value=date.today() - timedelta(days=1),
            key="c_start",
        )
    with col2:
        c_end = st.date_input(
            "End date",
            value=date.today() - timedelta(days=1),
            max_value=date.today() - timedelta(days=1),
            key="c_end",
        )

    if c_start >= c_end:
        st.error("Start date must be earlier than end date.")
    else:
        if st.button("Draw Map", type="primary", key="btn_composite"):
            with st.spinner("Fetching ERA5 data…"):
                try:
                    months = sorted(
                        {d.month for d in pd.date_range(str(c_start), str(c_end))}
                    )
                    days_ago = (date.today() - c_end).days
                    period_days = (c_end - c_start).days

                    if days_ago < 60 or period_days < 20:
                        ds = fetcher.fetch_daily(var_key, c_start, c_end)
                        field = fetcher.extract(ds, var_key).mean(dim="time")
                        wind_uv = _get_wind_uv(ds, time_mean=True)
                    else:
                        year_list = list(range(c_start.year, c_end.year + 1))
                        ds = fetcher.fetch_monthly(var_key, year_list, months)
                        field = fetcher.extract(ds, var_key).mean(dim="time")
                        wind_uv = _get_wind_uv(ds, time_mean=True)

                    if "Anomaly" in plot_type:
                        field = apply_anomaly(field, months)

                    title = (
                        f"{VARIABLES[var_key]['label']}\n"
                        f"{c_start} – {c_end}  •  {plot_type}"
                    )
                    render_map(field, title, wind_uv=wind_uv)

                except Exception as exc:
                    st.error(f"Error: {exc}")
                    st.info(
                        "Ensure **~/.cdsapirc** contains valid CDS API credentials. "
                        "See https://cds.climate.copernicus.eu/how-to-api"
                    )

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 2 — Analog Year Finder                                 ║
# ╚══════════════════════════════════════════════════════════════╝
with tab_analog:
    st.markdown("### Analog Year Finder")
    st.markdown(
        "Select a target period. The tool finds historical years whose large-scale "
        "pattern (area-weighted pattern correlation) most closely resembles it."
    )

    col1, col2 = st.columns(2)
    with col1:
        a_start = st.date_input(
            "Target start",
            value=date.today() - timedelta(days=30),
            max_value=date.today() - timedelta(days=1),
            key="a_start",
        )
    with col2:
        a_end = st.date_input(
            "Target end",
            value=date.today() - timedelta(days=1),
            max_value=date.today() - timedelta(days=1),
            key="a_end",
        )

    col3, col4 = st.columns(2)
    with col3:
        n_analogs = st.slider("Number of analogs", min_value=3, max_value=10, value=5)
    with col4:
        use_anomaly = st.checkbox(
            "Compare anomaly fields (recommended)",
            value=True,
            help="Removes the seasonal cycle before comparing patterns.",
        )

    search_start_year = st.number_input(
        "Search from year", min_value=1950, max_value=2020, value=1950, step=1
    )

    if a_start >= a_end:
        st.error("Start date must be earlier than end date.")
    else:
        if st.button("Find Analog Years", type="primary", key="btn_analog"):
            with st.spinner(
                "Downloading ERA5 monthly means for all years — this may take a few minutes on first run…"
            ):
                try:
                    finder = AnalogFinder(fetcher)
                    analogs = finder.find_analogs(
                        var_key=var_key,
                        target_start=a_start,
                        target_end=a_end,
                        region=region,
                        n_analogs=n_analogs,
                        search_years=(search_start_year, date.today().year - 1),
                        clim_years=(clim_start, clim_end),
                        use_anomaly=use_anomaly,
                    )

                    # ── Results table ────────────────────────────────
                    st.success(f"Top {len(analogs)} analog years found!")
                    df = pd.DataFrame(analogs)[
                        ["year", "correlation", "rmse", "start_date", "end_date"]
                    ]
                    df.columns = ["Year", "Correlation (r)", "RMSE", "Period start", "Period end"]
                    st.dataframe(df.set_index("Year"), use_container_width=True)

                    # ── Maps ─────────────────────────────────────────
                    st.markdown("### Maps")
                    months = sorted(
                        {d.month for d in pd.date_range(str(a_start), str(a_end))}
                    )

                    # Target map
                    with st.spinner("Plotting target period…"):
                        try:
                            target_ds = fetcher.fetch_daily(var_key, a_start, a_end)
                            target_field = fetcher.extract(target_ds, var_key).mean(dim="time")
                        except Exception:
                            year_list = list(range(a_start.year, a_end.year + 1))
                            target_ds = fetcher.fetch_monthly(var_key, year_list, months)
                            target_field = fetcher.extract(target_ds, var_key).mean(dim="time")

                        if "Anomaly" in plot_type:
                            target_field = apply_anomaly(target_field, months)

                        wind_uv = _get_wind_uv(target_ds, time_mean=True)
                        st.markdown(f"**Target period: {a_start} – {a_end}**")
                        render_map(
                            target_field,
                            f"TARGET  {a_start} – {a_end}  •  {VARIABLES[var_key]['label']}",
                            wind_uv=wind_uv,
                        )

                    # Analog year maps
                    for analog in analogs:
                        yr = analog["year"]
                        r = analog["correlation"]
                        y_start = analog["start_date"]
                        y_end = analog["end_date"]

                        with st.spinner(f"Plotting analog year {yr}…"):
                            year_ds = fetcher.fetch_monthly(var_key, [yr], months)
                            year_field = fetcher.extract(year_ds, var_key).mean(dim="time")
                            if "Anomaly" in plot_type:
                                year_field = apply_anomaly(year_field, months)

                            wind_uv = _get_wind_uv(year_ds, time_mean=True)
                            st.markdown(f"**Analog: {yr}** &nbsp;(r = {r:.3f})")
                            render_map(
                                year_field,
                                f"ANALOG {yr}  ({y_start} – {y_end})  r = {r:.3f}",
                                wind_uv=wind_uv,
                            )

                except Exception as exc:
                    st.error(f"Error: {exc}")
                    st.info(
                        "Ensure **~/.cdsapirc** contains valid CDS API credentials. "
                        "See https://cds.climate.copernicus.eu/how-to-api"
                    )
