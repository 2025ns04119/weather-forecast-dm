"""
Weather app - current weather, hourly and 7-day forecast for ANY place.

Every number on the Weather screen is produced by this project:
    forecasts (temperature, humidity, wind, cloud, rain chance)  -> ML model weather-7day
    current conditions                                           -> latest validated observation
    condition/icon, feels like, UV index, sunrise/sunset         -> our calculations (common/derived.py)
Open-Meteo is only the raw data source.

Bengaluru (default) uses the stored hourly-pipeline forecast; any searched place is
forecast on demand with the same validation rules, features and model.

Run locally:   streamlit run dashboard/app.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "dashboard")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# Streamlit Cloud: copy secrets into environment variables used by the pipeline code
for _k in ["NEON_DB_URL", "MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]:
    if not os.getenv(_k):
        try:
            if _k in st.secrets:
                os.environ[_k] = str(st.secrets[_k])
        except Exception:
            pass

from config.locations import DEFAULT_LOCATION          # noqa: E402
from ingestion.open_meteo import geocode               # noqa: E402
import weather_view as wv                              # noqa: E402

try:
    from streamlit_searchbox import st_searchbox
except ImportError:
    st_searchbox = None

REPO_URL = "https://github.com/2025ns04119/weather-forecast-dm"
DAGSHUB_URL = "https://dagshub.com/2025ns04119/weather-forecast-dm"
HOME = dict(DEFAULT_LOCATION, admin1="Karnataka")
FONT = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700&display=swap">')

C_FORECAST, C_ACTUAL = "#2a78d6", "#eb6834"
C_BAND, C_GRID, C_TEXT, C_MUTED = "rgba(42,120,214,0.13)", "#e6e5e0", "#0b0b0b", "#6b6a65"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}
PLOT_CFG = {"displayModeBar": False, "responsive": True}

INSIGHTS_CSS = """
<style>
  .block-container {padding-top: 4rem;}
  .pill {display:inline-flex; align-items:center; gap:.45rem; padding:.35rem .8rem; border-radius:999px;
         border:1px solid #e6e5e0; background:#fff; font-size:.85rem; color:#0b0b0b; white-space:nowrap;}
  .dot {width:.55rem; height:.55rem; border-radius:50%; display:inline-block;}
  .kpi {background:#fff; border:1px solid #e6e5e0; border-radius:12px; padding:.9rem 1.05rem; margin-bottom:.75rem;}
  .kpi-label {color:#6b6a65; font-size:.78rem; text-transform:uppercase; letter-spacing:.05em;}
  .kpi-value {font-size:1.6rem; font-weight:650; color:#0b0b0b; line-height:1.25; font-variant-numeric:tabular-nums;}
  .kpi-sub {color:#52514e; font-size:.83rem;}
  .note {color:#6b6a65; font-size:.85rem;}
  .wrap {display:flex; flex-wrap:wrap; gap:.5rem;}
  .flow {display:flex; flex-wrap:wrap; align-items:center; gap:.4rem .35rem;}
  .step {background:#fff; border:1px solid #d6d5cf; border-radius:10px; padding:.55rem .75rem; font-size:.85rem;
         line-height:1.25; color:#0b0b0b;}
  .step small {display:block; color:#6b6a65; font-size:.75rem;}
  .step.hl {border-color:#2a78d6; background:#eaf2fc;}
  .arrow {color:#8a8984; font-size:1.1rem;}
  .chart-title {font-weight:600; font-size:1rem; color:#0b0b0b; margin:.6rem 0 -.2rem;}
</style>"""


# ------------------------------------------------------------------ data access (cached)
def _db_url():
    url = os.getenv("NEON_DB_URL")
    if url and url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


@st.cache_resource
def engine():
    return create_engine(_db_url(), pool_pre_ping=True)


@st.cache_data(ttl=300, show_spinner=False)
def query(sql: str, **params) -> pd.DataFrame:
    with engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


@st.cache_resource(show_spinner="Preparing the forecast model (first time only)…")
def get_model():
    from common.model_loader import load_champion
    return load_champion()


@st.cache_data(ttl=86400, show_spinner=False)
def search_places(q: str) -> list[dict]:
    return geocode(q)


@st.cache_data(ttl=1800, show_spinner=False)
def on_demand(name: str, lat: float, lon: float) -> dict:
    from inference.on_demand import forecast_location
    model, version = get_model()
    out = forecast_location(model, name, lat, lon)
    return {"forecast": out["forecast"], "observed": out["observed"], "made_at": out["made_at"]}


OBS_COLS = ("ts, temperature_2m, relative_humidity_2m, dew_point_2m, wind_speed_10m, "
            "wind_direction_10m, cloud_cover, precipitation")


def stored_forecast(loc: str) -> pd.DataFrame:
    return query("""
        SELECT target_ts, predicted_temp, predicted_humidity AS humidity, predicted_wind AS wind,
               predicted_cloud AS cloud, rain_probability AS rain_prob, model_name, model_version, made_at
        FROM ml.predictions
        WHERE location = :loc AND made_at = (
            SELECT max(made_at) FROM ml.predictions
            WHERE location = :loc AND rain_probability IS NOT NULL)
        ORDER BY target_ts""", loc=loc)


def stored_observed(loc: str, days: int = 3) -> pd.DataFrame:
    return query(f"""SELECT {OBS_COLS} FROM clean.weather_hourly
                     WHERE location = :loc AND ts >= now() - interval '{int(days)} days' ORDER BY ts""", loc=loc)


def is_monitored(p: dict) -> bool:
    return (p["name"] == HOME["name"] and abs(p["latitude"] - HOME["latitude"]) < 0.3
            and abs(p["longitude"] - HOME["longitude"]) < 0.3)


def weather_data(place: dict) -> dict:
    """Stored pipeline forecast for the monitored city (if fresh), else on demand."""
    if is_monitored(place) and _db_url():
        try:
            fc = stored_forecast(place["name"])
            if not fc.empty:
                made = pd.to_datetime(fc["made_at"].iloc[0], utc=True)
                if pd.Timestamp.now(tz="UTC") - made < pd.Timedelta(hours=6):
                    obs = stored_observed(place["name"])
                    if not obs.empty:
                        return {"forecast": fc, "observed": obs, "made_at": made}
        except Exception:
            pass
    return on_demand(place["name"], place["latitude"], place["longitude"])


def place_label(p: dict) -> str:
    out = []
    for x in [p.get("name"), p.get("admin1"), p.get("country")]:
        if x and x not in out:
            out.append(x)
    return ", ".join(out)


def to_place(r: dict) -> dict:
    return {"name": r["name"], "admin1": r.get("admin1"), "country": r.get("country"),
            "latitude": float(r["latitude"]), "longitude": float(r["longitude"]),
            "timezone": r.get("timezone") or "UTC"}


def choose_place(value):
    st.session_state.place = value


def go_home():
    st.session_state.place = dict(HOME)


if "place" not in st.session_state:
    st.session_state.place = dict(HOME)


# ================================================================== PAGE 1: WEATHER
def weather_page():
    st.set_page_config(page_title="Weather", page_icon="🌤️", layout="wide")
    st.markdown(FONT + wv.CSS, unsafe_allow_html=True)

    # ---- header: brand · search · home
    c1, c2, c3 = st.columns([2, 5, 2], vertical_alignment="center")
    c1.markdown(
        '<div class="brand"><svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true">'
        '<rect width="34" height="34" rx="10" fill="#2A78D6"/><circle cx="13" cy="13" r="4.5" fill="none" '
        'stroke="#FFD27A" stroke-width="2"/><path d="M12 25h11a4.2 4.2 0 0 0 .4-8.4 5.8 5.8 0 0 0-11.1 1.6A3.4 3.4 0 0 0 12 25z" '
        'fill="#FFFFFF"/></svg><span><b>Weather</b><small>Now · Hourly · 7 days</small></span></div>',
        unsafe_allow_html=True)
    with c2:
        if st_searchbox is not None:
            def _search(term: str):
                if not term or len(term.strip()) < 2:
                    return []
                try:
                    return [(place_label(r), to_place(r)) for r in search_places(term.strip())]
                except Exception:
                    return []
            sel = st_searchbox(_search, placeholder="Search city or place", key="place_search",
                               clear_on_submit=True, debounce=250)
            if sel and sel != st.session_state.place:
                st.session_state.place = sel
                st.rerun()
        else:
            q = st.text_input("Search", placeholder="Search city or place", label_visibility="collapsed")
            if q and len(q.strip()) >= 2:
                for i, r in enumerate(search_places(q.strip())):
                    st.button(place_label(r), key=f"p{i}", use_container_width=True,
                              on_click=choose_place, args=(to_place(r),))
    with c3:
        st.button(f"⌂  Home · {HOME['name']}", on_click=go_home, use_container_width=True,
                  disabled=is_monitored(st.session_state.place))

    place = st.session_state.place
    with st.spinner(f"Loading weather for {place['name']}…"):
        try:
            data = weather_data(place)
            v = wv.build_view(place, data["forecast"], data["observed"], data["made_at"])
        except Exception as e:
            st.error(f"Couldn't load the weather for {place_label(place)} right now - please try again "
                     f"in a moment. ({type(e).__name__}: {e})")
            return

    left, right = st.columns([5, 8], gap="medium")
    left.markdown(wv.now_card(v), unsafe_allow_html=True)
    with right:
        with st.container(key="fc_card"):
            h1, h2 = st.columns([3, 2], vertical_alignment="center")
            h1.markdown('<div class="fc-title">Forecast</div>', unsafe_allow_html=True)
            view = h2.segmented_control("Forecast view", ["Hourly", "7 days"], default="Hourly",
                                        label_visibility="collapsed", key="fc_view") or "Hourly"
            st.markdown(wv.hourly_html(v) if view == "Hourly" else wv.week_html(v), unsafe_allow_html=True)
            st.markdown(f'<div class="src">Weather data: <a href="https://open-meteo.com" target="_blank">'
                        f'Open-Meteo.com</a> (CC BY 4.0) · Updated {wv.clock(v["made_at"])}</div>',
                        unsafe_allow_html=True)


# ================================================================== PAGE 2: PROJECT INSIGHTS
def kpi(label, value, sub=""):
    return (f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>')


def pill(status, label):
    icon = {"good": "✓", "warning": "!", "critical": "✕"}[status]
    return (f'<span class="pill"><span class="dot" style="background:{STATUS[status]}"></span>'
            f'<b>{icon}</b> {label}</span>')


def fmt_age(ts):
    mins = int((pd.Timestamp.now(tz="UTC") - ts).total_seconds() // 60)
    return f"{mins} min ago" if mins < 60 else (f"{mins // 60} h {mins % 60} min ago" if mins < 2880
                                                else f"{mins // 1440} days ago")


def to_local(s, tz):
    return pd.to_datetime(s, utc=True).dt.tz_convert(tz).dt.tz_localize(None)


def base_layout(fig, height=420, y_title="Temperature (°C)"):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#ffffff",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color=C_TEXT, size=13), hovermode="x unified",
                      hoverlabel=dict(bgcolor="#ffffff", bordercolor=C_GRID, font_color=C_TEXT),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                                  font=dict(color=C_MUTED)))
    fig.update_xaxes(showgrid=False, linecolor=C_GRID, tickfont=dict(color=C_MUTED), showspikes=True,
                     spikecolor=C_MUTED, spikethickness=1, spikedash="dot", spikemode="across")
    fig.update_yaxes(title=y_title, gridcolor=C_GRID, zeroline=False, tickfont=dict(color=C_MUTED),
                     title_font=dict(color=C_MUTED))
    return fig


def insights_page():
    st.set_page_config(page_title="Project insights", page_icon="📊", layout="wide")
    st.markdown(INSIGHTS_CSS, unsafe_allow_html=True)
    tz, loc = HOME["timezone"], HOME["name"]
    st.title("📊 Project insights")
    st.markdown(f'<div class="note">Data management for ML · monitored city: <b>{loc}</b> '
                f'(forecast every hour in the cloud, stored and verified)</div>', unsafe_allow_html=True)
    if not _db_url():
        st.info("Connect the database (NEON_DB_URL) to see project insights.")
        return

    fc = stored_forecast(loc)
    acc = query("""
        SELECT day_ahead, AVG(abs_err) AS mae, COUNT(*) AS n FROM (
            SELECT CEIL(EXTRACT(EPOCH FROM (p.target_ts - p.made_at)) / 86400.0)::int AS day_ahead,
                   ABS(p.predicted_temp - c.temperature_2m) AS abs_err
            FROM ml.predictions p
            JOIN clean.weather_hourly c ON c.location = p.location AND c.ts = p.target_ts
            WHERE p.location = :loc AND p.made_at > now() - interval '30 days') e
        WHERE day_ahead BETWEEN 1 AND 7 GROUP BY day_ahead ORDER BY day_ahead""", loc=loc)
    mae_by_day = dict(zip(acc["day_ahead"], acc["mae"])) if not acc.empty else {}

    k1, k2, k3 = st.columns(3)
    if not fc.empty:
        made = pd.to_datetime(fc["made_at"].iloc[0], utc=True)
        age_h = (pd.Timestamp.now(tz="UTC") - made).total_seconds() / 3600
        k1.markdown(kpi("Model", f"{fc['model_name'].iloc[0]} v{fc['model_version'].iloc[0]}",
                        "Global · 5 targets × 7 forecast days"), unsafe_allow_html=True)
        k2.markdown(kpi("Live next-day temperature error", f"±{mae_by_day[1]:.1f}°C" if 1 in mae_by_day else "—",
                        "Mean absolute error vs observed"), unsafe_allow_html=True)
        s = "good" if age_h <= 2 else "warning" if age_h <= 6 else "critical"
        k3.markdown(kpi("Latest forecast", fmt_age(made),
                        pill(s, "pipeline " + {"good": "healthy", "warning": "delayed", "critical": "stale"}[s])),
                    unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["🎯  Forecast & accuracy", "🗄️  Data pipeline", "ℹ️  About"])

    with t1:
        if not fc.empty:
            fc["time"] = to_local(fc["target_ts"], tz)
            made = pd.to_datetime(fc["made_at"].iloc[0], utc=True)
            fc["day_ahead"] = ((pd.to_datetime(fc["target_ts"], utc=True) - made).dt.total_seconds()
                               / 86400).apply(lambda x: max(1, int(-(-x // 1))))
            obs = stored_observed(loc, 3)
            st.markdown('<div class="chart-title">Hourly temperature · last 48 h observed and next 7 days forecast</div>',
                        unsafe_allow_html=True)
            fig = go.Figure()
            band = fc.assign(err=fc["day_ahead"].map(mae_by_day)).dropna(subset=["err"])
            if not band.empty:
                fig.add_trace(go.Scatter(x=band["time"], y=band["predicted_temp"] + band["err"], mode="lines",
                                         line=dict(width=0), hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=band["time"], y=band["predicted_temp"] - band["err"], mode="lines",
                                         line=dict(width=0), fill="tonexty", fillcolor=C_BAND, hoverinfo="skip",
                                         name="Typical error range"))
            if not obs.empty:
                obs["time"] = to_local(obs["ts"], tz)
                fig.add_trace(go.Scatter(x=obs["time"], y=obs["temperature_2m"], mode="lines", name="Observed",
                                         line=dict(color=C_ACTUAL, width=2),
                                         hovertemplate="Observed %{y:.1f}°C<extra></extra>"))
            fig.add_trace(go.Scatter(x=fc["time"], y=fc["predicted_temp"], mode="lines", name="Forecast",
                                     line=dict(color=C_FORECAST, width=2), customdata=fc["day_ahead"],
                                     hovertemplate="Forecast %{y:.1f}°C (day %{customdata})<extra></extra>"))
            base_layout(fig, 400)
            now = pd.Timestamp.now(tz=tz).tz_localize(None)
            fig.add_shape(type="line", x0=now, x1=now, y0=0, y1=1, yref="paper",
                          line=dict(color=C_MUTED, width=1, dash="dot"))
            fig.add_annotation(x=now, y=0.98, yref="paper", text="Now", showarrow=False, xanchor="left",
                               xshift=4, font=dict(color=C_MUTED, size=12))
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)

        c_left, c_right = st.columns([3, 2])
        ver = query("""
            SELECT DISTINCT ON (p.target_ts) p.target_ts, p.predicted_temp, c.temperature_2m
            FROM ml.predictions p JOIN clean.weather_hourly c ON c.location = p.location AND c.ts = p.target_ts
            WHERE p.location = :loc AND p.target_ts >= now() - interval '72 hours'
              AND p.made_at <= p.target_ts - interval '24 hours'
            ORDER BY p.target_ts, p.made_at DESC""", loc=loc)
        with c_left:
            if ver.empty:
                st.info("Forecast-vs-observed appears once forecast hours have passed (about 24 h of hourly runs).")
            else:
                ver["time"] = to_local(ver["target_ts"], tz)
                mae = (ver["predicted_temp"] - ver["temperature_2m"]).abs().mean()
                st.markdown(f'<div class="chart-title">Forecast vs observed · last 72 h · MAE {mae:.2f}°C</div>',
                            unsafe_allow_html=True)
                f2 = go.Figure()
                f2.add_trace(go.Scatter(x=ver["time"], y=ver["temperature_2m"], mode="lines", name="Observed",
                                        line=dict(color=C_ACTUAL, width=2),
                                        hovertemplate="Observed %{y:.1f}°C<extra></extra>"))
                f2.add_trace(go.Scatter(x=ver["time"], y=ver["predicted_temp"], mode="lines",
                                        name="Forecast issued ≥ 24 h earlier", line=dict(color=C_FORECAST, width=2),
                                        hovertemplate="Forecast %{y:.1f}°C<extra></extra>"))
                base_layout(f2, 360)
                st.plotly_chart(f2, use_container_width=True, config=PLOT_CFG)
        with c_right:
            if acc.empty:
                st.info("Error by forecast day appears after the first forecasts have been verified.")
            else:
                st.markdown('<div class="chart-title">Average temperature error by forecast day</div>',
                            unsafe_allow_html=True)
                f3 = go.Figure(go.Bar(x=[f"Day {d}" for d in acc["day_ahead"]], y=acc["mae"],
                                      marker=dict(color=C_FORECAST, cornerradius=4),
                                      text=[f"{x:.2f}°" for x in acc["mae"]], textposition="outside",
                                      textfont=dict(color=C_TEXT), customdata=acc["n"],
                                      hovertemplate="%{x}: MAE %{y:.2f}°C<br>%{customdata} verified hours<extra></extra>"))
                base_layout(f3, 360, "Mean absolute error (°C)")
                f3.update_layout(hovermode="closest", bargap=0.35, showlegend=False)
                f3.update_xaxes(showspikes=False)
                f3.update_yaxes(rangemode="tozero")
                st.plotly_chart(f3, use_container_width=True, config=PLOT_CFG)

    with t2:
        c = query("""
            SELECT (SELECT count(*) FROM clean.locations) AS locations,
                   (SELECT count(*) FROM raw.weather_api) AS raw_payloads,
                   (SELECT count(*) FROM clean.weather_hourly) AS clean_hours,
                   (SELECT count(*) FROM features.weather_features) AS feature_rows,
                   (SELECT count(*) FROM quarantine.weather_rejects) AS quarantined,
                   (SELECT count(*) FROM ml.predictions) AS forecast_rows""").iloc[0]
        cards = [("Locations", c.locations, "Location dimension table"),
                 ("Bronze · raw payloads", c.raw_payloads, "API responses stored as-is"),
                 ("Silver · clean hours", c.clean_hours, "Validated hourly rows"),
                 ("Gold · feature rows", c.feature_rows, "Feature versions v1 → v3"),
                 ("Quarantined", c.quarantined, "Failed quality rules"),
                 ("Forecast rows", c.forecast_rows, "Stored predictions")]
        for col, (lab, val, sub) in zip(st.columns(6), cards):
            col.markdown(kpi(lab, f"{int(val):,}", sub), unsafe_allow_html=True)

        st.markdown("#### Data flow")
        steps = [("Open-Meteo API", "raw observations", ""), ("Bronze", "raw.weather_api", ""),
                 ("Validation", "rules + Great Expectations", ""), ("Silver", "clean.weather_hourly", ""),
                 ("Gold", "features v3", ""), ("Models", "5 targets · MLflow · DVC", ""),
                 ("Calculations", "sun, UV, feels like, condition", ""), ("Weather app", "this dashboard", "hl")]
        st.markdown('<div class="flow">' + '<span class="arrow">→</span>'.join(
            f'<div class="step {k}">{t}<small>{s}</small></div>' for t, s, k in steps) + '</div>',
            unsafe_allow_html=True)

        st.markdown("#### Pipeline health")
        runs = query("""SELECT run_id, pipeline, started_at, finished_at, status, rows_in, rows_out, notes
                        FROM ml.pipeline_runs ORDER BY run_id DESC LIMIT 200""")
        if runs.empty:
            st.info("No pipeline runs logged yet.")
        else:
            runs["started"] = to_local(runs["started_at"], tz)
            runs["duration_s"] = (pd.to_datetime(runs["finished_at"], utc=True)
                                  - pd.to_datetime(runs["started_at"], utc=True)).dt.total_seconds().round(1)
            last = runs.sort_values("run_id").groupby("pipeline").tail(1).sort_values("pipeline")
            st.markdown('<div class="wrap">' + "".join(
                pill("good" if r["status"] == "success" else "critical",
                     f"{r['pipeline']} · {r['status']} · {r['started']:%d %b %H:%M}") for _, r in last.iterrows())
                + '</div>', unsafe_allow_html=True)
            st.write("")
            st.dataframe(runs[["started", "pipeline", "status", "rows_in", "rows_out", "duration_s", "notes"]]
                         .rename(columns={"started": "Started (IST)", "pipeline": "Pipeline", "status": "Status",
                                          "rows_in": "Rows in", "rows_out": "Rows out",
                                          "duration_s": "Duration (s)", "notes": "Notes / lineage"}).head(60),
                         use_container_width=True, hide_index=True, height=320)

        st.markdown("#### Data drift")
        try:
            drift = query("SELECT run_at, drift_detected, drift_share FROM ml.drift_reports ORDER BY run_at DESC LIMIT 30")
        except Exception:
            drift = pd.DataFrame()
        if drift.empty:
            st.info("Drift monitoring (Evidently) results will appear here once it is enabled.")
        else:
            d = drift.iloc[0]
            st.markdown(pill("critical" if d["drift_detected"] else "good",
                             f"{'Drift detected' if d['drift_detected'] else 'No drift'} · "
                             f"{(d['drift_share'] or 0):.0%} of features drifted"), unsafe_allow_html=True)

    with t3:
        a1, a2 = st.columns(2)
        a1.markdown(f"""
#### What the app shows, and how it is produced
| On screen | Produced by |
|---|---|
| Hourly & 7-day temperature | ML regression (day 1 … 7 models) |
| Humidity, wind, cloud cover forecasts | ML regression |
| Rain chance % | ML classification (probability of ≥ 0.1 mm/h) |
| Current conditions | latest validated observation (silver layer) |
| Condition text & icon | rules on predicted cloud cover & rain chance |
| Feels like | Steadman apparent-temperature formula |
| Sunrise / sunset | NOAA solar equations |
| UV index | clear-sky UV from sun angle × cloud factor |

**{loc}** is forecast every hour in the cloud and verified; any searched place is forecast on
demand with the same validation rules, features and model.
""")
        a2.markdown(f"""
#### Data management concepts
- **Medallion architecture:** bronze → silver → gold
- **Data modelling:** location dimension table
- **Data quality:** rule checks + Great Expectations, quarantine for bad rows
- **Lineage & metadata:** every run logged; each row traces to its raw payload
- **Versioning:** features (v1 → v3), DVC training snapshots, MLflow model registry
- **Automation:** GitHub Actions runs the pipeline hourly

#### Tools (all free & open source)
Python · PostgreSQL (Neon) · Great Expectations · DVC · MLflow (DagsHub) · scikit-learn · XGBoost ·
GitHub Actions · Streamlit · Plotly

[Source code on GitHub]({REPO_URL}) · [Experiments & models on DagsHub]({DAGSHUB_URL})

Raw weather data & geocoding: [Open-Meteo.com](https://open-meteo.com) (CC BY 4.0).
""")


# ================================================================== navigation
nav = st.navigation([st.Page(weather_page, title="Weather", icon="🌤️", default=True),
                     st.Page(insights_page, title="Project insights", icon="📊", url_path="insights")],
                    position="top")
nav.run()
