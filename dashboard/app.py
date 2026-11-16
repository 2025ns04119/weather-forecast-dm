"""
Weather app - current weather, hourly and 7-day forecast for ANY place.

Every number on the Weather screen is produced by this project:
    forecasts (temperature, humidity, wind, cloud, rain chance)  -> ML model weather-7day
    current conditions                                           -> latest validated observation
    condition/icon, feels like, UV index, sunrise/sunset         -> our calculations (pipeline/common/derived.py)
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

from pipeline.data_collection.locations import DEFAULT_LOCATION          # noqa: E402
from pipeline.data_collection.open_meteo import geocode               # noqa: E402
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
    from pipeline.common.model_loader import load_champion
    return load_champion()


@st.cache_data(ttl=86400, show_spinner=False)
def search_places(q: str) -> list[dict]:
    return geocode(q)


@st.cache_data(ttl=1800, show_spinner=False)
def on_demand(name: str, lat: float, lon: float) -> dict:
    from pipeline.model_deployment.on_demand import forecast_location
    model, version = get_model()
    out = forecast_location(model, name, lat, lon)
    return {"forecast": out["forecast"], "observed": out["observed"], "made_at": out["made_at"]}


OBS_COLS = ("ts, temperature_2m, relative_humidity_2m, dew_point_2m, wind_speed_10m, "
            "wind_direction_10m, cloud_cover, precipitation")


@st.cache_data(ttl=1800, show_spinner="Profiling data quality…")
def quality_profile() -> dict:
    """Data quality metrics from the database (same checks as reports/data_quality_report.pdf)."""
    from pipeline.profiling_validation.quality_metrics import latest_gx_result, profile
    info = profile(engine())
    info["gx"] = latest_gx_result(engine())
    return info


@st.cache_data(ttl=3600, show_spinner="Running exploratory data analysis…")
def eda_results() -> dict:
    """EDA computed with SQL aggregates (same analyses as reports/eda_summary.md)."""
    from pipeline.data_preparation.eda_metrics import collect
    return collect(engine())


@st.cache_data(ttl=3600, show_spinner="Loading evaluation results from MLflow…")
def evaluation_results():
    """Test-set metrics of the latest training, from MLflow (same as reports/model_performance.pdf)."""
    from pipeline.model_evaluation.results import collect
    return collect()


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

    t1, t2, tq, tx, te, t3 = st.tabs(["🎯  Forecast & accuracy", "🗄️  Data pipeline", "✅  Data quality",
                                      "🔍  Exploratory analysis", "📈  Model evaluation", "ℹ️  About"])

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

    with tq:
        try:
            dq = quality_profile()
        except Exception as e:
            dq = None
            st.info(f"Data quality metrics are not available right now ({type(e).__name__}).")
        if dq:
            comp = pd.DataFrame(dq["completeness"])
            overall = 100 * comp["hours"].sum() / max(comp["expected"].sum(), 1)
            missing = sum(v["count"] for v in dq["missing_values"].values())
            ranges = sum(dq["range_violations"].values())
            quar_w = sum(r["n"] for r in dq["quarantine_weather"])
            quar_l = sum(r["n"] for r in dq["quarantine_locations"])
            cards = [("Great Expectations", dq["gx"] or "—", "Latest validation run"),
                     ("Completeness", f"{overall:.2f}%", f"{int(comp['missing_hours'].sum()):,} missing hours"),
                     ("Missing values", f"{missing:,}", "Silver layer, 8 variables"),
                     ("Duplicates", f"{dq['duplicates']:,}", "Same location + hour"),
                     ("Range violations", f"{ranges:,}", "Values outside physical limits"),
                     ("Quarantined rows", f"{quar_w + quar_l:,}", f"weather {quar_w:,} · city CSV {quar_l:,}")]
            for col, (lab, val, sub) in zip(st.columns(6), cards):
                col.markdown(kpi(lab, val, sub), unsafe_allow_html=True)
            sch = dq["schema"]
            ok = not sch["missing_columns"] and not sch["unexpected_columns"]
            st.markdown(pill("good" if ok else "critical", "schema matches the expected columns" if ok else
                             f"schema mismatch: missing {sch['missing_columns']}, unexpected {sch['unexpected_columns']}"),
                        unsafe_allow_html=True)
            st.write("")
            q_left, q_right = st.columns([3, 2])
            with q_left:
                st.markdown('<div class="chart-title">Completeness per location (% of expected hourly rows)</div>',
                            unsafe_allow_html=True)
                c = comp.sort_values("completeness_pct")
                fq = go.Figure(go.Bar(x=c["completeness_pct"], y=c["location"], orientation="h",
                                      marker=dict(color=C_FORECAST, cornerradius=3),
                                      customdata=c[["hours", "missing_hours"]],
                                      hovertemplate="%{y}: %{x:.2f}%<br>%{customdata[0]:,} hours, "
                                                    "%{customdata[1]:,} missing<extra></extra>"))
                base_layout(fq, max(420, 22 * len(c)), "")
                fq.update_layout(hovermode="closest", showlegend=False, bargap=0.25)
                fq.update_xaxes(range=[max(0, float(c["completeness_pct"].min()) - 1), 100.2], showspikes=False,
                                title="Completeness (%)")
                st.plotly_chart(fq, use_container_width=True, config=PLOT_CFG)
            with q_right:
                st.markdown("**Missing values per column (silver)**")
                st.dataframe(pd.DataFrame([{"Column": k, "Missing": v["count"], "%": v["pct"]}
                                           for k, v in dq["missing_values"].items()]),
                             use_container_width=True, hide_index=True)
                st.markdown("**Rows per layer**")
                st.dataframe(pd.DataFrame([{"Table": k, "Rows": v} for k, v in dq["row_counts"].items()]),
                             use_container_width=True, hide_index=True)
                st.markdown("**Quarantine reasons**")
                qr = pd.DataFrame(dq["quarantine_weather"] + dq["quarantine_locations"])
                if qr.empty:
                    st.markdown(pill("good", "no rows quarantined"), unsafe_allow_html=True)
                else:
                    st.dataframe(qr.rename(columns={"reason": "Reason", "n": "Rows"}),
                                 use_container_width=True, hide_index=True)
                stale = comp.sort_values("age_hours", ascending=False).head(5)
                st.markdown("**Freshness – oldest data first** (only the monitored city is refreshed hourly)")
                st.dataframe(stale[["location", "age_hours"]].rename(columns={"location": "Location",
                                                                              "age_hours": "Hours since last row"}),
                             use_container_width=True, hide_index=True)
            st.caption(f"Profiled {dq['generated_at_utc']} UTC · same checks as reports/data_quality_report.pdf "
                       "(pipeline/profiling_validation/quality_metrics.py).")

    with tx:
        try:
            ed = eda_results()
        except Exception as e:
            ed = None
            st.info(f"EDA results are not available right now ({type(e).__name__}).")
        if ed:
            ov = ed["overview"]
            rain, daily, ft = ed["rain"], ed["daily"], ed["feature_target"]
            peak = int(daily.loc[daily["temperature"].idxmax(), "local_hour"]) if not daily.empty else None
            cards = [("Hourly records", f"{ov['rows']:,}", "Silver layer"),
                     ("Locations", f"{ov['locations']}", "32-city catalogue + searched"),
                     ("Period", ov["first_ts"], f"to {ov['last_ts']}"),
                     ("Rainy hours", f"{rain['rainy_pct'].mean():.1f}%", "average · ≥ 0.1 mm"),
                     ("Bengaluru warmest", f"{peak}:00" if peak is not None else "—", "local time (IST)"),
                     ("Top feature", ft.abs().idxmax() if not ft.empty else "—",
                      f"corr {ft[ft.abs().idxmax()]:.2f} with temp 24 h later" if not ft.empty else "")]
            for col, (lab, val, sub) in zip(st.columns(6), cards):
                col.markdown(kpi(lab, val, sub), unsafe_allow_html=True)

            # 1. distributions
            st.markdown("#### 1 · Distributions")
            var = st.segmented_control("Variable", list(ed["histograms"]), default="temperature_2m", key="eda_var")
            var = var or "temperature_2m"
            h = ed["histograms"][var]
            d_left, d_right = st.columns([3, 2])
            with d_left:
                title = f"{var} – hourly values" + (" (> 0 mm only, log scale)" if var == "precipitation" else "")
                st.markdown(f'<div class="chart-title">{title}</div>', unsafe_allow_html=True)
                fh = go.Figure(go.Bar(x=h["mid"], y=h["n"], marker=dict(color=C_FORECAST),
                                      hovertemplate="≈ %{x:.1f}: %{y:,} hours<extra></extra>"))
                base_layout(fh, 330, "Hours")
                fh.update_layout(bargap=0.05, hovermode="closest", showlegend=False)
                fh.update_xaxes(showspikes=False)
                if var == "precipitation":
                    fh.update_yaxes(type="log")
                st.plotly_chart(fh, use_container_width=True, config=PLOT_CFG)
            with d_right:
                st.markdown("**Summary statistics (silver layer)**")
                st.dataframe(ov["stats"].rename(columns={"variable": "Variable", "mean": "Mean", "std": "Std",
                                                         "min": "Min", "max": "Max", "missing": "Missing"}),
                             use_container_width=True, hide_index=True)

            # 2. correlation + feature vs target
            st.markdown("#### 2 · Relationships")
            r_left, r_right = st.columns(2)
            with r_left:
                corr = ed["correlation"]
                st.markdown('<div class="chart-title">Correlation between weather variables</div>',
                            unsafe_allow_html=True)
                fc_ = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                                           colorscale="RdBu", reversescale=True, text=corr.values,
                                           texttemplate="%{text:.2f}", textfont=dict(size=10),
                                           hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
                base_layout(fc_, 430, "")
                fc_.update_layout(hovermode="closest")
                fc_.update_xaxes(showspikes=False, showgrid=False, tickangle=-40)
                fc_.update_yaxes(autorange="reversed", showgrid=False)
                st.plotly_chart(fc_, use_container_width=True, config=PLOT_CFG)
            with r_right:
                st.markdown(f'<div class="chart-title">Gold features vs temperature 24 h later '
                            f'(Bengaluru, {ed["feature_version"]})</div>', unsafe_allow_html=True)
                ff = go.Figure(go.Bar(x=ft.values, y=ft.index, orientation="h",
                                      marker=dict(color=[C_ACTUAL if v < 0 else C_FORECAST for v in ft.values]),
                                      hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
                base_layout(ff, 430, "")
                ff.update_layout(hovermode="closest", showlegend=False)
                ff.update_xaxes(range=[-1, 1], showspikes=False, title="Correlation")
                st.plotly_chart(ff, use_container_width=True, config=PLOT_CFG)

            # 3. seasonal and daily patterns
            st.markdown("#### 3 · Seasonal and daily patterns")
            p_left, p_right = st.columns(2)
            with p_left:
                st.markdown('<div class="chart-title">Mean temperature by month and climate zone</div>',
                            unsafe_allow_html=True)
                fm = go.Figure()
                for zone, g in ed["monthly"].groupby("climate_zone"):
                    fm.add_trace(go.Scatter(x=g["month"], y=g["temperature"], mode="lines+markers", name=zone,
                                            hovertemplate=zone + " · month %{x}: %{y:.1f}°C<extra></extra>"))
                base_layout(fm, 380)
                fm.update_xaxes(tickmode="array", tickvals=list(range(1, 13)),
                                ticktext=["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
                st.plotly_chart(fm, use_container_width=True, config=PLOT_CFG)
            with p_right:
                st.markdown('<div class="chart-title">Bengaluru – average daily cycle (IST)</div>',
                            unsafe_allow_html=True)
                fd = go.Figure()
                fd.add_trace(go.Scatter(x=daily["local_hour"], y=daily["temperature"], mode="lines+markers",
                                        name="Temperature (°C)", line=dict(color=C_FORECAST, width=2),
                                        hovertemplate="%{x}:00 · %{y:.1f}°C<extra></extra>"))
                fd.add_trace(go.Scatter(x=daily["local_hour"], y=daily["humidity"], mode="lines+markers",
                                        name="Humidity (%)", yaxis="y2", line=dict(color=C_ACTUAL, width=2),
                                        hovertemplate="%{x}:00 · %{y:.0f}%<extra></extra>"))
                base_layout(fd, 380)
                fd.update_layout(yaxis2=dict(overlaying="y", side="right", title="Humidity (%)", showgrid=False))
                fd.update_xaxes(title="Local hour", dtick=3)
                st.plotly_chart(fd, use_container_width=True, config=PLOT_CFG)

            # 4. rain sparsity and completeness
            st.markdown("#### 4 · Rain sparsity and data completeness")
            c_left, c_right = st.columns([2, 3])
            with c_left:
                st.markdown('<div class="chart-title">% of hours with rain (≥ 0.1 mm)</div>',
                            unsafe_allow_html=True)
                fr = go.Figure(go.Bar(x=rain["rainy_pct"], y=rain["location"], orientation="h",
                                      marker=dict(color=C_FORECAST, cornerradius=3),
                                      hovertemplate="%{y}: %{x:.1f}% of hours<extra></extra>"))
                base_layout(fr, max(420, 20 * len(rain)), "")
                fr.update_layout(hovermode="closest", showlegend=False, bargap=0.25)
                fr.update_xaxes(showspikes=False, title="% of hours")
                st.plotly_chart(fr, use_container_width=True, config=PLOT_CFG)
            with c_right:
                cm = ed["completeness"]
                st.markdown('<div class="chart-title">Hourly data completeness by city and month (%)</div>',
                            unsafe_allow_html=True)
                fk = go.Figure(go.Heatmap(z=cm.values, x=cm.columns, y=cm.index, zmin=0, zmax=100,
                                          colorscale="Blues", colorbar=dict(title="%"),
                                          hovertemplate="%{y} · %{x}: %{z:.1f}%<extra></extra>"))
                base_layout(fk, max(420, 20 * len(cm)), "")
                fk.update_layout(hovermode="closest")
                fk.update_xaxes(showspikes=False, showgrid=False, tickangle=-40)
                fk.update_yaxes(showgrid=False)
                st.plotly_chart(fk, use_container_width=True, config=PLOT_CFG)

            with st.expander("Key findings and how they shaped the features"):
                st.markdown(f"""
- **Rain is sparse** – only {rain['rainy_pct'].mean():.1f}% of hours are rainy on average, so rain is predicted as a
  *probability* and judged with Brier score, AUC, precision/recall – not accuracy.
- **Seasons differ by climate zone and flip in the southern hemisphere** → *month* and *latitude* features.
- **Strong daily cycle** – Bengaluru peaks around {peak}:00 IST while humidity is lowest → *solar hour* feature.
- **Temperature is persistent** – recent temperatures (lags, rolling means) correlate most with the value 24 h later
  → lag and rolling-mean features; the persistence baseline is the benchmark the model must beat.
- **Humidity, dew point and temperature are correlated** – tree models handle this; linear models use scaling.
- **Completeness** – gaps (light cells) are filled only up to 3 h; longer gaps are dropped, not invented.
""")
            st.caption(f"Computed {ed['generated_at_utc']} UTC with SQL aggregates on the silver and gold layers · "
                       "same analyses as reports/eda_summary.md (pipeline/data_preparation/eda_metrics.py).")

    with te:
        try:
            ev, meta = evaluation_results()
        except Exception as e:
            ev, meta = pd.DataFrame(), {}
            st.info(f"Evaluation results are not available right now ({type(e).__name__}).")
        if not ev.empty:
            cards = [("Champion model", "weather-7day", f"run {str(meta.get('run_id', ''))[:8]}"),
                     ("Test period from", str(meta.get("test_from", "—"))[:10], "time-based split + embargo"),
                     ("Training rows", f"{int(float(meta.get('rows_total') or 0)):,}",
                      f"{meta.get('locations', '—')} locations · features {meta.get('feature_version', '—')}"),
                     ("Model size", f"{meta.get('model_size_mb', '—')} MB", "35 models (5 targets × 7 days)")]
            for col, (lab, val, sub) in zip(st.columns(4), cards):
                col.markdown(kpi(lab, val, sub), unsafe_allow_html=True)
            names = {"Temperature": "temp", "Humidity": "humidity", "Wind": "wind", "Cloud cover": "cloud"}
            units = {"temp": "°C", "humidity": "%", "wind": "km/h", "cloud": "%"}
            pick = st.segmented_control("Target", list(names), default="Temperature", key="eval_target")
            t = names.get(pick or "Temperature", "temp")
            sub = ev[ev["target"] == t].sort_values("day")
            e_left, e_right = st.columns([3, 2])
            with e_left:
                st.markdown(f'<div class="chart-title">{pick or "Temperature"} – MAE on the test period: model vs '
                            f'persistence baseline (lower is better)</div>', unsafe_allow_html=True)
                fe = go.Figure()
                fe.add_trace(go.Scatter(x=[f"Day {d}" for d in sub["day"]], y=sub.get("model_mae"),
                                        mode="lines+markers", name="Model", line=dict(color=C_FORECAST, width=3)))
                fe.add_trace(go.Scatter(x=[f"Day {d}" for d in sub["day"]], y=sub.get("baseline_mae"),
                                        mode="lines+markers", name="Persistence baseline",
                                        line=dict(color=C_ACTUAL, width=2, dash="dot")))
                base_layout(fe, 360, f"MAE ({units[t]})")
                fe.update_yaxes(rangemode="tozero")
                st.plotly_chart(fe, use_container_width=True, config=PLOT_CFG)
            with e_right:
                tab = pd.DataFrame({"Day": sub["day"],
                                    "MAE": sub.get("model_mae"), "RMSE": sub.get("model_rmse"),
                                    "R²": sub.get("model_r2"), "Baseline MAE": sub.get("baseline_mae")})
                tab["Improvement vs baseline"] = (1 - tab["MAE"] / tab["Baseline MAE"]).map(
                    lambda x: f"{x:.0%}" if pd.notna(x) else "—")
                st.markdown(f"**{pick or 'Temperature'} – test metrics by forecast day ({units[t]})**")
                st.dataframe(tab.round(3), use_container_width=True, hide_index=True)
            rain = ev[ev["target"] == "rain"].sort_values("day")
            if not rain.empty:
                st.markdown("#### Rain (yes/no ≥ 0.1 mm) – classification metrics")
                cols = {"day": "Day", "model_brier": "Brier (model)", "baseline_brier": "Brier (climatology)",
                        "model_auc": "AUC", "model_precision": "Precision", "model_recall": "Recall",
                        "model_f1": "F1"}
                st.dataframe(rain[[c for c in cols if c in rain.columns]].rename(columns=cols).round(3),
                             use_container_width=True, hide_index=True)
                if "model_precision" not in rain.columns:
                    st.caption("Precision / recall / F1 appear after the next training run.")
            with st.expander("What do these metrics mean?"):
                st.markdown("""
- **MAE** – average size of the error (same unit as the variable). **RMSE** – like MAE but punishes large errors more.
  **R²** – share of the variation explained (1 = perfect).
- **Persistence baseline** – "the weather in *d* days equals the weather now". A useful model must beat it.
- **Brier score** – error of the rain probability (0 = perfect); compared with **climatology** (the usual rain rate).
- **AUC** – how well rain probabilities rank rainy hours above dry ones (0.5 = random, 1 = perfect).
- **Precision / Recall / F1** – rain predicted when probability ≥ 50%: precision = how many predicted rainy hours
  were rainy; recall = how many rainy hours were caught; F1 = their balance. These replace the brief's
  Precision@K / Recall@K, which are ranking metrics for recommendation systems.
""")
            st.caption("Test-set results of the latest training run, read from MLflow on DagsHub · same numbers as "
                       "reports/model_performance.pdf (pipeline/model_evaluation/results.py).")

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
- **Data quality:** rule checks + Great Expectations, quarantine for bad rows, profiling (Data quality tab)
- **EDA:** distributions, correlations, seasonality, rain sparsity, completeness (Exploratory analysis tab)
- **Lineage & metadata:** every run logged; each row traces to its raw payload
- **Versioning:** features (v1 → v3), DVC training snapshots, MLflow model registry
- **Evaluation:** test metrics vs baselines tracked in MLflow (Model evaluation tab)
- **Automation:** Prefect flow, run hourly by GitHub Actions

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
