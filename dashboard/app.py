"""
Weather forecast dashboard - 7-day hourly temperature forecast for ANY location.

Default location (Bengaluru) is the MONITORED city: forecasts come from the hourly
cloud pipeline stored in Neon, with live accuracy tracking.
Any other place is searched with Open-Meteo geocoding and forecast ON DEMAND with the
same quality rules, the same v2 features and the same global ML model.

Run locally:   streamlit run dashboard/app.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from ingestion.open_meteo import fetch_recent, geocode  # noqa: E402

REPO_URL = "https://github.com/2025ns04119/weather-forecast-dm"
DAGSHUB_URL = "https://dagshub.com/2025ns04119/weather-forecast-dm"

# Chart palette (validated: colour-blind safe, >= 3:1 contrast on the surface)
C_FORECAST = "#2a78d6"     # series 1 - blue
C_ACTUAL = "#eb6834"       # series 2 - orange
C_BAND = "rgba(42,120,214,0.13)"
C_GRID = "#e6e5e0"
C_TEXT = "#0b0b0b"
C_MUTED = "#6b6a65"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}
PLOT_CFG = {"displayModeBar": False, "responsive": True}

# WMO weather codes -> (day icon, night icon, label)
WMO = {0: ("☀️", "🌙", "Clear sky"), 1: ("🌤️", "🌙", "Mainly clear"), 2: ("⛅", "☁️", "Partly cloudy"),
       3: ("☁️", "☁️", "Overcast"), 45: ("🌫️", "🌫️", "Fog"), 48: ("🌫️", "🌫️", "Rime fog"),
       51: ("🌦️", "🌧️", "Light drizzle"), 53: ("🌦️", "🌧️", "Drizzle"), 55: ("🌧️", "🌧️", "Heavy drizzle"),
       56: ("🌧️", "🌧️", "Freezing drizzle"), 57: ("🌧️", "🌧️", "Freezing drizzle"),
       61: ("🌦️", "🌧️", "Light rain"), 63: ("🌧️", "🌧️", "Rain"), 65: ("🌧️", "🌧️", "Heavy rain"),
       66: ("🌧️", "🌧️", "Freezing rain"), 67: ("🌧️", "🌧️", "Freezing rain"),
       71: ("🌨️", "🌨️", "Light snow"), 73: ("🌨️", "🌨️", "Snow"), 75: ("❄️", "❄️", "Heavy snow"),
       77: ("🌨️", "🌨️", "Snow grains"), 80: ("🌦️", "🌧️", "Rain showers"), 81: ("🌧️", "🌧️", "Rain showers"),
       82: ("⛈️", "⛈️", "Violent showers"), 85: ("🌨️", "🌨️", "Snow showers"), 86: ("🌨️", "🌨️", "Snow showers"),
       95: ("⛈️", "⛈️", "Thunderstorm"), 96: ("⛈️", "⛈️", "Thunderstorm, hail"), 99: ("⛈️", "⛈️", "Thunderstorm, hail")}

st.set_page_config(page_title="Weather Forecast · ML", page_icon="🌤️",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1280px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  .subtitle {color: #6b6a65; font-size: .92rem; margin-top: -.5rem;}
  .pill {display: inline-flex; align-items: center; gap: .45rem; padding: .35rem .8rem;
         border-radius: 999px; border: 1px solid #e6e5e0; background: #ffffff;
         font-size: .85rem; color: #0b0b0b; white-space: nowrap;}
  .dot {width: .55rem; height: .55rem; border-radius: 50%; display: inline-block;}
  .hero {border-radius: 18px; padding: 1.3rem 1.5rem; color: #0b0b0b;
         background: linear-gradient(135deg, #e3eefc 0%, #f4f8fe 60%, #ffffff 100%);
         border: 1px solid #d3e2f7; height: 100%;}
  .hero-place {font-size: 1.15rem; font-weight: 650;}
  .hero-sub {color: #52514e; font-size: .85rem;}
  .hero-main {display: flex; align-items: center; gap: .9rem; margin: .6rem 0 .3rem;}
  .hero-icon {font-size: 3.4rem; line-height: 1;}
  .hero-temp {font-size: 3.6rem; font-weight: 700; line-height: 1; font-variant-numeric: tabular-nums;}
  .hero-cond {font-size: 1.05rem; font-weight: 600;}
  .hero-meta {display: flex; flex-wrap: wrap; gap: .35rem 1.1rem; color: #3a3936; font-size: .88rem;
              margin-top: .5rem;}
  .kpi {background: #ffffff; border: 1px solid #e6e5e0; border-radius: 12px;
        padding: .9rem 1.05rem; margin-bottom: .75rem;}
  .kpi-label {color: #6b6a65; font-size: .78rem; text-transform: uppercase; letter-spacing: .05em;}
  .kpi-value {font-size: 1.6rem; font-weight: 650; color: #0b0b0b; line-height: 1.25;
              font-variant-numeric: tabular-nums;}
  .kpi-sub {color: #52514e; font-size: .83rem;}
  .day-card {background: #ffffff; border: 1px solid #e6e5e0; border-radius: 12px;
             padding: .85rem .9rem; text-align: center;}
  .day-name {font-weight: 600; color: #0b0b0b;}
  .day-date {color: #6b6a65; font-size: .8rem; margin-bottom: .45rem;}
  .day-hi {font-size: 1.35rem; font-weight: 650; color: #0b0b0b; font-variant-numeric: tabular-nums;}
  .day-lo {font-size: .95rem; color: #52514e; font-variant-numeric: tabular-nums;}
  .track {position: relative; height: 6px; border-radius: 3px; background: #f0efec; margin: .55rem .2rem .2rem;}
  .range {position: absolute; top: 0; height: 6px; border-radius: 3px; background: #2a78d6;}
  .note {color: #6b6a65; font-size: .85rem;}
  .wrap {display: flex; flex-wrap: wrap; gap: .5rem;}
  .flow {display: flex; flex-wrap: wrap; align-items: center; gap: .4rem .35rem;}
  .step {background: #ffffff; border: 1px solid #d6d5cf; border-radius: 10px; padding: .55rem .75rem;
         font-size: .85rem; line-height: 1.25; color: #0b0b0b;}
  .step small {display: block; color: #6b6a65; font-size: .75rem;}
  .step.hl {border-color: #2a78d6; background: #eaf2fc;}
  .arrow {color: #8a8984; font-size: 1.1rem;}
  .chart-title {font-weight: 600; font-size: 1rem; color: #0b0b0b; margin: .6rem 0 -.2rem;}
  .footer {color: #6b6a65; font-size: .8rem; border-top: 1px solid #e6e5e0;
           margin-top: 2rem; padding-top: .8rem;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data access
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


@st.cache_resource(show_spinner="Loading the forecast model from the registry…")
def get_model():
    from common.model_loader import load_champion
    return load_champion()


@st.cache_data(ttl=86400, show_spinner=False)
def search_places(q: str) -> list[dict]:
    return geocode(q)


@st.cache_data(ttl=900, show_spinner=False)
def current_conditions(lat: float, lon: float):
    try:
        return fetch_recent(lat, lon, past_days=0, with_current=True).get("current")
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def on_demand(name: str, lat: float, lon: float) -> dict:
    from inference.on_demand import forecast_location
    model, version = get_model()
    out = forecast_location(model, name, lat, lon)
    out["model_version"] = version
    return out


def to_local(s: pd.Series, tz: str) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.tz_convert(tz).dt.tz_localize(None)


def stored_forecast(loc: str) -> pd.DataFrame:
    return query("""
        SELECT target_ts, predicted_temp, model_name, model_version, made_at
        FROM ml.predictions
        WHERE location = :loc
          AND made_at = (SELECT max(made_at) FROM ml.predictions WHERE location = :loc)
        ORDER BY target_ts""", loc=loc)


def stored_observed(loc: str, days: int = 7) -> pd.DataFrame:
    return query(f"""
        SELECT ts, temperature_2m, relative_humidity_2m, wind_speed_10m, cloud_cover, precipitation
        FROM clean.weather_hourly
        WHERE location = :loc AND ts >= now() - interval '{int(days)} days'
        ORDER BY ts""", loc=loc)


def accuracy_by_day(loc: str) -> pd.DataFrame:
    return query("""
        SELECT day_ahead, AVG(abs_err) AS mae, COUNT(*) AS n
        FROM (
            SELECT CEIL(EXTRACT(EPOCH FROM (p.target_ts - p.made_at)) / 86400.0)::int AS day_ahead,
                   ABS(p.predicted_temp - c.temperature_2m) AS abs_err
            FROM ml.predictions p
            JOIN clean.weather_hourly c ON c.location = p.location AND c.ts = p.target_ts
            WHERE p.location = :loc AND p.made_at > now() - interval '30 days'
        ) e
        WHERE day_ahead BETWEEN 1 AND 7
        GROUP BY day_ahead ORDER BY day_ahead""", loc=loc)


def verification(loc: str, hours: int = 72) -> pd.DataFrame:
    """For each past hour: the latest forecast issued at least 24 h before it."""
    return query(f"""
        SELECT DISTINCT ON (p.target_ts) p.target_ts, p.predicted_temp, c.temperature_2m
        FROM ml.predictions p
        JOIN clean.weather_hourly c ON c.location = p.location AND c.ts = p.target_ts
        WHERE p.location = :loc
          AND p.target_ts >= now() - interval '{int(hours)} hours'
          AND p.made_at <= p.target_ts - interval '24 hours'
        ORDER BY p.target_ts, p.made_at DESC""", loc=loc)


def layer_counts() -> dict:
    return query("""
        SELECT (SELECT count(*) FROM raw.weather_api)              AS raw_payloads,
               (SELECT count(*) FROM clean.weather_hourly)         AS clean_hours,
               (SELECT count(*) FROM features.weather_features)    AS feature_rows,
               (SELECT count(*) FROM quarantine.weather_rejects)   AS quarantined,
               (SELECT count(*) FROM ml.predictions)               AS forecast_rows,
               (SELECT count(*) FROM clean.locations)              AS locations""").iloc[0].to_dict()


def pipeline_runs(limit: int = 200) -> pd.DataFrame:
    return query(f"""
        SELECT run_id, pipeline, started_at, finished_at, status, rows_in, rows_out, notes
        FROM ml.pipeline_runs ORDER BY run_id DESC LIMIT {int(limit)}""")


def drift_reports() -> pd.DataFrame:
    try:
        return query("SELECT run_at, drift_detected, drift_share FROM ml.drift_reports "
                     "ORDER BY run_at DESC LIMIT 30")
    except Exception:
        return pd.DataFrame()


# ------------------------------------------------------------------ helpers
def fmt_age(ts: pd.Timestamp) -> str:
    mins = int((pd.Timestamp.now(tz="UTC") - ts).total_seconds() // 60)
    if mins < 60:
        return f"{mins} min ago"
    if mins < 48 * 60:
        return f"{mins // 60} h {mins % 60} min ago"
    return f"{mins // 1440} days ago"


def freshness(made_at: pd.Timestamp) -> tuple[str, str]:
    age_h = (pd.Timestamp.now(tz="UTC") - made_at).total_seconds() / 3600
    if age_h <= 2:
        return "good", f"Live · updated {fmt_age(made_at)}"
    if age_h <= 6:
        return "warning", f"Delayed · updated {fmt_age(made_at)}"
    return "critical", f"Stale · last forecast {fmt_age(made_at)}"


def pill(status: str, label: str) -> str:
    icon = {"good": "✓", "warning": "!", "critical": "✕"}[status]
    return (f'<span class="pill"><span class="dot" style="background:{STATUS[status]}"></span>'
            f'<b>{icon}</b> {label}</span>')


def kpi(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>')


def base_layout(fig: go.Figure, height: int = 420, y_title: str = "Temperature (°C)") -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=C_TEXT, size=13),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=C_GRID, font_color=C_TEXT),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=C_MUTED)),
    )
    fig.update_xaxes(showgrid=False, linecolor=C_GRID, tickfont=dict(color=C_MUTED),
                     showspikes=True, spikecolor=C_MUTED, spikethickness=1, spikedash="dot",
                     spikemode="across")
    fig.update_yaxes(title=y_title, gridcolor=C_GRID, zeroline=False,
                     tickfont=dict(color=C_MUTED), title_font=dict(color=C_MUTED))
    return fig


def now_marker(fig: go.Figure, now_local: pd.Timestamp):
    fig.add_shape(type="line", x0=now_local, x1=now_local, y0=0, y1=1, yref="paper",
                  line=dict(color=C_MUTED, width=1, dash="dot"))
    fig.add_annotation(x=now_local, y=0.98, yref="paper", text="Now", showarrow=False,
                       xanchor="left", xshift=4, font=dict(color=C_MUTED, size=12))


def place_label(p: dict) -> str:
    parts = [p.get("name"), p.get("admin1"), p.get("country")]
    seen, out = set(), []
    for x in parts:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return ", ".join(out)


def is_monitored(p: dict) -> bool:
    return (p["name"] == DEFAULT_LOCATION["name"]
            and abs(p["latitude"] - DEFAULT_LOCATION["latitude"]) < 0.3
            and abs(p["longitude"] - DEFAULT_LOCATION["longitude"]) < 0.3)


def choose_place(r: dict):
    """Button callback: switch location and clear the search box."""
    st.session_state.place = {
        "name": r["name"], "admin1": r.get("admin1"), "country": r.get("country"),
        "latitude": float(r["latitude"]), "longitude": float(r["longitude"]),
        "timezone": r.get("timezone") or "UTC"}
    st.session_state.search_q = ""


def go_home():
    st.session_state.place = dict(DEFAULT_LOCATION, admin1="Karnataka")
    st.session_state.search_q = ""


# ------------------------------------------------------------------ location state
if "place" not in st.session_state:
    st.session_state.place = dict(DEFAULT_LOCATION, admin1="Karnataka")
place = st.session_state.place
tz = place.get("timezone") or "UTC"
monitored = is_monitored(place)

# ------------------------------------------------------------------ header + search
h_left, h_right = st.columns([3, 2], vertical_alignment="bottom")
with h_left:
    st.title("🌤️ Weather Forecast")
    st.markdown('<div class="subtitle">7-day hourly temperature forecast by a machine-learning '
                'model · live Open-Meteo data · search any place on Earth</div>',
                unsafe_allow_html=True)
with h_right:
    q = st.text_input("Search location", placeholder="🔍  Search a city or place, e.g. Chennai",
                      label_visibility="collapsed", key="search_q")

if q and len(q.strip()) >= 2:
    try:
        results = search_places(q.strip())
    except Exception as e:
        results = []
        st.warning(f"Location search is unavailable right now ({type(e).__name__}).")
    if results:
        st.caption(f"{len(results)} matching place{'s' if len(results) != 1 else ''} - tap one to see its forecast")
        sug = st.columns(min(len(results), 4))
        for i, r in enumerate(results):
            sug[i % len(sug)].button(f"📍 {place_label(r)}", key=f"pick_{i}_{r['latitude']}_{r['longitude']}",
                                     use_container_width=True, on_click=choose_place, args=(r,))
    elif q.strip():
        st.caption("No matching places found.")

if not monitored:
    st.button(f"🏠 Back to {DEFAULT_LOCATION['name']} (home)", on_click=go_home)

# ------------------------------------------------------------------ load data for the place
now_local = pd.Timestamp.now(tz=tz).tz_localize(None)
fc = obs = None
made_at = None
model_label = ""
acc = pd.DataFrame()
rejected = 0
current = current_conditions(place["latitude"], place["longitude"])

if monitored and _db_url():
    try:
        fc = stored_forecast(place["name"])
        obs = stored_observed(place["name"])
        acc = accuracy_by_day(place["name"])
        if not fc.empty:
            made_at = pd.to_datetime(fc["made_at"].iloc[0], utc=True)
            model_label = f"{fc['model_name'].iloc[0]} v{fc['model_version'].iloc[0]}"
            fc["day_ahead"] = ((pd.to_datetime(fc["target_ts"], utc=True) - made_at)
                               .dt.total_seconds() / 86400).apply(lambda x: max(1, int(-(-x // 1))))
    except Exception as e:
        st.error(f"Could not reach the database: {e}")
        st.stop()
else:
    with st.spinner(f"Fetching live data and forecasting {place['name']}…"):
        try:
            res = on_demand(place["name"], place["latitude"], place["longitude"])
        except Exception as e:
            st.error(f"Could not forecast {place_label(place)}: {e}")
            st.stop()
    fc, obs = res["forecast"], res["observed"]
    made_at, rejected = res["made_at"], res["rejected"]
    model_label = f"weather-temp-7day v{res['model_version']}"
    current = current or res.get("current")

if fc is None or fc.empty:
    st.info("No forecast yet. Run the pipeline once (GitHub → Actions → Run workflow).")
    st.stop()

fc = fc.copy()
fc["time"] = to_local(fc["target_ts"], tz)
obs = obs.copy() if obs is not None else pd.DataFrame(columns=["ts", "temperature_2m"])
if not obs.empty:
    obs["time"] = to_local(obs["ts"], tz)

# daily summary (today = observed so far + forecast)
past_today = obs[obs["time"].dt.date == now_local.date()][["time", "temperature_2m"]] if not obs.empty \
    else pd.DataFrame(columns=["time", "temperature_2m"])
combined = pd.concat([past_today.rename(columns={"temperature_2m": "temp"}),
                      fc[["time", "predicted_temp"]].rename(columns={"predicted_temp": "temp"})])
combined["date"] = combined["time"].dt.date
daily = combined.groupby("date")["temp"].agg(["min", "max", "count"]).reset_index()
daily = daily[(daily["date"] == now_local.date()) | (daily["count"] >= 18)].head(7)
today = daily[daily["date"] == now_local.date()]
tomorrow = daily[daily["date"] == (now_local + pd.Timedelta(days=1)).date()]
mae_by_day = dict(zip(acc["day_ahead"], acc["mae"])) if not acc.empty else {}

# ------------------------------------------------------------------ hero + KPIs
hero_col, kpi_col = st.columns([3, 2])
with hero_col:
    if current:
        code = int(current.get("weather_code", 0) or 0)
        day_icon, night_icon, cond = WMO.get(code, ("🌡️", "🌡️", "—"))
        icon = day_icon if current.get("is_day", 1) else night_icon
        temp_now = current.get("temperature_2m")
        meta = [f"Feels like {current.get('apparent_temperature', temp_now):.0f}°",
                f"Humidity {current.get('relative_humidity_2m', 0):.0f}%",
                f"Wind {current.get('wind_speed_10m', 0):.0f} km/h"]
    else:
        icon, cond = "🌡️", "Latest observation"
        temp_now = obs["temperature_2m"].iloc[-1] if not obs.empty else fc["predicted_temp"].iloc[0]
        meta = []
    if not today.empty:
        meta.insert(0, f"H {today['max'].iloc[0]:.0f}° · L {today['min'].iloc[0]:.0f}°")
    status, label = freshness(made_at) if monitored else ("good", "Forecast generated just now")
    st.markdown(f"""
      <div class="hero">
        <div class="hero-place">📍 {place_label(place)}</div>
        <div class="hero-sub">{now_local:%A, %d %b · %H:%M} local time · {place['latitude']:.2f}°, {place['longitude']:.2f}°</div>
        <div class="hero-main">
          <div class="hero-icon">{icon}</div>
          <div><div class="hero-temp">{temp_now:.0f}°C</div><div class="hero-cond">{cond}</div></div>
        </div>
        <div class="hero-meta">{''.join(f'<span>{m}</span>' for m in meta)}</div>
        <div style="margin-top:.8rem">{pill(status, label)}</div>
      </div>""", unsafe_allow_html=True)
with kpi_col:
    if not tomorrow.empty:
        r = tomorrow.iloc[0]
        st.markdown(kpi("Tomorrow", f"{r['max']:.0f}° / {r['min']:.0f}°", "High / low forecast"),
                    unsafe_allow_html=True)
    if monitored:
        if 1 in mae_by_day:
            st.markdown(kpi(f"Model · {model_label}", f"±{mae_by_day[1]:.1f}°C",
                            "Live next-day error (MAE), measured hourly"), unsafe_allow_html=True)
        else:
            st.markdown(kpi(f"Model · {model_label}", "Global", "Live accuracy builds up over 24 h"),
                        unsafe_allow_html=True)
    else:
        st.markdown(kpi(f"Model · {model_label}", "On demand",
                        f"Same quality rules & features as the pipeline · {rejected} rows rejected"),
                    unsafe_allow_html=True)

st.write("")

# ------------------------------------------------------------------ tabs
t_fc, t_acc, t_pipe, t_about = st.tabs(
    ["📅  7-Day Forecast", "🎯  Forecast Accuracy", "🗄️  Data Pipeline", "ℹ️  About"])

# ---------- forecast
with t_fc:
    if len(daily):
        cols = st.columns(len(daily))
        wk_min, wk_max = daily["min"].min(), daily["max"].max()
        span = max(wk_max - wk_min, 1e-6)
        for col, (_, r) in zip(cols, daily.iterrows()):
            d = pd.Timestamp(r["date"])
            name = "Today" if r["date"] == now_local.date() else d.strftime("%a")
            left_pct = (r["min"] - wk_min) / span * 100
            width_pct = max((r["max"] - r["min"]) / span * 100, 4)
            col.markdown(f"""
                <div class="day-card">
                  <div class="day-name">{name}</div>
                  <div class="day-date">{d:%d %b}</div>
                  <div class="day-hi">{r['max']:.0f}°</div>
                  <div class="day-lo">{r['min']:.0f}°</div>
                  <div class="track"><div class="range" style="left:{left_pct:.0f}%;width:{width_pct:.0f}%"></div></div>
                </div>""", unsafe_allow_html=True)

    st.markdown('<div class="chart-title">Hourly temperature · last 48 h observed and next 7 days forecast</div>',
                unsafe_allow_html=True)
    fig = go.Figure()
    if mae_by_day:
        band = fc.assign(err=fc["day_ahead"].map(mae_by_day)).dropna(subset=["err"])
        if not band.empty:
            fig.add_trace(go.Scatter(x=band["time"], y=band["predicted_temp"] + band["err"],
                                     mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=band["time"], y=band["predicted_temp"] - band["err"],
                                     mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor=C_BAND, hoverinfo="skip", name="Typical error range"))
    if not obs.empty:
        hist = obs[obs["time"] >= now_local - pd.Timedelta(hours=48)]
        fig.add_trace(go.Scatter(x=hist["time"], y=hist["temperature_2m"], mode="lines",
                                 name="Observed", line=dict(color=C_ACTUAL, width=2),
                                 hovertemplate="Observed %{y:.1f}°C<extra></extra>"))
    fig.add_trace(go.Scatter(x=fc["time"], y=fc["predicted_temp"], mode="lines",
                             name="Forecast", line=dict(color=C_FORECAST, width=2),
                             customdata=fc["day_ahead"],
                             hovertemplate="Forecast %{y:.1f}°C (day %{customdata})<extra></extra>"))
    base_layout(fig, 430)
    now_marker(fig, now_local)
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
    src = ("hourly cloud pipeline (stored in Neon)" if monitored
           else "on demand from live Open-Meteo data")
    st.markdown(f'<div class="note">Forecast generated {pd.Timestamp(made_at).tz_convert(tz):%d %b %Y, %H:%M} '
                f'local time by <b>{model_label}</b>, {src}. Times shown in {tz}.'
                + (' The shaded band shows the typical error measured on past forecasts.' if mae_by_day else '')
                + '</div>', unsafe_allow_html=True)

    with st.expander("View hourly forecast as a table"):
        tbl = fc[["time", "predicted_temp", "day_ahead"]].rename(columns={
            "time": "Local time", "predicted_temp": "Forecast (°C)", "day_ahead": "Day ahead"})
        st.dataframe(tbl.style.format({"Forecast (°C)": "{:.1f}"}),
                     use_container_width=True, hide_index=True, height=320)
        st.download_button("Download CSV", tbl.to_csv(index=False).encode(),
                           file_name=f"{place['name'].lower().replace(' ', '_')}_7day_forecast.csv",
                           mime="text/csv")

# ---------- accuracy
with t_acc:
    if not monitored:
        st.info(f"Live accuracy is tracked for **{DEFAULT_LOCATION['name']}**, the monitored city "
                f"whose forecasts are stored every hour. Forecasts for other places are generated on "
                f"demand and not stored, so they can't be verified later.")
    else:
        ver = verification(place["name"])
        c_left, c_right = st.columns([3, 2])
        with c_left:
            if ver.empty:
                st.info("Accuracy appears once forecast hours have passed (after about 24 h of hourly runs).")
            else:
                ver["time"] = to_local(ver["target_ts"], tz)
                mae = (ver["predicted_temp"] - ver["temperature_2m"]).abs().mean()
                st.markdown(f'<div class="chart-title">Forecast vs observed · last 72 h · MAE {mae:.2f}°C</div>',
                            unsafe_allow_html=True)
                f2 = go.Figure()
                f2.add_trace(go.Scatter(x=ver["time"], y=ver["temperature_2m"], mode="lines",
                                        name="Observed", line=dict(color=C_ACTUAL, width=2),
                                        hovertemplate="Observed %{y:.1f}°C<extra></extra>"))
                f2.add_trace(go.Scatter(x=ver["time"], y=ver["predicted_temp"], mode="lines",
                                        name="Forecast issued ≥ 24 h earlier",
                                        line=dict(color=C_FORECAST, width=2),
                                        hovertemplate="Forecast %{y:.1f}°C<extra></extra>"))
                base_layout(f2, 380)
                st.plotly_chart(f2, use_container_width=True, config=PLOT_CFG)
        with c_right:
            if acc.empty:
                st.info("Error by forecast day appears after the first forecasts have been verified.")
            else:
                st.markdown('<div class="chart-title">Average error by forecast day</div>',
                            unsafe_allow_html=True)
                f3 = go.Figure(go.Bar(
                    x=[f"Day {d}" for d in acc["day_ahead"]], y=acc["mae"],
                    marker=dict(color=C_FORECAST, cornerradius=4),
                    text=[f"{v:.2f}°" for v in acc["mae"]], textposition="outside",
                    textfont=dict(color=C_TEXT), customdata=acc["n"],
                    hovertemplate="%{x}: MAE %{y:.2f}°C<br>%{customdata} verified hours<extra></extra>"))
                base_layout(f3, 380, "Mean absolute error (°C)")
                f3.update_layout(hovermode="closest", bargap=0.35, showlegend=False)
                f3.update_xaxes(showspikes=False)
                f3.update_yaxes(rangemode="tozero")
                st.plotly_chart(f3, use_container_width=True, config=PLOT_CFG)
        st.markdown('<div class="note">Every forecast hour is compared with the observed temperature once '
                    'that hour has passed. Error grows with lead time, which is expected for weather '
                    'forecasting.</div>', unsafe_allow_html=True)

# ---------- data pipeline
with t_pipe:
    if not _db_url():
        st.info("Connect the database (NEON_DB_URL) to see pipeline statistics.")
    else:
        counts = layer_counts()
        m = st.columns(6)
        cards = [("Locations", counts["locations"], "Location dimension table"),
                 ("Bronze · raw payloads", counts["raw_payloads"], "API responses stored as-is"),
                 ("Silver · clean hours", counts["clean_hours"], "Validated hourly rows"),
                 ("Gold · feature rows", counts["feature_rows"], "Feature versions v1 + v2"),
                 ("Quarantined", counts["quarantined"], "Failed quality rules"),
                 ("Forecast rows", counts["forecast_rows"], "Stored predictions")]
        for col, (lab, val, sub) in zip(m, cards):
            col.markdown(kpi(lab, f"{int(val):,}", sub), unsafe_allow_html=True)

        st.markdown("#### Data flow")
        steps = [("Open-Meteo API", "live + archive", ""), ("Bronze", "raw.weather_api", ""),
                 ("Validation", "rules + Great Expectations", ""), ("Silver", "clean.weather_hourly", ""),
                 ("Gold", "features v2", ""), ("Global model", "MLflow registry · DVC data", ""),
                 ("Forecast", "stored hourly / on demand", ""), ("Dashboard", "this app", "hl")]
        html = '<span class="arrow">→</span>'.join(
            f'<div class="step {c}">{t}<small>{sub}</small></div>' for t, sub, c in steps)
        st.markdown(f'<div class="flow">{html}</div>', unsafe_allow_html=True)
        st.markdown('<div class="note" style="margin-top:.5rem">Rows failing validation go to '
                    '<b>quarantine.weather_rejects</b>. Searched places run through the same validation '
                    'rules and feature code in memory, so training and serving never drift apart.</div>',
                    unsafe_allow_html=True)

        st.markdown("#### Pipeline health")
        runs = pipeline_runs()
        if runs.empty:
            st.info("No pipeline runs logged yet.")
        else:
            runs["started"] = to_local(runs["started_at"], DEFAULT_LOCATION["timezone"])
            runs["duration_s"] = (pd.to_datetime(runs["finished_at"], utc=True)
                                  - pd.to_datetime(runs["started_at"], utc=True)).dt.total_seconds().round(1)
            last = runs.sort_values("run_id").groupby("pipeline").tail(1).sort_values("pipeline")
            pills = "".join(pill("good" if r["status"] == "success" else "critical",
                                 f"{r['pipeline']} · {r['status']} · {r['started']:%d %b %H:%M}")
                            for _, r in last.iterrows())
            st.markdown(f'<div class="wrap">{pills}</div>', unsafe_allow_html=True)
            st.write("")
            show = runs[["started", "pipeline", "status", "rows_in", "rows_out", "duration_s", "notes"]].rename(
                columns={"started": "Started (IST)", "pipeline": "Pipeline", "status": "Status",
                         "rows_in": "Rows in", "rows_out": "Rows out", "duration_s": "Duration (s)",
                         "notes": "Notes / lineage"})
            st.dataframe(show.head(60), use_container_width=True, hide_index=True, height=320)

        drift = drift_reports()
        st.markdown("#### Data drift")
        if drift.empty:
            st.info("Drift monitoring (Evidently) results will appear here once it is enabled.")
        else:
            d = drift.iloc[0]
            st.markdown(pill("critical" if d["drift_detected"] else "good",
                             f"{'Drift detected' if d['drift_detected'] else 'No drift'} · "
                             f"{(d['drift_share'] or 0):.0%} of features drifted · "
                             f"{pd.Timestamp(d['run_at']).tz_convert(DEFAULT_LOCATION['timezone']):%d %b %H:%M} IST"),
                        unsafe_allow_html=True)

# ---------- about
with t_about:
    a1, a2 = st.columns(2)
    with a1:
        st.markdown(f"""
#### What this is
An end-to-end **data management for machine learning** project. Live weather data is
ingested, validated, cleaned and turned into features, and a versioned **global ML model**
forecasts the temperature for the next 7 days, hour by hour, for **any place on Earth**.

- **{DEFAULT_LOCATION['name']}** is the monitored city: forecast every hour in the cloud,
  stored in the database and verified against what actually happened.
- **Any other place** is forecast on demand with the same validation rules,
  the same features and the same model.

#### How the forecast works
- **Direct multi-horizon strategy:** one model per forecast day (day 1 … day 7)
- Trained on 30+ cities across every continent; best of Linear Regression,
  Random Forest and XGBoost chosen per day
- Location-agnostic features: local solar hour, month, latitude, temperature lags
  (1 h, 3 h, 24 h), rolling means, humidity, pressure, wind, cloud cover
- Time-based train/test split with an embargo gap (no future leakage)
""")
    with a2:
        st.markdown(f"""
#### Data management concepts
- **Medallion architecture:** bronze → silver → gold layers
- **Data modelling:** location dimension table
- **Data quality:** rule checks + Great Expectations, quarantine for bad rows
- **Lineage & metadata:** every run logged; each row traces to its raw payload
- **Versioning:** feature versions (v1 → v2), DVC for training data, MLflow registry for models
- **Automation:** GitHub Actions runs the pipeline hourly

#### Tools (all free & open source)
Python · PostgreSQL (Neon) · Great Expectations · DVC · MLflow (DagsHub) ·
scikit-learn · XGBoost · GitHub Actions · Streamlit · Plotly

[Source code on GitHub]({REPO_URL}) · [Experiments & models on DagsHub]({DAGSHUB_URL})
""")

st.markdown("""<div class="footer">
Weather data and geocoding by <a href="https://open-meteo.com" target="_blank">Open-Meteo.com</a>
(<a href="https://creativecommons.org/licenses/by/4.0/" target="_blank">CC BY 4.0</a>), based on
ECMWF, NOAA and DWD models and ERA5 reanalysis (Copernicus). Current conditions are Open-Meteo
observations; temperature forecasts come from this project's own ML model.</div>""",
            unsafe_allow_html=True)
