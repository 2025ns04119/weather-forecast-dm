"""
Builds everything the Weather screen shows - ALL computed by this project:

    temperature, humidity, wind, cloud, rain chance   -> the ML model (weather-7day)
    current conditions                                -> latest validated observation (silver layer)
    condition text + icon, feels like, UV, sunrise    -> common/derived.py calculations
"""
import numpy as np
import pandas as pd

from common.derived import (apparent_temperature, compass, condition, solar_elevation,
                            sun_times, uv_index, uv_level)

# ---------------------------------------------------------------- icons (inline SVG)
_CLOUD = '<path stroke="#6E7F96" fill="#FFFFFF" d="M7.4 18.6h9.4a4.1 4.1 0 0 0 .4-8.18 5.6 5.6 0 0 0-10.8 1.6A3.3 3.3 0 0 0 7.4 18.6z"/>'
_CLOUD_HI = '<path stroke="#6E7F96" fill="#FFFFFF" d="M7.4 15.4h9.4a4.1 4.1 0 0 0 .4-8.18 5.6 5.6 0 0 0-10.8 1.6A3.3 3.3 0 0 0 7.4 15.4z"/>'
_CLOUD_SM = '<path stroke="#6E7F96" fill="#FFFFFF" d="M10.4 20h7.4a3.6 3.6 0 0 0 .3-7.18 4.9 4.9 0 0 0-9.4 1.35A2.95 2.95 0 0 0 10.4 20z"/>'
ICONS = {
    "sun": '<circle cx="12" cy="12" r="4.3" stroke="#E39A23"/><path stroke="#E39A23" d="M12 2.6v2.1M12 19.3v2.1M2.6 12h2.1M19.3 12h2.1M5.4 5.4l1.5 1.5M17.1 17.1l1.5 1.5M5.4 18.6l1.5-1.5M17.1 6.9l1.5-1.5"/>',
    "partly": '<circle cx="8.6" cy="8.4" r="3.1" stroke="#E39A23"/><path stroke="#E39A23" d="M8.6 2.4v1.4M2.6 8.4H4M4.4 4.2l1 1M12.8 4.2l-1 1"/>' + _CLOUD_SM,
    "cloud": _CLOUD,
    "rain": _CLOUD_HI + '<path stroke="#2A78D6" d="M8.4 18.2l-1 2.4M12.4 18.2l-1 2.4M16.4 18.2l-1 2.4"/>',
    "moon": '<path stroke="#5867C4" d="M19.6 14.6A7.9 7.9 0 1 1 9.4 4.4a6.4 6.4 0 0 0 10.2 10.2z"/>',
    "moonpartly": '<path stroke="#5867C4" d="M12.6 8.2A4.6 4.6 0 1 1 6.7 2.3a3.7 3.7 0 0 0 5.9 5.9z"/>' + _CLOUD_SM,
}


def icon(kind: str, size: int = 28) -> str:
    return (f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS.get(kind, ICONS["cloud"])}</svg>')


def hour_label(t: pd.Timestamp) -> str:
    """'7 PM', or '7:30 PM' in half-hour time zones such as India (UTC+5:30)."""
    h = t.hour % 12 or 12
    mm = f":{t.minute:02d}" if t.minute else ""
    return f"{h}{mm} {'AM' if t.hour < 12 else 'PM'}"


def deg(x) -> str:
    """Rounded degrees without a '-0'."""
    return str(int(round(float(x))) + 0)


def clock(t: pd.Timestamp) -> str:
    """'6:17 AM' (portable: Windows has no %-I)."""
    return f"{t.hour % 12 or 12}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'}"


def day_month(t: pd.Timestamp) -> str:
    """'9 Nov' (portable: Windows has no %-d)."""
    t = pd.Timestamp(t)
    return f"{t.day} {t:%b}"


def _local(s, tz):
    return pd.to_datetime(s, utc=True).dt.tz_convert(tz).dt.tz_localize(None)


# ---------------------------------------------------------------- assemble the screen data
def build_view(place: dict, fc: pd.DataFrame, obs: pd.DataFrame, made_at) -> dict:
    tz = place.get("timezone") or "UTC"
    lat, lon = place["latitude"], place["longitude"]
    now_utc = pd.Timestamp.now(tz="UTC")
    now = now_utc.tz_convert(tz).tz_localize(None)

    fc = fc.copy()
    fc["time"] = _local(fc["target_ts"], tz)
    fc["elev"] = solar_elevation(lat, lon, fc["target_ts"])
    obs = obs.copy().sort_values("ts")
    obs = obs[pd.to_datetime(obs["ts"], utc=True) <= now_utc]
    obs["time"] = _local(obs["ts"], tz)

    # ---- current conditions: latest validated observation
    cur = obs.iloc[-1]
    elev_now = float(solar_elevation(lat, lon, [now_utc])[0])
    is_day = elev_now > 0
    cur_kind, cur_label = condition(cur.get("cloud_cover"), None, is_day, rain_mm=cur.get("precipitation"))
    feels = float(apparent_temperature(cur["temperature_2m"], cur["relative_humidity_2m"], cur["wind_speed_10m"]))

    # ---- daily aggregates: today = observed so far + forecast
    today_obs = obs[obs["time"].dt.date == now.date()]
    hourly_all = pd.concat([
        pd.DataFrame({"time": today_obs["time"], "temp": today_obs["temperature_2m"],
                      "cloud": today_obs["cloud_cover"], "rain": np.nan,
                      "elev": solar_elevation(lat, lon, today_obs["ts"])}),
        pd.DataFrame({"time": fc["time"], "temp": fc["predicted_temp"], "cloud": fc["cloud"],
                      "rain": fc["rain_prob"], "elev": fc["elev"]}),
    ])
    hourly_all["date"] = hourly_all["time"].dt.date
    hourly_all["uv"] = uv_index(hourly_all["elev"], hourly_all["cloud"])

    days = []
    for date, g in hourly_all.groupby("date"):
        if date != now.date() and len(g) < 18:
            continue
        daytime = g[g["elev"] > 0]
        cloud_day = (daytime if len(daytime) else g)["cloud"].mean()
        rain_max = g["rain"].max(skipna=True)
        kind, label = condition(cloud_day, 0 if pd.isna(rain_max) else rain_max, True)
        days.append({"date": date, "hi": g["temp"].max(), "lo": g["temp"].min(), "kind": kind, "cond": label,
                     "rain": 0 if pd.isna(rain_max) else rain_max, "uv": g["uv"].max()})
        if len(days) == 7:
            break

    # ---- next 24 hours
    upcoming = fc[fc["time"] > now].head(24)
    hours = [{"t": "Now", "kind": cur_kind, "temp": cur["temperature_2m"], "rain": None}]
    for r in upcoming.itertuples(index=False):
        k, _ = condition(r.cloud, r.rain_prob, r.elev > 0)
        hours.append({"t": hour_label(r.time), "kind": k, "temp": r.predicted_temp, "rain": r.rain_prob})

    # ---- detail tiles
    rest_today = fc[fc["time"].dt.date == now.date()]
    rain_rest = float(rest_today["rain_prob"].max()) if len(rest_today) else (days[1]["rain"] if len(days) > 1 else 0)
    rise, sset = sun_times(lat, lon, now.date(), tz)
    rise_sub = "Today"
    if rise is not None and rise < now:
        rise, _ = sun_times(lat, lon, (now + pd.Timedelta(days=1)).date(), tz)
        rise_sub = "Tomorrow"
    uv_today = days[0]["uv"] if days else float(uv_index(elev_now, cur.get("cloud_cover")))
    dew = cur.get("dew_point_2m")
    details = [
        ("Humidity", f"{cur['relative_humidity_2m']:.0f}%", f"Dew point {deg(dew)}°" if dew is not None and not pd.isna(dew) else ""),
        ("Wind", f"{cur['wind_speed_10m']:.0f} km/h", compass(cur.get("wind_direction_10m"))),
        ("UV index", f"{uv_today:.0f}", uv_level(uv_today)),
        ("Rain chance", f"{rain_rest * 100:.0f}%", "Rest of today" if len(rest_today) else "Tomorrow"),
        ("Sunrise", clock(rise) if rise is not None else "—", rise_sub if rise is not None else "Polar day/night"),
        ("Sunset", clock(sset) if sset is not None else "—", "Today" if sset is not None else ""),
    ]

    today = days[0] if days else None
    return {"place": place, "now": now, "cur_temp": cur["temperature_2m"], "cur_kind": cur_kind,
            "cur_label": cur_label, "feels": feels, "today": today, "hours": hours, "days": days,
            "details": details, "made_at": pd.Timestamp(made_at).tz_convert(tz).tz_localize(None)}


# ---------------------------------------------------------------- HTML pieces
def now_card(v: dict) -> str:
    p, t = v["place"], v["today"]
    region = ", ".join(x for x in [p.get("admin1"), p.get("country")] if x)
    hl = (f'<span class="chip hi">High {deg(t["hi"])}°</span><span class="chip lo">Low {deg(t["lo"])}°</span>'
          if t else "")
    tiles = "".join(f'<div class="tile"><span class="tl">{a}</span><span class="tv">{b}</span>'
                    f'<span class="ts">{c}</span></div>' for a, b, c in v["details"])
    return (f'<section class="card now" aria-label="Current weather">'
            f'<div class="place"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2A78D6" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 21s-6.5-5.6-6.5-11a6.5 6.5 0 0 1 13 0c0 5.4-6.5 11-6.5 11z"/>'
            f'<circle cx="12" cy="10" r="2.4"/></svg><h1>{p["name"]}</h1></div>'
            f'<div class="sub">{region} · {v["now"]:%a} {day_month(v["now"])}, {clock(v["now"])}</div>'
            f'<div class="nowrow">{icon(v["cur_kind"], 84)}<div class="nowtxt">'
            f'<span class="big">{deg(v["cur_temp"])}°</span><span class="cond">{v["cur_label"]}</span>'
            f'<span class="feels">Feels like {deg(v["feels"])}°</span></div></div>'
            f'<div class="chips">{hl}</div><div class="tiles">{tiles}</div></section>')


def hourly_html(v: dict) -> str:
    hrs = v["hours"]
    n, w = len(hrs), 56
    temps = [h["temp"] for h in hrs]
    lo, hi = min(temps), max(temps)
    span = max(hi - lo, 1e-6)
    xs = [w / 2 + i * w for i in range(n)]
    ys = [100 - (t - lo) / span * 80 for t in temps]
    pts = " ".join(f"{x:.0f},{y:.1f}" for x, y in zip(xs, ys))
    area = f"M{xs[0]:.0f} {ys[0]:.1f} " + " ".join(f"L{x:.0f} {y:.1f}" for x, y in zip(xs[1:], ys[1:])) \
        + f" L{xs[-1]:.0f} 120 L{xs[0]:.0f} 120 Z"
    cells = []
    for i, h in enumerate(hrs):
        rain_txt = "" if h["rain"] is None or h["rain"] < 0.1 else "{:.0f}%".format(h["rain"] * 100)
        first = " first" if i == 0 else ""
        cells.append(f'<div class="hc{first}"><span class="ht">{h["t"]}</span>{icon(h["kind"], 30)}'
                     f'<span class="hv">{deg(h["temp"])}°</span><span class="hr">{rain_txt}</span></div>')
    cells = "".join(cells)
    return (f'<div class="hscroll"><div class="hinner" style="min-width:{n * w}px">'
            f'<svg viewBox="0 0 {n * w} 120" preserveAspectRatio="none" width="100%" height="210" aria-hidden="true">'
            f'<path d="{area}" fill="#2A78D6" fill-opacity="0.08"/>'
            f'<polyline points="{pts}" fill="none" stroke="#2A78D6" stroke-width="2.5" stroke-linejoin="round" '
            f'vector-effect="non-scaling-stroke"/></svg>'
            f'<div class="hgrid" style="grid-template-columns:repeat({n},minmax(0,1fr))">{cells}</div></div></div>'
            f'<div class="cap">Next 24 hours · scroll sideways for more · rain chance shown when 10% or higher</div>')


def week_html(v: dict) -> str:
    days = v["days"]
    if not days:
        return ""
    wmin, wmax = min(d["lo"] for d in days), max(d["hi"] for d in days)
    span = max(wmax - wmin, 1e-6)
    rows = []
    for i, d in enumerate(days):
        name = "Today" if i == 0 else pd.Timestamp(d["date"]).strftime("%a")
        left, width = (d["lo"] - wmin) / span * 100, max((d["hi"] - d["lo"]) / span * 100, 8)
        rows.append(
            f'<div class="wk"><span class="wd"><b>{name}</b><small>{day_month(d["date"])}</small></span>'
            f'{icon(d["kind"], 30)}<span class="wc">{d["cond"]}</span>'
            f'<span class="wr"><svg width="12" height="12" viewBox="0 0 24 24" fill="#2A78D6" aria-hidden="true">'
            f'<path d="M12 3s-6 6.6-6 11a6 6 0 0 0 12 0c0-4.4-6-11-6-11z"/></svg>{d["rain"] * 100:.0f}%</span>'
            f'<span class="wl">{deg(d["lo"])}°</span><span class="bar"><span style="left:{left:.0f}%;width:{width:.0f}%"></span></span>'
            f'<span class="wh">{deg(d["hi"])}°</span></div>')
    return f'<div class="week">{"".join(rows)}</div>'


CSS = """
<style>
  .block-container {padding-top: 3.6rem; padding-bottom: 2rem; max-width: 1260px;}
  .stApp, .stMarkdown, .stButton button, .stApp input {font-family: Manrope, "Segoe UI", system-ui, sans-serif;}
  [data-testid="stIconMaterial"] {font-family: "Material Symbols Rounded" !important;}
  .stApp {background: #F3F6FA;}
  .brand {display:flex; align-items:center; gap:10px; height:48px;}
  .brand b {font-size:18px; color:#0E1A2B; display:block; line-height:1.1}
  .brand small {font-size:12px; color:#556377}
  .card {background:#FFFFFF; border:1px solid #E3E9F2; border-radius:24px; box-sizing:border-box; color:#0E1A2B;}
  .now {padding:24px; display:flex; flex-direction:column; gap:14px; min-height:540px;}
  .now .place {display:flex; align-items:center; gap:8px}
  .now h1 {margin:0; padding:0; font-size:26px; font-weight:700; color:#0E1A2B}
  .now .sub, .feels {font-size:14px; color:#556377}
  .nowrow {display:flex; align-items:center; gap:18px}
  .nowtxt {display:flex; flex-direction:column}
  .big {font-size:84px; font-weight:300; line-height:.95; letter-spacing:-.03em; font-variant-numeric:tabular-nums}
  .cond {font-size:18px; font-weight:600; margin-top:6px}
  .chips {display:flex; gap:8px; flex-wrap:wrap}
  .chip {padding:6px 12px; border-radius:999px; font-size:14px; font-weight:600}
  .chip.hi {background:#FFF4E2; color:#7A4A06} .chip.lo {background:#E8F1FC; color:#164C8C}
  .tiles {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; margin-top:auto}
  .tile {display:flex; flex-direction:column; gap:2px; padding:10px 12px; background:#F6F8FB; border-radius:14px}
  .tl {font-size:12px; color:#556377; font-weight:600} .ts {font-size:12px; color:#556377}
  .tv {font-size:17px; font-weight:700; font-variant-numeric:tabular-nums}
  .st-key-fc_card {background:#FFFFFF; border:1px solid #E3E9F2; border-radius:24px; padding:20px 24px 16px;
                   min-height:540px; box-sizing:border-box; gap:.6rem;}
  .fc-title {font-size:18px; font-weight:700; color:#0E1A2B; padding-top:6px}
  .hscroll {overflow-x:auto; padding-bottom:6px}
  .hinner {display:flex; flex-direction:column; gap:2px}
  .hgrid {display:grid}
  .hc {display:flex; flex-direction:column; align-items:center; gap:6px; padding:10px 0; border-radius:14px}
  .hc.first {background:#E8F1FC}
  .ht {font-size:12px; color:#556377; font-weight:600; white-space:nowrap}
  .hv {font-size:16px; font-weight:700; color:#0E1A2B; font-variant-numeric:tabular-nums}
  .hr {font-size:11px; color:#1F63B5; font-weight:600; min-height:14px}
  .cap, .src {margin:0 !important; font-size:12px !important; color:#556377 !important; line-height:1.5}
  .src {padding-top:10px; border-top:1px solid #EDF1F6; margin-top:8px}
  .src a {color:#1F63B5}
  .week {display:flex; flex-direction:column}
  .wk {display:grid; grid-template-columns:84px 32px 128px 56px 30px minmax(60px,1fr) 32px; align-items:center;
       gap:12px; min-height:52px; border-top:1px solid #EDF1F6; color:#0E1A2B; font-variant-numeric:tabular-nums}
  .wk:first-child {border-top:none}
  .wd {display:flex; flex-direction:column; line-height:1.2} .wd b {font-size:15px} .wd small {font-size:12px; color:#556377}
  .wc {font-size:14px; color:#33435A; white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
  .wr {font-size:13px; color:#1F63B5; font-weight:600; display:inline-flex; align-items:center; gap:4px}
  .wl {font-size:15px; color:#556377; text-align:right} .wh {font-size:15px; font-weight:700}
  .bar {position:relative; height:6px; border-radius:3px; background:#EDF1F6}
  .bar span {position:absolute; top:0; height:6px; border-radius:3px; background:linear-gradient(90deg,#6FA8E8,#F0B04A)}
  @media (max-width: 640px) {
    .block-container {padding-left:12px; padding-right:12px}
    .now, .st-key-fc_card {min-height:0}
    .big {font-size:72px}
    .wk {grid-template-columns:58px 28px 0 48px 26px minmax(40px,1fr) 28px; gap:8px}
    .wc {display:none}
  }
</style>
"""
