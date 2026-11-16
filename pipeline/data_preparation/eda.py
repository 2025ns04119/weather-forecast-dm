"""
Exploratory data analysis (EDA) of the prepared data: distributions, correlations, seasonal and
daily patterns, rain sparsity, completeness, and feature-target relations. Saves plots + a summary.

Input : clean.weather_hourly (silver), clean.locations, features.weather_features (gold)
Output: reports/figures/01..07_*.png and reports/eda_summary.md

Usage (from the project root):
    python -m pipeline.data_preparation.eda
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import text

from pipeline.common.db import finish_run, get_engine, start_run
from pipeline.common.logger import get_logger
from pipeline.feature_engineering.build_features import FEATURE_COLS, FEATURE_VERSION

log = get_logger("data_preparation")
FIG = Path("reports/figures")
SUMMARY = Path("reports/eda_summary.md")
BLUE, ORANGE = "#2a78d6", "#eb6834"
VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "pressure_msl",
        "cloud_cover", "wind_speed_10m", "wind_direction_10m", "precipitation"]
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name: str) -> str:
    fig.tight_layout()
    path = FIG / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("figure saved: %s", path.as_posix())
    return path.as_posix()


def load(engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    with engine.connect() as conn:
        clean = pd.read_sql(text("SELECT * FROM clean.weather_hourly"), conn)
        locs = pd.read_sql(text("SELECT location, country, continent, climate_zone, latitude, role "
                                "FROM clean.locations"), conn)
        feats = pd.read_sql(text("SELECT * FROM features.weather_features WHERE feature_version = :v "
                                 "AND location = 'Bengaluru'"), conn, params={"v": FEATURE_VERSION})
    clean["ts"] = pd.to_datetime(clean["ts"], utc=True)
    df = clean.merge(locs, on="location", how="left")
    df["climate_zone"] = df["climate_zone"].fillna("Unknown")
    return df, feats


def overview(df: pd.DataFrame) -> list[str]:
    stats = df[VARS].describe().T.round(2)
    missing = df[VARS].isna().sum()
    hours = df.groupby("location").size()
    return ["## 1. Overview", "",
            f"- Rows: **{len(df):,}** hourly records, **{df['location'].nunique()}** locations",
            f"- Period: {df['ts'].min():%Y-%m-%d} to {df['ts'].max():%Y-%m-%d}",
            f"- Hours per location: min {hours.min():,}, max {hours.max():,}",
            f"- Missing values in the silver layer: {int(missing.sum()):,}", "",
            "| Variable | mean | std | min | max | missing |", "|---|---|---|---|---|---|",
            *[f"| {v} | {stats.loc[v, 'mean']} | {stats.loc[v, 'std']} | {stats.loc[v, 'min']} | "
              f"{stats.loc[v, 'max']} | {int(missing[v])} |" for v in VARS], ""]


def plot_distributions(df):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, c in zip(axes.flat, VARS):
        data = df[c].dropna()
        title = c
        if c == "precipitation":
            data, title = data[data > 0], "precipitation (> 0 mm, log scale)"
            ax.set_yscale("log")
        ax.hist(data, bins=50, color=BLUE)
        ax.set_title(title, loc="left", fontsize=10)
    return save(fig, "01_distributions")


def plot_correlation(df):
    corr = df[VARS].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(VARS)), VARS, rotation=45, ha="right")
    ax.set_yticks(range(len(VARS)), VARS)
    for i in range(len(VARS)):
        for j in range(len(VARS)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, shrink=0.8)
    ax.set_title("Correlation between weather variables", loc="left")
    path = save(fig, "02_correlation_heatmap")
    pairs = corr.where(~corr.isna()).stack()
    pairs = pairs[pairs.index.get_level_values(0) < pairs.index.get_level_values(1)]
    top = pairs.abs().sort_values(ascending=False).head(3)
    return path, [f"{a} vs {b}: {corr.loc[a, b]:.2f}" for a, b in top.index]


def plot_patterns(df):
    df = df.assign(month=df["ts"].dt.month)
    monthly = df.groupby(["climate_zone", "month"])["temperature_2m"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly.plot(ax=ax, marker="o")
    ax.set_title("Mean temperature by month and climate zone", loc="left")
    ax.set_ylabel("°C")
    ax.legend(frameon=False)
    p1 = save(fig, "03_monthly_by_climate_zone")

    blr = df[df["location"] == "Bengaluru"].copy()
    blr["local_hour"] = blr["ts"].dt.tz_convert("Asia/Kolkata").dt.hour
    daily = blr.groupby("local_hour")[["temperature_2m", "relative_humidity_2m"]].mean()
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(daily.index, daily["temperature_2m"], color=BLUE, marker="o", label="Temperature (°C)")
    ax2 = ax1.twinx()
    ax2.plot(daily.index, daily["relative_humidity_2m"], color=ORANGE, marker="o", label="Humidity (%)")
    ax1.set_xlabel("Local hour (IST)")
    ax1.set_title("Bengaluru - average daily cycle", loc="left")
    fig.legend(loc="upper left", frameon=False)
    p2 = save(fig, "04_bengaluru_daily_cycle")
    peak = int(daily["temperature_2m"].idxmax())
    return p1, p2, peak


def plot_rain(df):
    rain = df.assign(rainy=df["precipitation"] >= 0.1).groupby("location")["rainy"].mean().sort_values() * 100
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(rain.index, rain.values, color=BLUE)
    ax.set_xlabel("% of hours with rain (>= 0.1 mm)")
    ax.set_title("Rain is a sparse, imbalanced target", loc="left")
    return save(fig, "05_rain_sparsity"), float(rain.mean())


def plot_completeness(df):
    ym = df["ts"].dt.strftime("%Y-%m")
    have = df.assign(ym=ym).groupby(["location", "ym"]).size().unstack(fill_value=0)
    days = pd.to_datetime(have.columns + "-01").days_in_month.values
    completeness = (have / (days * 24) * 100).clip(upper=100)
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(completeness, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_yticks(range(len(completeness)), completeness.index)
    step = max(1, len(completeness.columns) // 12)
    ax.set_xticks(range(0, len(completeness.columns), step), completeness.columns[::step],
                  rotation=45, ha="right")
    fig.colorbar(im, label="% of hours present")
    ax.set_title("Hourly data completeness by city and month", loc="left")
    return save(fig, "06_completeness_heatmap")


def plot_feature_target(feats):
    corr = (feats[FEATURE_COLS + ["target_temp_24h"]].corr()["target_temp_24h"]
            .drop("target_temp_24h").sort_values())
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(corr.index, corr.values, color=[ORANGE if v < 0 else BLUE for v in corr.values])
    ax.set_title(f"Correlation with temperature 24 h later (Bengaluru, {FEATURE_VERSION})", loc="left")
    return save(fig, "07_feature_target_correlation"), corr.abs().sort_values(ascending=False).head(3)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    engine = get_engine()
    run_id = start_run(engine, "eda")
    try:
        df, feats = load(engine)
        log.info("EDA data loaded: %d rows, %d locations", len(df), df["location"].nunique())
        lines = ["# Exploratory Data Analysis - summary", ""] + overview(df)
        p1 = plot_distributions(df)
        p2, top_pairs = plot_correlation(df)
        p3, p4, peak = plot_patterns(df)
        p5, rainy = plot_rain(df)
        p6 = plot_completeness(df)
        p7, top_feats = plot_feature_target(feats)
        lines += ["## 2. Findings", "",
                  f"- **Distributions** ({p1}): most hours have no rain, so precipitation is skewed; "
                  "temperature is wide because the corpus covers every climate.",
                  f"- **Correlations** ({p2}): strongest pairs - " + "; ".join(top_pairs) + ".",
                  f"- **Seasonality** ({p3}): monthly curves differ by climate zone and are reversed in the "
                  "southern hemisphere - reason for the *month* and *latitude* features.",
                  f"- **Daily cycle** ({p4}): Bengaluru is warmest around {peak}:00 IST while humidity is "
                  "lowest - reason for the *solar hour* feature.",
                  f"- **Rain sparsity** ({p5}): on average only {rainy:.1f}% of hours are rainy, so rain is "
                  "modelled as a probability and evaluated with Brier score and AUC.",
                  f"- **Completeness** ({p6}): Bengaluru has 2 years of history, other cities 1 year; "
                  "light cells inside the range would mean gaps.",
                  f"- **Features vs target** ({p7}): most related to temperature 24 h later - "
                  + ", ".join(f"{k} ({v:.2f})" for k, v in top_feats.items()) + ".", "",
                  "## 3. Data preparation applied", "",
                  "| Step | What was done | Where |", "|---|---|---|",
                  "| Validation | Required fields, physical ranges, no future hours | "
                  "`pipeline/profiling_validation/clean_weather.py` |",
                  "| Duplicates | One row per location + hour | silver layer primary key |",
                  "| Missing values | Gaps up to 3 h interpolated; rows still missing a feature dropped | "
                  "`pipeline/feature_engineering/build_features.py` |",
                  "| Encoding | No categorical model inputs: location enters through latitude and solar hour; "
                  "climate zone is used only for analysis | - |",
                  "| Scaling | StandardScaler for linear/logistic models; tree models use raw values | "
                  "`pipeline/model_training/train.py` |",
                  "| Prepared dataset | Silver layer exported to `data/prepared/` and versioned with DVC | "
                  "`pipeline/versioning_lineage/export_datasets.py` |"]
        SUMMARY.write_text("\n".join(lines), encoding="utf-8")
        finish_run(engine, run_id, "success", rows_in=len(df), rows_out=7, notes="7 figures + eda_summary.md")
        log.info("EDA finished: 7 figures in reports/figures, summary in %s", SUMMARY.as_posix())
    except Exception as e:
        finish_run(engine, run_id, "failed", notes=str(e)[:500])
        log.exception("EDA FAILED: %s", e)
        raise


if __name__ == "__main__":
    main()
