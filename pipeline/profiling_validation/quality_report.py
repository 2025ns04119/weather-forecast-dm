"""
Data quality report (profiling) for the silver layer and the quarantine tables.

Checks: schema, row counts per layer, missing values per column, duplicates,
range violations, completeness (missing hours) per city, freshness,
quarantine reasons, and a Great Expectations run on the last 7 days.

Output:
    reports/data_quality_report.pdf     (summary page + charts)
    reports/data_quality_summary.json   (all numbers, machine-readable)

Usage (from the project root):
    python -m pipeline.profiling_validation.quality_report
"""
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from pipeline.common.db import finish_run, get_engine, start_run
from pipeline.common.logger import get_logger
from pipeline.profiling_validation.clean_weather import COLUMNS, run_great_expectations
from pipeline.profiling_validation.quality_metrics import profile as sql_profile, q

log = get_logger("validation")
OUT_DIR = Path("reports")
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a8a"


def profile(engine) -> dict:
    """Data quality metrics (quality_metrics.profile) + Great Expectations on the last 7 days."""
    out = sql_profile(engine)
    # 8. Great Expectations on the last 7 days
    recent = q(engine, f"""SELECT location, ts, {', '.join(COLUMNS)} FROM clean.weather_hourly
                           WHERE ts >= now() - interval '7 days'""")
    out["great_expectations"] = run_great_expectations(recent) if len(recent) else {"evaluated": 0}
    out["great_expectations"]["rows_checked"] = len(recent)
    return out


def text_page(pdf, title: str, lines: list[str]):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.95, title, fontsize=16, weight="bold")
    y = 0.91
    for line in lines:
        for part in textwrap.wrap(line, 95) or [""]:
            fig.text(0.08, y, part, fontsize=9.5, family="monospace")
            y -= 0.018
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(info: dict, path: Path):
    comp = pd.DataFrame(info["completeness"])
    ge = info["great_expectations"]
    lines = [f"Generated: {info['generated_at_utc']} UTC", "",
             "1. Schema check (clean.weather_hourly)",
             f"   missing columns: {info['schema']['missing_columns'] or 'none'}",
             f"   unexpected columns: {info['schema']['unexpected_columns'] or 'none'}", "",
             "2. Rows per layer"]
    lines += [f"   {k:<30} {v:>12,}" for k, v in info["row_counts"].items()]
    lines += ["", "3. Missing values (silver)"]
    lines += [f"   {c:<24} {m['count']:>8,}  ({m['pct']}%)" for c, m in info["missing_values"].items()]
    lines += ["", f"4. Duplicate (location, hour) rows: {info['duplicates']}", "",
              "5. Range violations in silver (expected 0)"]
    lines += [f"   {c:<24} {n:>8}" for c, n in info["range_violations"].items()]
    lines += ["", "6. Completeness",
              f"   locations: {len(comp)}   overall completeness: "
              f"{100 * comp['hours'].sum() / max(comp['expected'].sum(), 1):.2f}%   "
              f"missing hours: {int(comp['missing_hours'].sum()):,}",
              f"   freshest location age: {comp['age_hours'].min()} h", "",
              "7. Quarantine reasons (weather)"]
    lines += [f"   {r['n']:>8,}  {r['reason']}" for r in info["quarantine_weather"]] or ["   none"]
    lines += ["   Quarantine (city catalogue CSV): "
              + (", ".join(f"{r['reason']} ({r['n']})" for r in info["quarantine_locations"]) or "none"),
              "", "8. Great Expectations (last 7 days)",
              f"   rows checked: {ge.get('rows_checked', 0):,}   expectations passed: "
              f"{ge.get('passed', 0)}/{ge.get('evaluated', 0)}   success: {ge.get('success')}"]

    with PdfPages(path) as pdf:
        text_page(pdf, "Data Quality Report - Weather Forecast Pipeline", lines)

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        c = comp.sort_values("completeness_pct")
        ax.barh(c["location"], c["completeness_pct"], color=BLUE)
        ax.set_xlim(max(0, c["completeness_pct"].min() - 1), 100.2)
        ax.set_xlabel("Completeness (% of expected hourly rows)")
        ax.set_title("Completeness per location (silver layer)", loc="left", weight="bold")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        mv = pd.Series({k: v["count"] for k, v in info["missing_values"].items()})
        axes[0].barh(mv.index, mv.values, color=ORANGE)
        axes[0].set_title("Missing values per column", loc="left", weight="bold")
        qw = pd.DataFrame(info["quarantine_weather"])
        if not qw.empty:
            axes[1].barh([textwrap.shorten(r, 40) for r in qw["reason"]], qw["n"], color=GREY)
        axes[1].set_title("Quarantined rows by reason", loc="left", weight="bold")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main():
    engine = get_engine()
    run_id = start_run(engine, "quality_report")
    try:
        info = profile(engine)
        OUT_DIR.mkdir(exist_ok=True)
        (OUT_DIR / "data_quality_summary.json").write_text(json.dumps(info, indent=2, default=str))
        write_pdf(info, OUT_DIR / "data_quality_report.pdf")
        ge = info["great_expectations"]
        finish_run(engine, run_id, "success", rows_in=info["row_counts"]["clean.weather_hourly"],
                   notes=f"GX {ge.get('passed')}/{ge.get('evaluated')}; duplicates={info['duplicates']}")
        log.info("data quality report written: reports/data_quality_report.pdf (GX %s/%s passed)",
                 ge.get("passed"), ge.get("evaluated"))
    except Exception as e:
        finish_run(engine, run_id, "failed", notes=str(e)[:500])
        log.exception("data quality report FAILED: %s", e)
        raise


if __name__ == "__main__":
    main()
