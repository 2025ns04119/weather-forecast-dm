"""
Model performance report from MLflow (DagsHub).

Reads the latest training runs (one per target) and the champion run, and writes:
    reports/model_performance.pdf   tables + charts (model vs baseline by forecast day)
    reports/model_performance.csv   all numbers

Usage (from the project root):
    python -m training.performance_report
"""
import os
import warnings
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from dotenv import load_dotenv
from matplotlib.backends.backend_pdf import PdfPages
from mlflow.tracking import MlflowClient

load_dotenv()
EXPERIMENT = "weather-forecast"
TARGETS = {"temp": "Temperature (°C)", "humidity": "Humidity (%)", "wind": "Wind speed (km/h)",
           "cloud": "Cloud cover (%)", "rain": "Rain yes/no"}
DAYS = range(1, 8)
OUT_DIR = Path("reports")
BLUE, ORANGE = "#2a78d6", "#eb6834"


def latest_run(client, exp_id, name: str):
    runs = client.search_runs([exp_id], f"attributes.run_name = '{name}'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    return runs[0] if runs else None


def collect() -> tuple[pd.DataFrame, dict]:
    client = MlflowClient()
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise RuntimeError(f"MLflow experiment '{EXPERIMENT}' not found - run training first")
    rows = []
    for target in TARGETS:
        run = latest_run(client, exp.experiment_id, f"target_{target}")
        if run is None:
            continue
        m = run.data.metrics
        for d in DAYS:
            row = {"target": target, "day": d, "run_id": run.info.run_id}
            for k in ("mae", "rmse", "r2", "brier", "auc"):
                if f"best_{k}_day{d}" in m:
                    row[f"model_{k}"] = m[f"best_{k}_day{d}"]
                if f"baseline_{k}_day{d}" in m:
                    row[f"baseline_{k}"] = m[f"baseline_{k}_day{d}"]
            rows.append(row)
    champ = latest_run(client, exp.experiment_id, "global_champion")
    meta = {}
    if champ:
        p = champ.data.params
        meta = {"run_id": champ.info.run_id, "data_md5": p.get("data_md5"),
                "feature_version": p.get("feature_version"), "locations": p.get("locations"),
                "test_from": p.get("test_from"), "rows_total": p.get("rows_total"),
                "model_size_mb": round(champ.data.metrics.get("model_size_mb", float("nan")), 1),
                "models": {t: p.get(f"model_{t}_day1") for t in TARGETS}}
    return pd.DataFrame(rows), meta


def write_pdf(df: pd.DataFrame, meta: dict, path: Path):
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.05, 0.94, "Model Performance Report - weather-7day", fontsize=16, weight="bold")
        info = [f"Champion run: {meta.get('run_id')}",
                f"Feature version: {meta.get('feature_version')}   Locations: {meta.get('locations')}   "
                f"Rows: {meta.get('rows_total')}   Test period from: {meta.get('test_from')}",
                f"Training data MD5 (DVC): {meta.get('data_md5')}   Model size: {meta.get('model_size_mb')} MB",
                "Baselines: persistence (value now) for regression targets; climatology (training rain rate) for rain"]
        for i, line in enumerate(info):
            fig.text(0.05, 0.88 - i * 0.035, line, fontsize=10)
        reg = df[df["target"] != "rain"].copy()
        tab = reg.pivot(index="day", columns="target", values="model_mae").round(2)
        base = reg.pivot(index="day", columns="target", values="baseline_mae").round(2)
        cells = [[f"{tab.loc[d, t]} / {base.loc[d, t]}" for t in tab.columns] for d in tab.index]
        ax = fig.add_axes([0.05, 0.40, 0.9, 0.3]); ax.axis("off")
        ax.set_title("MAE by forecast day - model / persistence baseline (lower is better)", loc="left")
        ax.table(cellText=cells, rowLabels=[f"Day {d}" for d in tab.index],
                 colLabels=[TARGETS[t] for t in tab.columns], loc="center").scale(1, 1.4)
        rain = df[df["target"] == "rain"]
        if not rain.empty:
            ax2 = fig.add_axes([0.05, 0.08, 0.9, 0.25]); ax2.axis("off")
            ax2.set_title("Rain (yes/no): Brier score (lower is better) and AUC (higher is better)", loc="left")
            ax2.table(cellText=[[f"{r.model_brier:.3f}", f"{r.baseline_brier:.3f}", f"{r.model_auc:.2f}"]
                                for r in rain.itertuples()],
                      rowLabels=[f"Day {d}" for d in rain["day"]],
                      colLabels=["Model Brier", "Climatology Brier", "Model AUC"], loc="center").scale(1, 1.2)
        pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        for ax, t in zip(axes.flat, ["temp", "humidity", "wind", "cloud"]):
            s = df[df["target"] == t]
            ax.plot(s["day"], s["model_mae"], marker="o", color=BLUE, label="Model")
            ax.plot(s["day"], s["baseline_mae"], marker="o", color=ORANGE, label="Persistence baseline")
            ax.set_title(f"{TARGETS[t]} - MAE by forecast day", loc="left")
            ax.set_xlabel("Forecast day"); ax.set_xticks(list(DAYS)); ax.legend(frameon=False)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def main():
    df, meta = collect()
    if df.empty:
        raise SystemExit("No training runs found in MLflow - run `python -m training.train` first")
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "model_performance.csv", index=False)
    write_pdf(df, meta, OUT_DIR / "model_performance.pdf")
    t = df[df["target"] == "temp"][["day", "model_mae", "baseline_mae"]].round(2)
    print("Temperature MAE (°C) by forecast day\n" + t.to_string(index=False))
    print("\nWritten: reports/model_performance.pdf and reports/model_performance.csv")


if __name__ == "__main__":
    main()
