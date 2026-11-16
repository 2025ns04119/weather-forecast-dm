"""
Model performance report from MLflow (DagsHub).

Reads the latest training runs (one per target) and the champion run, and writes:
    reports/model_performance.pdf   tables + charts (model vs baseline by forecast day)
    reports/model_performance.csv   all numbers

Usage (from the project root):
    python -m pipeline.model_evaluation.performance_report
"""
import os
import warnings
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from pipeline.model_evaluation.results import DAYS, TARGETS, collect

OUT_DIR = Path("reports")
BLUE, ORANGE = "#2a78d6", "#eb6834"


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
            ax2.set_title("Rain (yes/no): Brier (lower is better); AUC, precision, recall, F1 (higher is better)",
                          loc="left")
            get = lambda r, k: f"{getattr(r, k):.2f}" if hasattr(r, k) and getattr(r, k) == getattr(r, k) else "-"
            ax2.table(cellText=[[f"{r.model_brier:.3f}", f"{r.baseline_brier:.3f}", get(r, "model_auc"),
                                 get(r, "model_precision"), get(r, "model_recall"), get(r, "model_f1")]
                                for r in rain.itertuples()],
                      rowLabels=[f"Day {d}" for d in rain["day"]],
                      colLabels=["Model Brier", "Climatology Brier", "AUC", "Precision", "Recall", "F1"],
                      loc="center").scale(1, 1.2)
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
        raise SystemExit("No training runs found in MLflow - run `python -m pipeline.model_training.train` first")
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "model_performance.csv", index=False)
    write_pdf(df, meta, OUT_DIR / "model_performance.pdf")
    t = df[df["target"] == "temp"][["day", "model_mae", "baseline_mae"]].round(2)
    print("Temperature MAE (°C) by forecast day\n" + t.to_string(index=False))
    print("\nWritten: reports/model_performance.pdf and reports/model_performance.csv")


if __name__ == "__main__":
    main()
