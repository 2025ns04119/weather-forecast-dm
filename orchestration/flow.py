"""
Pipeline orchestration with Prefect.

DAG (each box is a Prefect task; a task starts only when the previous one succeeded):

    ingest_csv -> ingest_api -> validate -> build_features -> feature_store -> [train] -> predict
                                   \\-> [quality_report]

    ingestion tasks retry twice (60 s apart) - the API sometimes times out
    every task's output goes to the Prefect log; every step also writes logs/<step>.log
    and a row in ml.pipeline_runs

Usage (from the project root):
    python -m orchestration.flow                    # hourly run: data + forecast
    python -m orchestration.flow --train            # also retrain the model
    python -m orchestration.flow --report           # also build the data quality report
    python -m orchestration.flow --full --train     # first run: history for all cities + training
    python -m orchestration.flow --serve            # keep running, scheduled every hour (Prefect UI)

Prefect UI (optional, for screenshots):  prefect server start   ->  http://127.0.0.1:4200
"""
import argparse
import subprocess
import sys

from prefect import flow, get_run_logger, task


def run_module(module: str, *args: str) -> str:
    """Run one pipeline step (`python -m module args`), stream its output to the Prefect log."""
    logger = get_run_logger()
    cmd = [sys.executable, "-m", module, *args]
    logger.info("running: %s", " ".join(cmd[1:]))
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in (proc.stdout + proc.stderr).splitlines():
        if line.strip():
            logger.info(line)
    if proc.returncode != 0:
        raise RuntimeError(f"{module} failed with exit code {proc.returncode}")
    return proc.stdout


@task(name="1 ingest city catalogue (CSV)", retries=2, retry_delay_seconds=60)
def ingest_csv():
    run_module("ingestion.ingest_csv")


@task(name="2 ingest weather API (bronze)", retries=2, retry_delay_seconds=60)
def ingest_api(full: bool, days: int):
    if full:
        run_module("ingestion.fetch_weather", "--mode", "corpus", "--days", str(days))
    run_module("ingestion.fetch_weather", "--mode", "live")


@task(name="3 validate and clean (silver)")
def validate():
    run_module("validation.clean_weather")


@task(name="3b data quality report")
def quality_report():
    run_module("validation.quality_report")


@task(name="4 build features (gold)")
def build_features(full: bool):
    run_module("features.build_features", *(["--all-locations"] if full else []))


@task(name="5 feature store (registry + checks)")
def feature_store():
    run_module("feature_store.store", "--sync", "--check")


@task(name="6 train and register model")
def train():
    run_module("training.train")


@task(name="7 7-day forecast")
def predict():
    run_module("inference.predict")


@flow(name="weather-data-pipeline", log_prints=True)
def weather_pipeline(train_model: bool = False, report: bool = False,
                     full: bool = False, days: int = 365):
    """End-to-end pipeline: ingestion -> validation -> features -> feature store -> training -> inference."""
    ingest_csv()
    ingest_api(full, days)
    validate()
    if report:
        quality_report()
    build_features(full)
    feature_store()
    if train_model:
        train()
    predict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="retrain the model in this run")
    parser.add_argument("--report", action="store_true", help="build the data quality report")
    parser.add_argument("--full", action="store_true", help="load history for all catalogue cities")
    parser.add_argument("--days", type=int, default=365, help="history length for --full")
    parser.add_argument("--serve", action="store_true", help="run on an hourly schedule (Prefect)")
    args = parser.parse_args()

    if args.serve:
        weather_pipeline.serve(name="hourly", cron="17 * * * *",
                               parameters={"train_model": False, "report": False})
    else:
        weather_pipeline(train_model=args.train, report=args.report, full=args.full, days=args.days)


if __name__ == "__main__":
    main()
