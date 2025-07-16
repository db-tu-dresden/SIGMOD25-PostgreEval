#!/usr/bin/env python3
# ARI experiment control script

import argparse
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal


import pandas as pd

Benchmarks = Literal["job", "stats"]
ResultsDir = Path("/ari/results")


@dataclass
class LogEntry:
    script: str
    event: Literal["start", "end"]
    timestamp: datetime
    args: str


def has_db() -> bool:
    pass


def log(entry: LogEntry, log_path: str | Path) -> None:
    log_path = Path(log_path)
    if not log_path.exists():
        skeleton = pd.DataFrame(columns=["script", "event", "timestamp", "args"])
        skeleton.to_csv(log_path, index=False)

    log_df = pd.DataFrame([entry])
    log_df.to_csv(log_path, mode="a", header=False, index=False)


def start_experiment(script: str, args: dict[str, str], switches: list[str]) -> None:
    absolute_path = Path("/ari/experiments") / script
    formatted_args = [str(elem) for pair in args.items() for elem in pair]
    all_args = formatted_args + switches
    args_str = " ".join(all_args)

    log(LogEntry(script=script, event="start", timestamp=datetime.now(), args=args_str), "/ari/results/progress.log")
    subprocess.run(["python3", absolute_path] + all_args)
    log(LogEntry(script=script, event="end", timestamp=datetime.now(), args=args_str), "/ari/results/progress.log")


def experiment_native_runtimes(benchmark: Benchmarks) -> None:
    out_dir = ResultsDir / "base"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_experiment("experiment-00-native-runtimes.py", {
        "--bench": benchmark,
        "--db-conn": f"/ari/.psycopg_connection_{benchmark}",
        "--out": out_dir / f"native-runtimes-{benchmark}.csv",
    }, ["--prewarm"])


def experiment_cardinality_distortion(benchmark: Benchmarks) -> None:
    out_dir = ResultsDir / "experiment-01-cardinality-distortion"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_experiment("experiment-01-cardinality-distortion.py", {
        "--benchmark": benchmark,
        "--base-cards": "actual",
        "--cards-source": f"/ari/datasets/00-base/intermediate-cards-{benchmark}.csv",
        "--db-conn": f"/ari/.psycopg_connection_{benchmark}",
        "--out": out_dir / f"card-distortion-{benchmark}.csv",
    }, ["--include-vanilla", "--include-default-underest", "--include-default-overest"])


def experiment_plan_space_analysis(benchmark: Benchmarks) -> None:
    out_dir = ResultsDir / "experiment-03-plan-space-analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_experiment("experiment-03-plan-space-analysis.py", {
        "--bench": benchmark,
        "--native-rts": ResultsDir / "base" / f"native-runtimes-{benchmark}.csv",
        "--cardinalities": f"/ari/datasets/00-base/intermediate-cards-{benchmark}.csv",
        "--db-conn": f"/ari/.psycopg_connection_{benchmark}",
        "--out-dir": out_dir
    }, [])


def experiment_architecture_ablation(benchmark: Benchmarks) -> None:
    out_dir = ResultsDir / "experiment-04-beyond-textbook"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_experiment("experiment-07-optimizer-architectures.py", {
        "--benchmark": benchmark,
        "--experiments": "native-fixed robust-fixed",
        "--db-conn": f"/ari/.psycopg_connection_{benchmark}",
        "--out-dir": out_dir
    }, ["full"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Main control script for all experiments.")
    parser.add_argument("experiment", choices=["all"])

    args = parser.parse_args()

    if args.experiment == "all":
        experiment_native_runtimes("job")

    if args.experiment == "all":
        experiment_cardinality_distortion("job")

    if args.experiment == "all":
        experiment_plan_space_analysis("job")

    if args.experiment == "all":
        experiment_architecture_ablation("job")


if __name__ == "__main__":
    main()
