#!/usr/bin/env python3
# ARI experiment control script

import argparse
import itertools
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

Benchmarks = Literal["job", "stats", "stack"]
ResultsDir = Path("/ari/results")


@dataclass
class LogEntry:
    label: str
    script: str
    event: Literal["start", "end"]
    timestamp: datetime
    args: str


def has_db(benchmark: Benchmarks) -> bool:
    psql_res = subprocess.run(
        [
            "psql",
            "-t",
            "-c",
            "select datname from pg_database where datistemplate is false",
        ],
        capture_output=True,
        text=True,
    )

    all_dbs = {db.strip() for db in psql_res.stdout.splitlines()}

    if benchmark == "job":
        return "job" in all_dbs or "imdb" in all_dbs
    return benchmark in all_dbs


def console(*args) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(timestamp, *args)


def log(entry: LogEntry, log_path: str | Path) -> None:
    log_path = Path(log_path)
    if not log_path.exists():
        skeleton = pd.DataFrame(
            columns=["label", "script", "event", "timestamp", "args"]
        )
        skeleton.to_csv(log_path, index=False)

    log_df = pd.DataFrame([entry])
    log_df.to_csv(log_path, mode="a", header=False, index=False)


def start_experiment(
    script: str,
    args: Optional[dict[str, str | list[str]]] = None,
    switches: Optional[list[str]] = None,
    *,
    label: str = "",
) -> None:
    args = args or {}
    switches = switches or []
    label = label or script.removesuffix(".py")
    absolute_path = Path("/ari/experiments") / script

    formatted_args: list[str] = []
    for arg, val in args.items():
        if isinstance(val, list):
            formatted_args.append(arg)
            formatted_args.extend(str(v) for v in val)
            continue

        formatted_args.append(arg)
        formatted_args.append(str(val))

    all_args = formatted_args + switches
    args_str = " ".join(all_args)

    log(
        LogEntry(
            label=label,
            script=script,
            event="start",
            timestamp=datetime.now(),
            args=args_str,
        ),
        "/ari/results/progress.log",
    )
    subprocess.run(["python3", absolute_path] + all_args)
    log(
        LogEntry(
            label=label,
            script=script,
            event="end",
            timestamp=datetime.now(),
            args=args_str,
        ),
        "/ari/results/progress.log",
    )


def experiment_native_runtimes(benchmark: Benchmarks) -> None:
    out_dir = ResultsDir / "base"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_experiment(
        "experiment-00-native-runtimes.py",
        {
            "--bench": benchmark,
            "--workloads-dir": "/ari/postbound/workloads",
            "--db-conn": f"/ari/.psycopg_connection_{benchmark}",
            "--out": out_dir / f"native-runtimes-{benchmark}.csv",
        },
        ["--prewarm"],
        label=f"native-runtimes-{benchmark}",
    )


def experiment_cardinality_distortion(benchmark: Benchmarks) -> None:
    out_dir = ResultsDir / "experiment-01-cardinality-distortion"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_experiment(
        "experiment-01-cardinality-distortion.py",
        {
            "--benchmark": benchmark,
            "--base-cards": "actual",
            "--cards-source": f"/ari/datasets/00-base/intermediate-cards-{benchmark}.csv",
            "--workloads-dir": "/ari/postbound/workloads",
            "--db-conn": f"/ari/.psycopg_connection_{benchmark}",
            "--out": out_dir / f"card-distortion-{benchmark}.csv",
        },
        [
            "--include-vanilla",
            "--include-default-underest",
            "--include-default-overest",
        ],
        label=f"card-distortion-{benchmark}",
    )


def distortion_ablation_settings(stage: int, enable_cout: bool) -> str:
    settings: list[str] = ["LOAD 'coutstar';"]
    settings.append(
        "SET enable_cout TO on;" if enable_cout else "SET enable_cout TO off;"
    )

    # minimal required operators
    settings.extend(["SET enable_seqscan TO on;", "SET enable_nestloop TO on;"])

    # all basic operators
    if stage >= 2:
        settings.extend(
            [
                "SET enable_indexscan TO on;",
                "SET enable_hashjoin TO on;",
                "SET enable_mergejoin TO on;",
                "SET enable_sort TO on;",
            ]
        )
    else:
        settings.extend(
            [
                "SET enable_indexscan TO off;",
                "SET enable_hashjoin TO off;",
                "SET enable_mergejoin TO off;",
                "SET enable_sort TO off;",
            ]
        )

    # optimized basic operators
    if stage >= 3:
        settings.extend(
            [
                "SET enable_indexonlyscan TO on;",
                "SET enable_bitmapscan TO on;",
                "SET enable_hashagg TO on;",
                "SET enable_incremental_sort TO on;",
            ]
        )
    else:
        settings.extend(
            [
                "SET enable_indexonlyscan TO off;",
                "SET enable_bitmapscan TO off;",
                "SET enable_hashagg TO off;",
                "SET enable_incremental_sort TO off;",
            ]
        )

    # intermediate operators
    if stage >= 4:
        settings.extend(["SET enable_material TO on;", "SET enable_memoize TO on;"])
    else:
        settings.extend(["SET enable_material TO off;", "SET enable_memoize TO off;"])

    # parallelism settings
    if stage >= 5:
        settings.extend(
            ["SET enable_gathermerge TO on;", "SET enable_parallel_hash TO on;"]
        )
    else:
        # for disabling parallelism we also need to set the max parallel workers!
        settings.extend(
            [
                "SET max_parallel_workers_per_gather TO 0;",
                "SET enable_gathermerge TO off;",
                "SET enable_parallel_hash TO off;",
            ]
        )

    return "\n".join(settings)


def experiment_distortion_ablation(benchmark: Benchmarks) -> None:
    base_out_dir = ResultsDir / "experiment-02-distortion-ablation"

    stages = range(1, 5 + 1)
    cout = [True, False]

    for stage, enable_cout in itertools.product(stages, cout):
        out_dir = base_out_dir / benchmark
        out_dir.mkdir(parents=True, exist_ok=True)
        cost_model_suffix = "cout" if enable_cout else "vanilla"
        out_file = (
            out_dir
            / f"{benchmark}-distortion-{cost_model_suffix}-cost-stage-{stage}.csv"
        )

        start_experiment(
            "experiment-01-cardinality-distortion.py",
            {
                "--benchmark": benchmark,
                "--base-cards": "actual",
                "--cards-source": f"/ari/datasets/00-base/intermediate-cards-{benchmark}.csv",
                "--workloads-dir": "/ari/postbound/workloads",
                "--db-conn": f"/ari/.psycopg_connection_{benchmark}",
                "--pg-conf": distortion_ablation_settings(stage, enable_cout),
                "--out": out_file,
            },
            [
                "--include-vanilla",
                "--include-default-underest",
                "--include-default-overest",
                "--explain-only",
            ],
            label=f"distortion-ablation-{benchmark}-stage-{stage}-{cost_model_suffix}",
        )


def experiment_plan_space_analysis(benchmark: Benchmarks) -> None:
    out_dir = ResultsDir / "experiment-03-plan-space-analysis" / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)

    start_experiment(
        "experiment-03-plan-space-analysis.py",
        {
            "--bench": benchmark,
            "--native-rts": ResultsDir / "base" / f"native-runtimes-{benchmark}.csv",
            "--cardinalities": f"/ari/datasets/00-base/intermediate-cards-{benchmark}.csv",
            "--workloads-dir": "/ari/postbound/workloads",
            "--db-conn": f"/ari/.psycopg_connection_{benchmark}",
            "--out-dir": out_dir,
        },
        [],
        label=f"plan-space-analysis-{benchmark}",
    )

    start_experiment(
        "experiment-03-plan-space-analysis.py",
        {
            "--bench": benchmark,
            "--native-rts": ResultsDir / "base" / f"native-runtimes-{benchmark}.csv",
            "--cardinalities": f"/ari/datasets/00-base/intermediate-cards-{benchmark}.csv",
            "--workloads-dir": "/ari/postbound/workloads",
            "--db-conn": f"/ari/.psycopg_connection_{benchmark}",
            "--out-dir": out_dir,
        },
        ["--fill-samples"],
        label=f"plan-space-analysis-{benchmark}-gaps",
    )

    resample_file = out_dir / f"{benchmark}-base-join-queries-f1.csv"
    subprocess.run(
        [
            "python3",
            "/ari/ari/base-join-analysis.py",
            "--workload",
            benchmark,
            "--out",
            resample_file,
        ],
    )

    start_experiment(
        "experiment-04-base-join-impact.py",
        {
            "--target-queries": resample_file,
            "--join-dir": out_dir,
            "--native-rts": ResultsDir / "base" / f"native-runtimes-{benchmark}.csv",
            "--out-dir": out_dir,
            "--workloads-dir": "/ari/postbound/workloads",
            "--workload": benchmark,
            "--cards": f"/ari/datasets/00-base/intermediate-cards-{benchmark}.csv",
            "--db-conn": f"/ari/.psycopg_connection_{benchmark}",
        },
        [],
        label=f"base-join-impact-{benchmark}",
    )


def experiment_analyze_stability(benchmark: Benchmarks) -> None:
    out_dir = ResultsDir / "experiment-05-analyze-stability" / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)

    start_experiment(
        "experiment-05-analyze-stability.py",
        {
            "--benchmark": benchmark,
            "--workloads-dir": "/ari/postbound/workloads",
            "--db-conn": f"/ari/.psycopg_connection_{benchmark}",
            "--out-dir": out_dir,
            "--repetitions": 10,
            "--suffix": "dynprog",
        },
        [],
        label=f"analyze-stability-{benchmark}-dynprog",
    )

    start_experiment(
        "experiment-05-analyze-stability.py",
        {
            "--benchmark": benchmark,
            "--workloads-dir": "/ari/postbound/workloads",
            "--db-conn": f"/ari/.psycopg_connection_{benchmark}",
            "--out-dir": out_dir,
            "--repetitions": 10,
            "--suffix": "geqo",
            "--min-tables": 12,
        },
        ["--with-geqo"],
        label=f"analyze-stability-{benchmark}-geqo",
    )


def reset_db(benchmark: Benchmarks) -> None:
    os.system(f"cd /ari/postbound/postgres && ./workload-{benchmark}-setup.sh --force")


def experiment_analyze_stability_shift(benchmark: Benchmarks) -> None:
    base_dir = ResultsDir / "experiment-06-analyze-stability-shift" / benchmark
    fill_factors = [0.05, 0.1, 0.25, 0.5, 0.75]

    for fill_factor in fill_factors:
        out_dir = base_dir / f"fill-factor{fill_factor}"
        out_dir.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            [
                "python3",
                "/ari/experiments/experiment-07-optimizer-architectures.py",
                "--benchmark",
                benchmark,
                "--workloads-dir",
                "/ari/postbound/workloads",
                "--db-conn",
                f"/ari/.psycopg_connection_{benchmark}",
                "--fill-factor",
                str(fill_factor),
                "--out-dir",
                out_dir,
                "shift only",
            ]
        )

        start_experiment(
            "experiment-05-analyze-stability.py",
            {
                "--benchmark": benchmark,
                "--workloads-dir": "/ari/postbound/workloads",
                "--db-conn": f"/ari/.psycopg_connection_{benchmark}",
                "--out-dir": out_dir,
                "--repetitions": 10,
                "--suffix": "dynprog",
            },
            ["--explain-only"],
            label=f"analyze-shift-{benchmark}-dynprog-ff-{fill_factor}",
        )

        start_experiment(
            "experiment-05-analyze-stability.py",
            {
                "--benchmark": benchmark,
                "--workloads-dir": "/ari/postbound/workloads",
                "--db-conn": f"/ari/.psycopg_connection_{benchmark}",
                "--out-dir": out_dir,
                "--repetitions": 10,
                "--suffix": "geqo",
                "--min-tables": 12,
            },
            ["--with-geqo", "--explain-only"],
            label=f"analyze-shift-{benchmark}-geqo-ff-{fill_factor}",
        )

    reset_db(benchmark)


def experiment_architecture_ablation(benchmark: Benchmarks) -> None:
    out_dir = ResultsDir / "experiment-07-beyond-textbook" / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)

    start_experiment(
        "experiment-07-optimizer-architectures.py",
        {
            "--benchmark": benchmark,
            "--experiments": ["native-fixed", "robust-fixed"],
            "--workloads-dir": "/ari/postbound/workloads",
            "--db-conn": f"/ari/.psycopg_connection_{benchmark}",
            "--disk-type": "SSD",
            "--out-dir": out_dir,
        },
        ["full"],
        label=f"beyond-textbook-{benchmark}",
    )

    reset_db(benchmark)


def run_experiments(
    exp: str, *, benchmarks: list[str], experiment_scripts: dict[str, Callable]
) -> None:
    executor = experiment_scripts[exp]
    for bench in benchmarks:
        if bench == "stack" and exp != "5-analyze-stability":
            continue

        console("Running experiment", exp, "for benchmark", bench)
        executor(bench)


def evaluate_results(notebook: str) -> None:
    notebook_path = Path("/ari/ari/eval") / f"{notebook}.ipynb"
    out_dir = ResultsDir / "eval" / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(["jupyter", "execute", "--inplace", notebook_path])
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "pdf",
            "--TagRemovePreprocessor.remove_cell_tags",
            '{"hide"}',  # don't change the quoting. Otherwise, nbconvert won't recognize the tag
            "--output-dir",
            out_dir,
            notebook_path,
        ]
    )


def run_evaluation(exp: str, *, eval_notebooks: dict[str, str]) -> None:
    notebook = eval_notebooks.get(exp)
    if not notebook:
        return

    console("Evaluating results for experiment", exp)
    evaluate_results(notebook)


def main() -> None:
    experiments = {
        "0-base": experiment_native_runtimes,
        "1-card-distortion": experiment_cardinality_distortion,
        "2-distortion-ablation": experiment_distortion_ablation,
        "3-plan-space": experiment_plan_space_analysis,
        "4-analyze-stability": experiment_analyze_stability,
        "5-analyze-shift": experiment_analyze_stability_shift,
        "6-beyond-textbook": experiment_architecture_ablation,
    }
    eval_notebooks = {
        "0-base": None,  # no core results for this experiment
        "1-card-distortion": "01-Card-Distortion",
        "2-distortion-ablation": "02-Distortion-Ablation",
        "3-plan-space": "03-Plan-Space",
        "4-analyze-stability": "04-Analyze-Stability",
        "5-analyze-shift": None,  # no core results for this experiment
        "6-beyond-textbook": "05-Beyond-Textbook",
    }
    benchmarks = ["job", "stats", "stack"]

    parser = argparse.ArgumentParser(
        description="Main control script for all experiments."
    )
    parser.add_argument(
        "--mode",
        type=str,
        action="store",
        choices=["full", "experiments", "eval"],
        default="full",
        help="Whether to only gather the experiment results (experiments), "
        "evaluate the results from a previous run (eval), "
        "or do both (full).",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        action="store",
        choices=["all"] + benchmarks,
        default="job",
        help="The benchmark/dataset to run the experiments on. "
        "Note that the Stack benchmark can only be used for experiment 5 and "
        "will be ignored for all others.",
    )
    parser.add_argument(
        "experiment",
        type=str,
        action="store",
        choices=["all"] + list(experiments.keys()),
        nargs="+",
        default=["all"],
        help="The experiments to execute. Can be any combination of experiments or all. "
        "Note that almost all experiments require the data from experiment 0.",
    )

    args = parser.parse_args()

    if args.benchmark == "all":
        selected_benchmarks = [bench for bench in benchmarks if has_db(bench)]
    elif not has_db(args.benchmark):
        raise ValueError(
            f"Benchmark {args.benchmark} has not been set-up during container initialization."
        )
    else:
        selected_benchmarks = [args.benchmark]
    console("Selected benchmarks:", selected_benchmarks)

    if "all" in args.experiment:
        selected_experiments = list(experiments.keys())
    else:
        selected_experiments = args.experiment
    console("Selected experiments:", selected_experiments)

    if args.mode == "full":
        tasks = ["experiments", "eval"]
    else:
        tasks = [args.mode]

    for exp in selected_experiments:
        if "experiments" in tasks:
            run_experiments(
                exp,
                benchmarks=selected_benchmarks,
                experiment_scripts=experiments,
            )

        if "eval" in tasks:
            run_evaluation(exp, eval_notebooks=eval_notebooks)


if __name__ == "__main__":
    main()
