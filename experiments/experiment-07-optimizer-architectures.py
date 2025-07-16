from __future__ import annotations

import argparse
import json
import pathlib
import os
import textwrap
import typing
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

import pandas as pd

from postbound import postbound as pb
from postbound.qal import base, qal, transform
from postbound.db import postgres
from postbound.experiments import workloads
from postbound.optimizer import jointree, presets
from postbound.optimizer.policies import cardinalities as cards
from postbound.util import jsonize, logging, proc

OutDir = "robustness-shift"
BaselineFilling = 0.6
ShiftStep = 0.05
ShiftSpan = 0.4
QueryTimeout = 60 * 5  # 5 minutes max timeout --> n seconds to minutes * n minutes
ExperimentType = Literal["native", "robust", "native-true-cards", "native-fixed", "robust-fixed"]
EnabledExperiments = ["native", "native-fixed", "robust-fixed"]
NoGeQO = postgres.PostgresSetting("geqo", "off")

log = logging.make_logger(prefix=lambda: f"{logging.timestamp()} ::")


@dataclass
class ExperimentConfig:
    workload: workloads.Workload
    setup_db_script: str

    pg_instance: postgres.PostgresInterface
    pg_settings: postgres.PostgresConfiguration
    pg_commands: list[str]

    fact_table: base.TableReference
    fact_marker_column: base.ColumnReference
    marker_table: base.TableReference
    marker_column: base.ColumnReference
    delete_marker_file: pathlib.Path

    enabled_experiments: list[ExperimentType]
    out_dir: str


def update_database(conf: ExperimentConfig) -> None:
    pg_instance = conf.pg_instance
    db_name = pg_instance.database_name()
    data_dir = pg_instance.execute_query("SHOW data_directory;", cache_enabled=False)
    logfile = (pathlib.Path(".") / conf.out_dir / "pg.log").resolve()
    pg_instance.close()

    proc.run_cmd([f"./{conf.setup_db_script}", "--force"], work_dir=os.environ["PG_CTL_PATH"]).raise_if_error()

    (proc
     .run_cmd(["./postgres-config-generator.py", "--out", "pg-conf.sql", data_dir], work_dir=os.environ["PG_CTL_PATH"])
     .raise_if_error())

    proc.run_cmd(["psql", db_name, "-f", "pg-conf.sql"], work_dir=os.environ["PG_CTL_PATH"]).raise_if_error()

    proc.run_cmd(["pg_ctl", "-l", logfile, "restart"], work_dir=os.environ["PG_CTL_PATH"]).raise_if_error()

    pg_instance.reset_connection()


def obtain_baseline_plans(conf: ExperimentConfig) -> None:
    outfile = conf.out_dir + "/baseline.json"
    pg_instance = conf.pg_instance
    conn = pg_instance.obtain_new_local_connection()
    conn.autocommit = False
    cursor = conn.cursor()
    workload_shifter = postgres.WorkloadShifter(pg_instance)

    for cmd in conf.pg_commands:
        pg_instance.apply_configuration(cmd)

    ues_optimizer = pb.TwoStageOptimizationPipeline(pg_instance)
    ues_optimizer = ues_optimizer.load_settings(presets.fetch("ues")).build()

    log("Building delete sample")
    workload_shifter.generate_marker_table(conf.fact_table.full_name, 1 - BaselineFilling + ShiftSpan)
    workload_shifter.export_marker_table(target_table=conf.fact_table.full_name, out_file=conf.delete_marker_file)
    cursor.execute(f"CREATE TABLE delete_marker_buffer (LIKE {conf.marker_table.full_name});")

    total_marked_tuples = pg_instance.statistics().total_rows(conf.marker_table)
    max_marker_idx = round(0.5 * total_marked_tuples)
    baseline_marker_query = textwrap.dedent(f"""
                                            INSERT INTO delete_marker_buffer (marker_idx, {conf.marker_column.name})
                                            SELECT * FROM {conf.marker_table.full_name}
                                            WHERE marker_idx <= {max_marker_idx};""")
    cursor.execute(baseline_marker_query)
    conn.commit()
    pg_instance.reset_connection()

    log("Creating baseline data shift")
    workload_shifter.remove_marked(conf.fact_table.full_name, marker_table="delete_marker_buffer", vacuum=True)

    native_plans: dict[str, jointree.PhysicalQueryPlan] = {}
    ues_plans: dict[str, jointree.PhysicalQueryPlan] = {}
    cursor.execute(NoGeQO)
    for label, query in conf.workload.entries():
        log("Obtaining native plan for query", label)
        native_plan = pg_instance.optimizer().query_plan(query)
        native_plans[label] = jointree.PhysicalQueryPlan.load_from_query_plan(native_plan, query)

    pg_instance.statistics().cache_enabled = True
    for label, query in conf.workload.entries():
        # we obtain native and robust plans in two separate loops to ensure that the native plans are not influenced by any
        # settings that are set for the robust plans
        log("Obtaining UES plan for query", label)
        ues_plan = ues_optimizer.query_execution_plan(query)
        ues_plans[label] = ues_plan
    pg_instance.statistics().cache_enabled = False
    pg_instance.reset_cache()

    baseline = {"native_plans": native_plans, "robust_plans": ues_plans}
    with open(outfile, "w") as out:
        out.write(jsonize.to_json(baseline))


@dataclass
class DataShiftResult:
    fill_ratio: float
    plan_type: ExperimentType
    label: str
    query: str
    query_plan: str
    total_runtime: float
    timeout: bool
    db_config: str


def obtain_data_shift_result(fill_ratio: float, label: str, query: qal.SqlQuery, plan_type: ExperimentType, *,
                             conf: ExperimentConfig,
                             query_plans: Optional[dict] = None,
                             cardinality_estimator: Optional[cards.CardinalityHintsGenerator] = None) -> DataShiftResult:
    pg_instance = conf.pg_instance
    timeout_executor = postgres.TimeoutQueryExecutor(pg_instance)
    ues_optimizer = pb.TwoStageOptimizationPipeline(pg_instance)
    ues_optimizer = ues_optimizer.load_settings(presets.fetch("ues")).build()

    for cmd in conf.pg_commands:
        pg_instance.apply_configuration(cmd)

    pretty_fill_ratio = round(fill_ratio, 2)
    log("Executing", plan_type, "query", label, "at fill factor", pretty_fill_ratio)
    if plan_type == "native":
        explain_query = query
    elif plan_type == "robust":
        pg_instance.statistics().cache_enabled = True
        explain_query = ues_optimizer.optimize_query(query)
        pg_instance.statistics().cache_enabled = False
    elif plan_type == "robust-fixed":
        ues_plan = jointree.read_from_json(query_plans["robust_plans"][label], include_cardinalities=False)
        explain_query = pg_instance.hinting().generate_hints(query, ues_plan)
    elif plan_type == "native-fixed":
        native_plan = jointree.read_from_json(query_plans["native_plans"][label], include_cardinalities=False)
        explain_query = pg_instance.hinting().generate_hints(query, native_plan)
    elif plan_type == "native-true-cards":
        cardinality_hints = cardinality_estimator.estimate_cardinalities(query)
        explain_query = pg_instance.hinting().generate_hints(query, plan_parameters=cardinality_hints)
    else:
        raise ValueError("Unknown experiment type: {}".format(plan_type))

    explain_query = transform.as_explain_analyze(explain_query)
    try:
        start_time = datetime.now()
        explain_plan = timeout_executor.execute_query(explain_query, QueryTimeout)
        end_time = datetime.now()
        total_runtime = (end_time - start_time).total_seconds()
        timeout = False
    except TimeoutError:
        explain_plan = pg_instance.execute_query(transform.as_explain(explain_query))
        total_runtime = QueryTimeout
        timeout = True

    pg_instance.apply_configuration(conf.pg_settings)

    # this should already be included in the previous command, but better safe than sorry
    pg_instance.apply_configuration(NoGeQO)

    # we need to temporarily disable the cache to ensure that the schema info (especially relation sizes) for our target
    # database is actually inferred on the current data shift
    stats_cache = pg_instance.statistics().cache_enabled
    pg_instance.statistics().cache_enabled = False
    db_info = jsonize.to_json(pg_instance.describe())
    pg_instance.statistics().cache_enabled = stats_cache

    result = DataShiftResult(fill_ratio=fill_ratio, plan_type=plan_type, label=label,
                             query=str(query), query_plan=jsonize.to_json(explain_plan),
                             total_runtime=total_runtime, timeout=timeout, db_config=db_info)
    return result


def simulate_data_shift(conf: ExperimentConfig) -> None:
    baseline_file = conf.out_dir + "/baseline.json"
    outfile = conf.out_dir + "/data-shift.csv"
    pg_instance = conf.pg_instance
    conn = pg_instance.obtain_new_local_connection()
    conn.autocommit = False
    cursor = conn.cursor()
    workload_shifter = postgres.WorkloadShifter(pg_instance)

    log("Importing marker table")
    total_n_tuples = pg_instance.statistics().total_rows(conf.fact_table)
    tuples_to_drop: int = round(ShiftStep * total_n_tuples)
    workload_shifter.import_marker_table(target_table=conf.fact_table.full_name, in_file=conf.delete_marker_file)
    cursor.execute(f"CREATE TABLE IF NOT EXISTS delete_marker_buffer (LIKE {conf.fact_table.full_name}_delete_marker);")
    conn.commit()
    pg_instance.reset_connection()

    cardinality_estimator = cards.PreciseCardinalityHintGenerator(pg_instance, enable_cache=True)
    with open(baseline_file, "r") as baselines:
        query_plans: dict = json.load(baselines)

    results: list[DataShiftResult] = []
    start_marker_idx, end_marker_idx = 0, tuples_to_drop
    data_step = BaselineFilling + ShiftSpan
    while round(data_step, ndigits=2) >= round(BaselineFilling - ShiftSpan, ndigits=2):
        pretty_data_step = round(data_step, 2)
        log("Now at data shift pct", pretty_data_step)

        pg_instance.apply_configuration(NoGeQO)
        pg_instance.statistics().cache_enabled = True
        pg_instance.reset_cache()
        for label, query in conf.workload.entries():
            log("Evaluating query", label, "at data shift pct", pretty_data_step)
            for experiment_type in conf.enabled_experiments:
                pg_instance.prewarm_tables(query.tables())
                experiment_result = obtain_data_shift_result(data_step, label, query, experiment_type,
                                                             query_plans=query_plans,
                                                             cardinality_estimator=cardinality_estimator,
                                                             conf=conf)
                results.append(experiment_result)

        cardinality_estimator.reset_cache()
        pg_instance.statistics().cache_enabled = False
        pg_instance.reset_cache()

        log("Performing data shift at", pretty_data_step, "pct for indexes between", start_marker_idx, "and", end_marker_idx)
        cursor.execute("DELETE FROM delete_marker_buffer;")
        marker_inflation_query = textwrap.dedent(f"""
                                                 INSERT INTO delete_marker_buffer (marker_idx, {conf.marker_column.name})
                                                 SELECT * FROM {conf.marker_table.full_name}
                                                 WHERE marker_idx BETWEEN {start_marker_idx} AND {end_marker_idx};""")
        cursor.execute(marker_inflation_query)
        conn.commit()
        pg_instance.reset_connection()
        workload_shifter.remove_marked(conf.fact_table.full_name, marker_table="delete_marker_buffer", vacuum=True)
        start_marker_idx = end_marker_idx
        end_marker_idx += tuples_to_drop
        data_step -= ShiftStep

    result_df = pd.DataFrame(results)
    result_df.to_csv(outfile, index=False)


def benchmark_current_db(conf: ExperimentConfig, *, fill_factor: float) -> None:
    baseline_file = conf.out_dir + "/baseline.json"
    pretty_fill_factor = round(fill_factor, 2)
    outfile = conf.out_dir + f"/data-shift-{pretty_fill_factor}.csv"
    pg_instance = conf.pg_instance

    log("Evaluating for current database only")
    cardinality_estimator = cards.PreciseCardinalityHintGenerator(pg_instance, enable_cache=True)
    with open(baseline_file, "r") as baselines:
        query_plans: dict = json.load(baselines)

    pg_instance.apply_configuration(NoGeQO)
    pg_instance.statistics().cache_enabled = True
    pg_instance.reset_cache()

    results: list[DataShiftResult] = []
    for label, query in conf.workload.entries():
        log("Evaluating query", label, "at data shift pct", pretty_fill_factor)
        for experiment_type in conf.enabled_experiments:
            pg_instance.prewarm_tables(query.tables())
            experiment_result = obtain_data_shift_result(fill_factor, label, query, experiment_type, conf=conf,
                                                         query_plans=query_plans, cardinality_estimator=cardinality_estimator)
            results.append(experiment_result)

    pg_instance.reset_cache()
    result_df = pd.DataFrame(results)
    result_df.to_csv(outfile, index=False)


def manual_shift(conf: ExperimentConfig, *, fill_factor: float) -> None:
    pg_instance = conf.pg_instance
    conn = pg_instance.obtain_new_local_connection()
    conn.autocommit = False
    cursor = conn.cursor()
    workload_shifter = postgres.WorkloadShifter(pg_instance)
    total_n_tuples = pg_instance.statistics().total_rows(conf.fact_table)

    log("Building delete sample")
    workload_shifter.generate_marker_table(conf.fact_table.full_name, 1)
    workload_shifter.export_marker_table(target_table=conf.fact_table.full_name, out_file=conf.delete_marker_file)
    cursor.execute(f"CREATE TABLE IF NOT EXISTS delete_marker_buffer (LIKE {conf.fact_table.full_name}_delete_marker);")
    conn.commit()
    pg_instance.reset_connection()

    end_marker_idx = round((1 - fill_factor) * total_n_tuples)
    marker_query = textwrap.dedent(f"""
                                   INSERT INTO delete_marker_buffer (marker_idx, {conf.marker_column.name})
                                   SELECT * FROM {conf.marker_table.full_name}
                                   WHERE marker_idx <= {end_marker_idx};""")
    cursor.execute(marker_query)
    conn.commit()
    pg_instance.reset_connection()

    log("Removing marked tuples")
    workload_shifter.remove_marked(conf.fact_table.full_name, marker_table="delete_marker_buffer", vacuum=True)


def configure_pg(conf: ExperimentConfig) -> None:
    pg_instance = conf.pg_instance
    pg_instance.cache_enabled = False
    pg_stats = pg_instance.statistics()
    pg_stats.cache_enabled = False
    pg_stats.emulated = True


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", default="full", choices=["baseline", "full", "shift", "single", "shift-only"],
                        help="What to do: 'full' resets database, generates baseline and benchmarks the entire data shift. "
                        "'baseline' resets the database and generates the baseline, but does not perform the data shift. "
                        "'shift' assumes the baseline is already present and benchmarks the entire data shift. "
                        "'single' executes the benchmark on the current fill factor but does not perform any data shift. "
                        "'shift-only' resets the database and performs the data shift to the desired fill factor without "
                        "benchmarking.")
    parser.add_argument("--benchmark", "-b", default="job", choices=["job", "stats"], help="The benchmark to use")
    parser.add_argument("--experiments", nargs="+", default=EnabledExperiments, choices=typing.get_args(ExperimentType),
                        help="The query plans to benchmark: 'native' runs the vanilla query plan on the current data shift. "
                        "'robust' runs UES on the current data shift. 'native-true-cards' runs the vanilla optimizer with "
                        "perfect cardinalities. This takes a loooong time to compute. 'native-fixed' and 'robust-fixed' run "
                        "the query plans obtained at the baseline with their respective optimization strategies.")
    parser.add_argument("--fill-factor", type=float, default=BaselineFilling, help="The specific fill factor to use. This is "
                        "required for the 'single' to correctly identify the current DB state and for the 'shift-only' mode "
                        "to identify the target fill factor.")
    parser.add_argument("--workloads-dir", action="store", help="The directory where the workloads are stored.")
    parser.add_argument("--db-conn", "-c", action="store", help="The path to the database connection file.")
    parser.add_argument("--pg-cmds", nargs="+", default=[], help="Specific Postgres commands to execute before the benchmark. "
                        "*Each command has to be idempotent.*")
    parser.add_argument("--out-dir", "-o", default=OutDir, help="Directory to store the results in (delete markers, baseline "
                        "and results). Will be created if necessary.")

    args = parser.parse_args()

    if args.workloads_dir:
        workloads.workloads_base_dir = args.workloads_dir

    if args.db_conn:
        pg_conf = args.db_conn
    else:
        pg_conf = ".psycopg_connection_job" if args.benchmark == "job" else ".psycopg_connection_stats"
    pg_instance = postgres.connect(config_file=pg_conf, cache_enabled=False)

    for cmd in args.pg_cmds:
        pg_instance.apply_configuration(cmd)
    pg_instance.apply_configuration(NoGeQO)
    pg_settings = pg_instance.current_configuration(runtime_changeable_only=True)

    fact_table = base.TableReference("title") if args.benchmark == "job" else base.TableReference("posts")
    fact_marker_column = base.ColumnReference("id", fact_table)
    marker_table = base.TableReference(f"{fact_table.full_name}_delete_marker")
    marker_column = base.ColumnReference(f"{fact_table.full_name}_{fact_marker_column.name}", marker_table)

    conf = ExperimentConfig(
        workload=workloads.job() if args.benchmark == "job" else workloads.stats(),
        setup_db_script="workload-job-setup.sh" if args.benchmark == "job" else "workload-stats-setup.sh",
        pg_instance=pg_instance,
        pg_settings=pg_settings,
        pg_commands=args.pg_cmds,
        fact_table=fact_table,
        fact_marker_column=fact_marker_column,
        marker_table=marker_table,
        marker_column=marker_column,
        delete_marker_file=pathlib.Path(args.out_dir, "delete-markers.csv").resolve(),
        enabled_experiments=args.experiments,
        out_dir=args.out_dir
    )

    configure_pg(conf)

    if args.mode == "shift-only":
        os.makedirs(conf.out_dir, exist_ok=True)
        log(f"Setting up fresh {args.benchmark} instance")
        update_database(conf)
        log("Performing manual data shift")
        manual_shift(conf, fill_factor=args.fill_factor)
        return

    if args.mode == "baseline" or args.mode == "full":
        os.makedirs(conf.out_dir, exist_ok=True)
        log(f"Setting up fresh {args.benchmark} instance")
        update_database(conf)
        log("Obtaining baseline plans")
        obtain_baseline_plans(conf)
    if args.mode == "full":
        log("Resetting IMDB instance")
        update_database(conf)
    if args.mode == "shift" or args.mode == "full":
        log("Simulating data shift")
        simulate_data_shift(conf)
    if args.mode == "single":
        benchmark_current_db(conf, fill_factor=args.fill_factor)


if __name__ == "__main__":
    main()
