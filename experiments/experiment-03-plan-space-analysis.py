
from __future__ import annotations

import argparse
import collections
import dataclasses
import itertools
import json
import math
import pathlib
import signal
from collections.abc import Callable, Generator, Sequence
from datetime import datetime
from typing import Literal, Optional

import pandas as pd

from postbound.db import postgres
from postbound.experiments import workloads
from postbound.qal import base, qal, transform
from postbound.optimizer import jointree
from postbound.optimizer.policies import cardinalities
from postbound.optimizer.strategies import enumeration, randomized
from postbound.util import jsonize, logging


NoGeQO = postgres.PostgresSetting("geqo", "off")
logger = logging.make_logger(prefix=logging.timestamp)

# ========================
# Sampling infrastructure
# ========================

PlanStructure = Literal["bushy", "left-deep"]
CostArea = Literal["lower", "higher", "close", "arbitrary"]
SampleGenerator = Literal["exhaustive", "sampled_guided", "sampled_uniform"]
SamplingSuccessPolicy = Callable[[jointree.LogicalJoinTree], bool]


@dataclasses.dataclass
class SamplingReport:
    label: str
    query: qal.SqlQuery
    base_table: Optional[base.TableReference]
    plan_structure: PlanStructure
    cost_area: CostArea
    sampling_duration: float
    n_requested: int
    n_produced: int
    generator: SampleGenerator = "sampled"


SamplingResult = collections.namedtuple("SamplingResult", ["plans", "reports"])


def try_exhaustive_enumeration(query: qal.SqlQuery, *, query_label: str, n_plans: int) -> Optional[SamplingResult]:
    enumerator = enumeration.ExhaustiveJoinOrderEnumerator("bushy")
    n_generated = 0
    plans: list[jointree.LogicalJoinTree] = []

    enumeration_start = datetime.now()
    for plan in enumerator.all_join_orders_for(query):
        plans.append(plan)
        n_generated += 1
        if n_generated > n_plans:
            break
    enumeration_end = datetime.now()

    if n_generated > n_plans:
        # Too many plans to enumerate exhaustively
        return None

    enumeration_duration = (enumeration_end - enumeration_start).total_seconds()
    report = SamplingReport(label=query_label, query=query,
                            base_table=None, plan_structure="bushy", cost_area="arbitrary",
                            sampling_duration=enumeration_duration,
                            n_requested=n_plans, n_produced=n_generated,
                            generator="exhaustive")
    return SamplingResult(plans=set(plans), reports=[report])


def initialize_plan_sampler(plan_structure: PlanStructure) -> randomized.RandomJoinOrderGenerator:
    return randomized.RandomJoinOrderGenerator(eliminate_duplicates=False, tree_structure=plan_structure)


def sample_uniform_base_table_plans(generator: randomized.RandomJoinOrderGenerator, query: qal.SqlQuery, *,
                                    plan_structure: PlanStructure, query_label: str,
                                    n_plans: int, retry_factor: float = 5.0,
                                    card_provider: cardinalities.PreComputedCardinalities,
                                    pg_instance: postgres.PostgresInterface) -> SamplingResult:
    cost_areas = ["lower", "higher", "close"]
    base_tables = query.tables()
    plans_per_table = math.ceil(n_plans / len(base_tables))

    all_plans: set[jointree.LogicalJoinTree] = set()
    reports: list[SamplingReport] = []

    for cost_area, base_table in itertools.product(cost_areas, base_tables):
        current_result = sample_cost_based(generator.random_join_orders_for(query, base_table=base_table), query,
                                           cost_area=cost_area, base_table=base_table, plan_structure=plan_structure,
                                           query_label=query_label,
                                           n_plans=plans_per_table, retry_factor=retry_factor,
                                           card_provider=card_provider,
                                           pg_instance=pg_instance)
        all_plans.update(current_result.plans)
        reports.extend(current_result.reports)

    return SamplingResult(plans=all_plans, reports=reports)


def sample_cost_based(generator: Generator[jointree.LogicalJoinTree, None, None], query: qal.SqlQuery, *,
                      cost_area: CostArea, base_table: base.TableReference, plan_structure: PlanStructure,
                      n_plans: int, retry_factor: float = 5.0,
                      card_provider: cardinalities.PreComputedCardinalities,
                      query_label: str,
                      pg_instance: postgres.PostgresInterface) -> SamplingResult:
    pg_instance.apply_configuration(NoGeQO)
    native_query_plan = pg_instance.optimizer().query_plan(query)
    physical_plan = jointree.PhysicalQueryPlan.load_from_query_plan(native_query_plan, query, operators_only=True)
    actual_cardinalities = card_provider.generate_plan_parameters(query, physical_plan.as_logical_join_tree(), None)
    actual_cardinality_hinted_query = pg_instance.hinting().generate_hints(query, physical_plan,
                                                                           plan_parameters=actual_cardinalities)

    native_plan_cost = pg_instance.optimizer().cost_estimate(actual_cardinality_hinted_query)

    n_tries = 0
    max_tries = math.ceil(n_plans * retry_factor)
    sampled_plans: set[jointree.LogicalJoinTree] = set()

    sampling_start = datetime.now()
    while len(sampled_plans) < n_plans and n_tries < max_tries:
        next_plan = next(generator)

        hinted_query = pg_instance.hinting().generate_hints(query, physical_plan, plan_parameters=actual_cardinalities)
        current_cost = pg_instance.optimizer().cost_estimate(hinted_query)

        lower_match = cost_area == "lower" and 1.3 * current_cost <= native_plan_cost
        higher_match = cost_area == "higher" and current_cost >= 1.3 * native_plan_cost
        close_match = cost_area == "close" and 0.7 * native_plan_cost <= current_cost <= 1.3 * native_plan_cost
        if lower_match or higher_match or close_match:
            sampled_plans.add(next_plan)

        n_tries += 1
    sampling_end = datetime.now()

    sampling_duration = (sampling_end - sampling_start).total_seconds()
    report = SamplingReport(label=query_label, query=query,
                            base_table=base_table, plan_structure=plan_structure, cost_area=cost_area,
                            generator="sampled_guided", sampling_duration=sampling_duration,
                            n_requested=n_plans, n_produced=len(sampled_plans))
    return SamplingResult(plans=sampled_plans, reports=[report])


def sample_uniform(query_label: str, query: qal.SqlQuery, *, n_plans: int, retry_factor: float = 5.0,
                   existing_plans: set[jointree.LogicalJoinTree]) -> SamplingResult:
    all_sampled_plans = set(existing_plans)
    plan_generator = (randomized.RandomJoinOrderGenerator(eliminate_duplicates=False, tree_structure="bushy")
                      .random_join_orders_for(query))
    fresh_plans: set[jointree.LogicalJoinTree] = set()
    n_tries = 0
    max_tries = math.ceil(n_plans * retry_factor)

    sampling_start = datetime.now()
    while len(fresh_plans) < n_plans and n_tries < max_tries:
        n_tries += 1
        next_plan = next(plan_generator)

        if next_plan in all_sampled_plans:
            continue
        fresh_plans.add(next_plan)
        all_sampled_plans.add(next_plan)
    sampling_end = datetime.now()

    sampling_duration = (sampling_end - sampling_start).total_seconds()
    report = SamplingReport(label=query_label, query=query,
                            base_table=None, plan_structure="bushy", cost_area="arbitrary",
                            sampling_duration=sampling_duration,
                            n_requested=n_plans, n_produced=len(fresh_plans),
                            generator="sampled_uniform")
    return SamplingResult(plans=fresh_plans, reports=[report])

# ================
# Execution setup
# ================


@dataclasses.dataclass
class ExperimentResult:
    label: str
    runtime: float
    timeout: bool
    query_plan: postgres.PostgresExplainPlan


def execute_query(label: str, join_order: jointree.LogicalJoinTree, *,
                  benchmark: workloads.Workload,
                  cardinalitiy_provider: cardinalities.CardinalityHintsGenerator,
                  database: postgres.PostgresInterface, timeout: float) -> ExperimentResult:
    executor = postgres.TimeoutQueryExecutor(database)
    query = benchmark[label]
    true_cardinalities = cardinalitiy_provider.generate_plan_parameters(query, join_order, None)
    hinted_query = database.hinting().generate_hints(query, join_order, plan_parameters=true_cardinalities)
    hinted_query = transform.as_explain_analyze(hinted_query)

    database.apply_configuration(NoGeQO)
    database.prewarm_tables(query.tables())

    try:
        runtime_start = datetime.now()
        raw_explain_output = executor.execute_query(hinted_query, timeout=timeout)
        runtime_end = datetime.now()
        query_plan = postgres.PostgresExplainPlan(raw_explain_output)
        total_runtime = (runtime_end - runtime_start).total_seconds()
        timeout = False
    except TimeoutError:
        hinted_query = transform.as_explain(hinted_query)
        raw_explain_output = database.execute_query(hinted_query, cache_enabled=False)
        query_plan = postgres.PostgresExplainPlan(raw_explain_output)
        total_runtime = timeout
        timeout = True

    return ExperimentResult(label=label, runtime=total_runtime, timeout=timeout, query_plan=query_plan)


def determine_timeout(label: str, *, min_timeout: float, max_timeout: Optional[float], native_runtime_references: pd.DataFrame,
                      label_col: str = "label", runtime_col: str = "execution_time") -> float:
    native_runtime = native_runtime_references.query(f"{label_col} == @label")[runtime_col].median()
    timeout = max(min_timeout, 3 * native_runtime)
    if max_timeout is not None and timeout > max_timeout:
        timeout = 1.01 * native_runtime
    return timeout


# ========================
# Main experiment control
# ========================

SampleOutFileFormat = "plan-space-analysis-{label}.csv"
CancelEvaluation = False


def skip_existing_results(out_dir: str, *, benchmark: workloads.Workload) -> workloads.Workload[str]:
    wdir = pathlib.Path(out_dir)
    existing_files = {file.name for file in wdir.glob("*.csv")}
    queries_to_evaluate: set[str] = set()
    skipped_queries: set[str] = set()

    for label in benchmark.labels():
        target_file = SampleOutFileFormat.format(label=label)
        if target_file not in existing_files:
            queries_to_evaluate.add(label)
        else:
            skipped_queries.add(label)

    logger("Skipping queries with existing results:", skipped_queries)
    return benchmark.with_labels(queries_to_evaluate)


def flush_sampling_reports(reports: Sequence[SamplingReport], out_dir: str) -> None:
    target_path = pathlib.Path(out_dir) / "sampling-report.csv"
    existing_reports = pd.read_csv(target_path) if target_path.exists() else pd.DataFrame()
    new_reports = pd.DataFrame(reports)
    new_reports["base_table"] = new_reports.base_table.apply(jsonize.to_json)
    combined_reports = pd.concat([existing_reports, new_reports], ignore_index=True)
    combined_reports.to_csv(target_path, index=False)


def ctl_c_handler(sig, frame) -> None:
    global CancelEvaluation
    CancelEvaluation = True


def benchmark_missing_queries(out_dir: str, *,
                              benchmark: workloads.Workload,
                              pg_instance: postgres.PostgresInterface,
                              n_plans: int, linear_fraction: float = 0.7, retry_factor: float = 5.0,
                              min_timeout: float = 15.0, max_timeout: Optional[float],
                              native_runtimes_file: str, cardinalities_file: str,
                              forced_queries: Optional[set[str]] = None) -> None:
    signal.signal(signal.SIGINT, ctl_c_handler)
    true_card_generator = cardinalities.PreComputedCardinalities(benchmark, cardinalities_file, error_on_missing_card=False)
    if forced_queries:
        queries_debug_msg = ", ".join(forced_queries)
        logger("Only evaluating queries", queries_debug_msg)
        remaining_workload = benchmark.with_labels(forced_queries)
    else:
        remaining_workload = skip_existing_results(out_dir, benchmark=benchmark)

    native_runtimes = pd.read_csv(native_runtimes_file)
    n_linear_plans = math.ceil(n_plans * linear_fraction)
    n_bushy_plans = n_plans - n_linear_plans

    for label, query in remaining_workload.entries():
        if CancelEvaluation:
            break

        current_timeout = determine_timeout(label, min_timeout=min_timeout, max_timeout=max_timeout,
                                            native_runtime_references=native_runtimes)
        current_plans: list[jointree.LogicalJoinTree] = []
        current_reports: list[SamplingReport] = []
        current_results: list[ExperimentResult] = []

        exhaustive_generation = try_exhaustive_enumeration(query, query_label=label, n_plans=n_plans)
        if exhaustive_generation is not None:
            logger("Generating", n_plans, "plans for label", label, "exhaustively")
            sampled_plans, sampling_reports = exhaustive_generation
            current_plans.extend(sampled_plans)
            current_reports.extend(sampling_reports)
        else:
            logger("Falling back to sampling-based generation for label", label)
            linear_plan_generator = initialize_plan_sampler("left-deep")
            sampled_plans, sampling_reports = sample_uniform_base_table_plans(linear_plan_generator, query,
                                                                              plan_structure="left-deep", query_label=label,
                                                                              n_plans=n_linear_plans,
                                                                              retry_factor=retry_factor,
                                                                              card_provider=true_card_generator,
                                                                              pg_instance=pg_instance)
            current_plans.extend(sampled_plans)
            current_reports.extend(sampling_reports)

            bushy_plan_generator = initialize_plan_sampler("bushy")
            sampled_plans, sampling_reports = sample_uniform_base_table_plans(bushy_plan_generator, query,
                                                                              plan_structure="bushy", query_label=label,
                                                                              n_plans=n_bushy_plans,
                                                                              retry_factor=retry_factor,
                                                                              card_provider=true_card_generator,
                                                                              pg_instance=pg_instance)
            current_plans.extend(sampled_plans)
            current_reports.extend(sampling_reports)

        logger("Executing", len(current_plans), "plans for label", label, "(timeout =", round(current_timeout, 2), "seconds)")
        for plan in current_plans:
            if CancelEvaluation:
                break

            result = execute_query(label, plan,
                                   benchmark=benchmark, database=pg_instance,
                                   cardinalitiy_provider=true_card_generator,
                                   timeout=current_timeout)
            current_results.append(result)
        if CancelEvaluation:
            break

        flush_sampling_reports(current_reports, out_dir)
        current_df = pd.DataFrame(current_results)
        current_df["query_plan"] = current_df.query_plan.apply(jsonize.to_json)

        n_timeouts = current_df.timeout.sum()
        logger("Flushing results for label", label, "::", n_timeouts, "plans out of", len(current_df), "timed out")

        out_file = pathlib.Path(out_dir) / SampleOutFileFormat.format(label=label)
        current_df.to_csv(out_file, index=False)


def make_explain_parser(label: str, *, benchmark: workloads.Workload) -> Callable[[str], jointree.LogicalJoinTree]:
    def loader(raw_plan: str) -> jointree.LogicalJoinTree:
        json_data = json.loads(raw_plan)
        pg_plan = postgres.PostgresExplainPlan(json_data)
        return jointree.LogicalJoinTree.load_from_query_plan(pg_plan.as_query_execution_plan(), benchmark[label])
    return loader


def fill_missed_samples(out_dir: str, *,
                        benchmark: workloads.Workload, pg_instance: postgres.PostgresInterface,
                        n_target_plans: int, retry_factor: float = 5.0,
                        min_timeout: float = 15.0, max_timeout: Optional[float], native_runtimes_file: str,
                        cardinalities_file: str) -> None:
    signal.signal(signal.SIGINT, ctl_c_handler)
    native_runtimes = pd.read_csv(native_runtimes_file)
    true_card_generator = cardinalities.PreComputedCardinalities(benchmark, cardinalities_file, error_on_missing_card=False)
    prev_sampling_report = pd.read_csv(pathlib.Path(out_dir) / "sampling-report.csv")

    for label, query in benchmark.entries():
        current_report = prev_sampling_report.query("label == @label")
        if len(current_report) == 1:
            current_report = current_report.iloc[0]
            if current_report["generator"] == "exhaustive":
                logger("Skipping query", label, "as it was already sampled exhaustively")
                continue

        out_file = pathlib.Path(out_dir) / SampleOutFileFormat.format(label=label)
        if not out_file.exists():
            logger("Skipping query", label, "as no results are available")
            continue

        new_results: list[ExperimentResult] = []
        current_timeout = determine_timeout(label, min_timeout=min_timeout, max_timeout=max_timeout,
                                            native_runtime_references=native_runtimes)

        raw_existing_results = pd.read_csv(out_file)
        existing_results = raw_existing_results.copy()
        existing_results["query_plan"] = existing_results["query_plan"].apply(make_explain_parser(label, benchmark=benchmark))

        # we determine the number of missing query plans directly based on the number of existing samples
        # this is easier then querying the sampling report which contains individual numbers for each base table
        n_missing_samples = n_target_plans - len(existing_results)
        if n_missing_samples <= 0:
            logger("Skipping query", label, "as no results are missing")
            continue

        logger("Adding missing samples for query", label)
        tested_samples = set(existing_results["query_plan"])
        current_samples, sampling_report = sample_uniform(label, query, n_plans=n_missing_samples, retry_factor=retry_factor,
                                                          existing_plans=tested_samples)
        logger("Generated", len(current_samples), "samples out of", n_missing_samples, "requested for query", label,
               "(timeout =", round(current_timeout, 2), "seconds)")
        if not len(current_samples):
            continue

        for plan in current_samples:
            if CancelEvaluation:
                break
            result = execute_query(label, plan,
                                   benchmark=benchmark, database=pg_instance,
                                   cardinalitiy_provider=true_card_generator,
                                   timeout=current_timeout)
            new_results.append(result)
        if CancelEvaluation:
            break

        flush_sampling_reports(sampling_report, out_dir)
        new_df = pd.DataFrame(new_results)
        new_df["query_plan"] = new_df.query_plan.apply(jsonize.to_json)

        n_timeouts = new_df.timeout.sum()
        logger("Flushing results for label", label, "::", n_timeouts, "plans out of", len(new_df), "timed out")

        out_file = pathlib.Path(out_dir) / SampleOutFileFormat.format(label=label)
        all_results = pd.concat([raw_existing_results, new_df])
        all_results.to_csv(out_file, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=str, default="job", choices=["job", "stats"])
    parser.add_argument("--out-dir", type=str, default="results/local")
    parser.add_argument("--n-plans", type=int, default=1000)
    parser.add_argument("--native-rts", type=str, default="results/job/job-native-runtimes.csv")
    parser.add_argument("--cardinalities", type=str, default="results/job/job-intermediate-cardinalities.csv")
    parser.add_argument("--queries", type=str)
    parser.add_argument("--fill-samples", action="store_true")
    parser.add_argument("--max-timeout", type=float, default=None)
    parser.add_argument("--workloads-dir", action="store", help="The directory where the workloads are stored.")
    parser.add_argument("--db-conn", "-c", action="store", help="The path to the database connection file.")

    args = parser.parse_args()

    if args.workloads_dir:
        workloads.workloads_base_dir = args.workloads_dir

    if args.bench == "job":
        benchmark = workloads.job()
        db_conf = args.db_conn if args.db_conn else ".psycopg_connection_job"
        pg_instance = postgres.connect(config_file=db_conf)
    elif args.bench == "stats":
        benchmark = workloads.stats()
        db_conf = args.db_conn if args.db_conn else ".psycopg_connection_stats"
        pg_instance = postgres.connect(config_file=db_conf)
    else:
        parser.error(f"Invalid workload specified: '{args.bench}'")

    if args.fill_samples:
        fill_missed_samples(args.out_dir, benchmark=benchmark, pg_instance=pg_instance,
                            n_target_plans=args.n_plans, native_runtimes_file=args.native_rts,
                            max_timeout=args.max_timeout,
                            cardinalities_file=args.cardinalities)
    else:
        queries = set(args.queries.split(",")) if args.queries is not None else []
        benchmark_missing_queries(args.out_dir, benchmark=benchmark, pg_instance=pg_instance,
                                  n_plans=args.n_plans, native_runtimes_file=args.native_rts,
                                  max_timeout=args.max_timeout,
                                  cardinalities_file=args.cardinalities, forced_queries=queries)


if __name__ == "__main__":
    main()
