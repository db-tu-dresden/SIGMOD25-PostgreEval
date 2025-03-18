
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from postbound.db import postgres
from postbound.qal import base, qal, transform
from postbound.experiments import workloads
from postbound.optimizer import jointree, planparams
from postbound.optimizer.policies import cardinalities
from postbound.util import jsonize, logging

JoinOrderResultsDir = "results/local/join-order-variation"
MinTimeout = 15.0
NoGeQO = postgres.PostgresSetting("geqo", "off")

pg_instance = postgres.connect(config_file=".psycopg_connection_job")
job = workloads.job()
true_cards = cardinalities.PreComputedCardinalities(job, "results/job/job-intermediate-cardinalities.csv")
log = logging.make_logger(prefix=logging.timestamp)


def parse_query_plan(raw_plan: str, *, label: str) -> jointree.LogicalJoinTree | float:
    if not raw_plan:
        return math.nan
    jsonized_plan = json.loads(raw_plan)
    pg_plan = postgres.PostgresExplainPlan(jsonized_plan)
    return jointree.LogicalJoinTree.load_from_query_plan(pg_plan.as_query_execution_plan(), job[label])


def check_base_join(plan: jointree.LogicalJoinTree, target_join: set[base.TableReference]) -> bool:
    for join in plan.join_sequence():
        if target_join != join.tables():
            continue

        # our target table is part of the current join
        return join.is_base_join()  # if the join is not a base join, we are done. If it is, we found a correct plan.

    return False


def join_orders_with_base_join(label: str, join: set[base.TableReference], *,
                               join_dir: str) -> Iterable[jointree.LogicalJoinTree]:
    results_df = pd.read_csv(f"{join_dir}/plan-space-analysis-{label}.csv",
                             converters={"query_plan": lambda raw_plan: parse_query_plan(raw_plan, label=label)})
    results_df.dropna(subset=["query_plan"], inplace=True)
    matching_samples = results_df[results_df["query_plan"].apply(check_base_join, target_join=join)]
    return matching_samples["query_plan"].values


def generate_workload(label: str, join: set[base.TableReference],
                      logical_plans: Iterable[jointree.LogicalJoinTree]) -> list[qal.SqlQuery]:
    query = job[label]
    hinted_queries = []

    cardinality_hints = planparams.PlanParameterization()
    tab1, tab2 = join
    cardinality_hints.add_cardinality_hint({tab1}, true_cards.calculate_estimate(query, {tab1}))
    cardinality_hints.add_cardinality_hint({tab2}, true_cards.calculate_estimate(query, {tab2}))
    cardinality_hints.add_cardinality_hint(join, true_cards.calculate_estimate(query, join))

    for plan in logical_plans:
        hinted = pg_instance.hinting().generate_hints(query, plan, plan_parameters=cardinality_hints)
        hinted_queries.append(hinted)
    return hinted_queries


@dataclass
class ExperimentResult:
    label: str
    runtime: float
    timeout: bool
    query_plan: postgres.PostgresExplainPlan


def execute_query(label: str, hinted_query: qal.SqlQuery,
                  database: postgres.PostgresInterface, timeout: float) -> ExperimentResult:
    executor = postgres.TimeoutQueryExecutor(database)
    query = job[label]
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


def determine_timeout(label: str, *, min_timeout: float, native_runtime_references: pd.DataFrame,
                      label_col: str = "label", runtime_col: str = "exec_time") -> float:
    native_runtime = native_runtime_references.query(f"{label_col} == @label")[runtime_col].median()
    return max(min_timeout, 3 * native_runtime)


def execute_workload(label: str, join: set[base.TableReference], *, native_rts: pd.DataFrame, join_dir: str,
                     out_dir: str) -> None:
    matching_plans = join_orders_with_base_join(label, join, join_dir=join_dir)
    target_queries = generate_workload(label, join, matching_plans)

    log("Executing", len(target_queries), "plans for query", label)
    results: list[ExperimentResult] = []
    timeout = determine_timeout(label, min_timeout=MinTimeout, native_runtime_references=native_rts)  # TODO

    for query in target_queries:
        current_res = execute_query(label, query, pg_instance, timeout)
        results.append(current_res)

    result_df = pd.DataFrame(results)
    result_df["query_plan"] = result_df.query_plan.apply(jsonize.to_json)
    result_df.to_csv(f"{out_dir}/base-tab-operator-flexibility-{label}.csv", index=False)


def parse_base_join(raw_join: str) -> set[base.TableReference]:
    json_data = json.loads(raw_join)
    return {base.TableReference(tab["full_name"], tab["alias"]) for tab in json_data}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-queries", type=str, help="")
    parser.add_argument("--join-dir", type=str, help="")
    parser.add_argument("--native-rts", type=str, help="")
    parser.add_argument("--out-dir", type=str, help="")

    args = parser.parse_args()

    target_queries = pd.read_csv(args.target_queries, converters={"base_join": parse_base_join})
    native_rts = pd.read_csv(args.native_rts)
    native_rts["median_rt"] = native_rts.groupby("label")["exec_time"].transform("median")
    native_rts = native_rts.query("exec_time == median_rt").copy()

    for _, row in target_queries.iterrows():
        label: str = row.label
        join: set[base.TableReference] = row.base_join

        log("Now processing query", label)
        execute_workload(label, join, native_rts=native_rts, join_dir=args.join_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
