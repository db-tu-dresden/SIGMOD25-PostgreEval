
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import typing
import warnings
from collections.abc import Iterable
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from postbound.db import db, postgres
from postbound.qal import qal, transform
from postbound.experiments import workloads
from postbound.optimizer import jointree, planparams
from postbound.optimizer.policies import cardinalities
from postbound.util import jsonize, logging, dicts as dict_ext

DistortionType = typing.Literal["overestimation", "underestimation", "none"]
CardAssignment = planparams.PlanParameterization
ObservedPlansDict = dict[jointree.PhysicalQueryPlan, tuple[float, db.QueryExecutionPlan]]
DistortionStep = 0.05
DefaultTimeout = 60 * 5  # * 60 (seconds -> minutes) * 5 (1 minute -> 5 minutes)
DefaultOutfile = "results/local/job-pg-card-distortion.csv"
NoGeQO = postgres.PostgresSetting("geqo", "off")
log = logging.make_logger(prefix=logging.timestamp)

warnings.filterwarnings("ignore", category=db.HintWarning)
warnings.filterwarnings("ignore", category=db.QueryCacheWarning)


def obtain_query_plan(query: qal.SqlQuery, cardinality_hints: planparams.PlanParameterization, *,
                      pg_instance: postgres.PostgresInterface) -> tuple[jointree.PhysicalQueryPlan, qal.SqlQuery]:
    hinted_query = pg_instance.hinting().generate_hints(query, plan_parameters=cardinality_hints)
    pg_instance.apply_configuration(NoGeQO)
    explain_plan = pg_instance.optimizer().query_plan(hinted_query)
    return jointree.PhysicalQueryPlan.load_from_query_plan(explain_plan, query, operators_only=True), hinted_query


@dataclasses.dataclass
class DistortionResult:
    label: str
    query: str
    distortion_factor: float
    distortion_type: DistortionType
    query_plan: dict
    runtime: float
    cached: bool
    card_estimator: str
    db_config: str
    query_hints: str = ""


def write_distortion_results(results: Iterable[DistortionResult], out: str) -> None:
    fresh_results_df = pd.DataFrame(results)
    if not os.path.exists(out):
        fresh_results_df.to_csv(out, index=False)
        return
    existing_results_df = pd.read_csv(out)
    merged_df = pd.concat([existing_results_df, fresh_results_df])
    merged_df.to_csv(out, index=False)


def obtain_vanilla_results(workload: workloads.Workload[str], *, out: str, pg_instance: postgres.PostgresInterface,
                           card_generator: cardinalities.CardinalityHintsGenerator, base_cards: str,
                           explain_only: bool = False) -> ObservedPlansDict:
    distortion_results: list[DistortionResult] = []
    observed_plans = dict_ext.CustomHashDict(jointree.PhysicalQueryPlan.plan_hash)
    for label, query in workload.entries():
        log("Now obtaining vanilla results for query", label)

        if not explain_only:
            pg_instance.prewarm_tables(query.tables())
        pg_instance.apply_configuration(NoGeQO)

        if isinstance(card_generator, cardinalities.NativeCardinalityHintGenerator):
            native_plan = pg_instance.optimizer().query_plan(query)
            hinted_query = query
            query_plan = jointree.PhysicalQueryPlan.load_from_query_plan(native_plan, query)
        else:
            parameterization = card_generator.estimate_cardinalities(query)
            query_plan, hinted_query = obtain_query_plan(query, parameterization, pg_instance=pg_instance)
        hinted_query = transform.as_explain(hinted_query) if explain_only else transform.as_explain_analyze(hinted_query)

        start_time = datetime.now()
        explain_data = pg_instance.execute_query(hinted_query, cache_enabled=False)
        end_time = datetime.now()
        total_runtime = (end_time - start_time).total_seconds()

        observed_plans[query_plan] = total_runtime, explain_data
        distortion_results.append(DistortionResult(label, query, 1.0, "none", jsonize.to_json(explain_data), total_runtime,
                                                   cached=False, card_estimator=base_cards,
                                                   db_config=jsonize.to_json(pg_instance.describe()),
                                                   query_hints=hinted_query.hints))
    write_distortion_results(distortion_results, out)
    return observed_plans


def obtain_distortion_results(workload: workloads.Workload[str], distortion_factors: Iterable[float], *,
                              out: str, pg_instance: postgres.PostgresInterface, timeout: float,
                              card_generator: cardinalities.CardinalityHintsGenerator, base_cards: str,
                              observed_plans: Optional[ObservedPlansDict] = None,
                              explain_only: bool = False) -> ObservedPlansDict:
    timeout_executor = postgres.TimeoutQueryExecutor(pg_instance)
    observed_plans = (dict_ext.CustomHashDict(jointree.PhysicalQueryPlan.plan_hash)
                      if observed_plans is None else observed_plans)
    for current_distortion in distortion_factors:
        log("Simulating distorted workload for pct", current_distortion)

        distortion_results: list[DistortionResult] = []
        if current_distortion == 0:
            distortion_type = "none"
        elif current_distortion < 1.0:
            distortion_type = "underestimation"
        else:
            distortion_type = "overestimation"
        card_distortion = cardinalities.CardinalityDistortion(card_generator, current_distortion, distortion_strategy="fixed")

        for label, query in workload.entries():
            log("Executing query", label, "for distortion factor", current_distortion)

            if not explain_only:
                pg_instance.prewarm_tables(query.tables())
            pg_instance.apply_configuration(NoGeQO)
            distorted_parameterization = card_distortion.estimate_cardinalities(query)

            query_plan, card_query = obtain_query_plan(query, distorted_parameterization, pg_instance=pg_instance)
            if query_plan in observed_plans:
                log("Reusing existing plan for query", label, "for distortion factor", current_distortion)
                total_runtime, explain_data = observed_plans[query_plan]
                cached = True
            else:
                # we better base our hinted query on the entire query plan to prevent Postgres from switching plans for the
                # cardinality hints
                hinted_query = pg_instance.hinting().generate_hints(query, query_plan)
                hinted_query = (transform.as_explain(hinted_query) if explain_only
                                else transform.as_explain_analyze(hinted_query))
                try:
                    start_time = datetime.now()
                    explain_data = timeout_executor.execute_query(hinted_query, timeout)
                    end_time = datetime.now()
                    total_runtime = (end_time - start_time).total_seconds()
                except TimeoutError:
                    total_runtime = math.inf
                    hinted_query = transform.as_explain(hinted_query)
                    explain_data = pg_instance.execute_query(hinted_query, cache_enabled=False)
                observed_plans[query_plan] = total_runtime, explain_data
                cached = False

            distortion_results.append(DistortionResult(label, query,
                                                       current_distortion, distortion_type,
                                                       jsonize.to_json(explain_data), total_runtime,
                                                       cached=cached, card_estimator=base_cards,
                                                       db_config=jsonize.to_json(pg_instance.describe()),
                                                       query_hints=card_query.hints))

        write_distortion_results(distortion_results, out)

    return observed_plans


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment studying the effect of cardinality misestimates on query plans.")
    parser.add_argument("--benchmark", default="job", choices=["job", "stats"], help="The benchmark to evaluate")
    parser.add_argument("--include-vanilla", action="store_true", help="Include baseline plans without misestimates")
    parser.add_argument("--include-default-underest", action="store_true",
                        help="Simulate underestimation in range [0.05, 0.95] in steps of 0.05")
    parser.add_argument("--include-default-overest", action="store_true",
                        help="Simulate overestimation in range [1.05, 3.0] in steps of 0.05")
    parser.add_argument("--include-extreme-overest", action="store_true", help="Simulate very large overestimations")
    parser.add_argument("--distortion-factor", action="store", type=float, nargs="+", default=[],
                        help="Simulate custom misestimate factors")
    parser.add_argument("--base-cards", action="store", choices=["native", "actual"], default="actual",
                        help="Change baseline cardinalities to use for the misestimates. Allowed values are native "
                        "(using the Postgres optimizer), or the true cardinalities. The latter is the default option.")
    parser.add_argument("--cards-source", action="store", type=str, help="")
    parser.add_argument("--queries", action="store", type=str, nargs="+", default=[],
                        help="Simulate misestimates only for the given query labels")
    parser.add_argument("--workloads-dir", action="store", help="The directory where the workloads are stored.")
    parser.add_argument("--db-conn", "-c", action="store", help="The path to the database connection file.")
    parser.add_argument("--timeout", action="store", type=float, default=DefaultTimeout,
                        help="Max query runtime in seconds. Defaults to 5 minutes.")
    parser.add_argument("--explain-only", action="store_true", help="Only explain the query plans, do not execute them")
    parser.add_argument("--pg-conf", action="store", nargs="*", default=[],
                        help="Specific Postgres configurations applied to the connection before executing any query "
                        "(e.g. SET enable_nestloop TO off;)")
    parser.add_argument("-o", "--out", action="store", default=DefaultOutfile,
                        help=f"Output file to write the results to, defaults to {DefaultOutfile}")

    args = parser.parse_args()

    if args.workloads_dir:
        workloads.workloads_base_dir = args.workloads_dir
    workload = workloads.job() if args.benchmark == "job" else workloads.stats()
    if args.queries:
        workload = workload & set(args.queries)

    distortion_factors = np.empty(0)
    if args.include_default_underest:
        default_underest_factors = [max(round(1.0 - i * DistortionStep, 2), 0) for i in range(1, 20)]
        distortion_factors = np.concatenate([distortion_factors, default_underest_factors])
    if args.include_default_overest:
        default_overest_factors = [round(1.0 + i * DistortionStep, 2) for i in range(1, 40 + 1)]
        distortion_factors = np.concatenate([distortion_factors, default_overest_factors])
    if args.include_extreme_overest:
        extreme_estimation_points = [10, 25, 50, 75]
        extreme_estimation_factors = [10**i for i in range(7)]
        extreme_distortion_factors = sorted(np.outer(extreme_estimation_points, extreme_estimation_factors).flatten())
        distortion_factors = np.concatenate([distortion_factors, extreme_distortion_factors])
    distortion_factors = np.concatenate([distortion_factors, args.distortion_factor])
    if distortion_factors.size == 0 and not args.include_vanilla:
        parser.error("At least one distortion factor is required, but none were given.")

    if args.db_conn:
        pg_conf = args.db_conn
    else:
        pg_conf = ".psycopg_connection_job" if args.benchmark == "job" else ".psycopg_connection_stats"

    if args.cards_source:
        cardinalities_file = args.cards_source
    else:
        cardinalities_file = ("results/job/job-intermediate-cardinalities.csv" if args.benchmark == "job"
                              else "results/stats-ceb/stats-ceb-intermediate-cardinalities.csv")

    pg_instance = postgres.connect(config_file=pg_conf)
    card_generator: cardinalities.CardinalityHintsGenerator = (
        cardinalities.PreComputedCardinalities(workload, cardinalities_file, live_fallback=True)
        if args.base_cards == "actual" else cardinalities.NativeCardinalityHintGenerator(pg_instance))
    observed_plans = dict_ext.CustomHashDict(jointree.PhysicalQueryPlan.plan_hash)

    for pg_param in args.pg_conf:
        pg_instance.apply_configuration(pg_param)

    if args.include_vanilla:
        log("Obtaining vanilla query runtimes")
        observed_plans = obtain_vanilla_results(workload, out=args.out, pg_instance=pg_instance,
                                                card_generator=card_generator, base_cards=args.base_cards,
                                                explain_only=args.explain_only)
    if distortion_factors.size > 0:
        log("Obtaining distorted query runtimes for factors", distortion_factors.tolist())
        obtain_distortion_results(workload, distortion_factors, out=args.out, pg_instance=pg_instance, timeout=args.timeout,
                                  card_generator=card_generator, base_cards=args.base_cards, observed_plans=observed_plans,
                                  explain_only=args.explain_only)


if __name__ == "__main__":
    main()
