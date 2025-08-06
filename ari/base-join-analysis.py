#!/usr/bin/env python3
# This script is basically a stripped-down version of the analysis in the Section-5-CandidatePlans notebook.
# It contains only those parts that we need to compute the queries with the highest base join importance for usage
# in the ARI pipeline.

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from postbound.db import db, postgres
from postbound.experiments import workloads
from postbound.util import collections as collection_utils
from postbound.util import jsonize


def load_pg_explain(raw_explain: str) -> Optional[postgres.PostgresExplainPlan]:
    plan_json = json.loads(raw_explain)
    if not plan_json:
        return None
    return postgres.PostgresExplainPlan(plan_json)


def load_plan_space_df(benchmark: workloads.Workload) -> pd.DataFrame:
    full_df = pd.DataFrame()
    results_dir = (
        Path("/ari/results/experiment-03-plan-space-analysis") / benchmark.name.lower()
    )

    for label in benchmark.labels():
        csv_path = results_dir / f"plan-space-analysis-{label}.csv"

        current_df = pd.read_csv(csv_path, converters={"query_plan": load_pg_explain})
        current_df["label"] = pd.Categorical(
            current_df["label"], categories=benchmark.labels(), ordered=True
        )
        current_df["estimated_cost"] = current_df["query_plan"].map(
            lambda plan: plan.explain_data["Plan"]["Total Cost"]
        )
        current_df["plan_hash"] = current_df["query_plan"].map(hash)

        full_df = pd.concat([full_df, current_df], ignore_index=True)

    return full_df


def lookup_base_joins(
    qep: db.QueryExecutionPlan | postgres.PostgresExplainPlan,
) -> set[db.QueryExecutionPlan]:
    qep = (
        qep if isinstance(qep, db.QueryExecutionPlan) else qep.as_query_execution_plan()
    )
    return (
        {qep}
        if qep.is_base_join()
        else collection_utils.set_union(
            lookup_base_joins(child) for child in qep.children
        )
    )


def make_base_join_df(df: pd.DataFrame) -> pd.DataFrame:
    df["base_join"] = (
        df["query_plan"]
        .map(lookup_base_joins)
        .map(lambda joins: {frozenset(join.tables()) for join in joins})
    )
    df = df.explode("base_join")
    df["join_label"] = df["base_join"].map(
        lambda join: " ⋈ ".join(tab.identifier() for tab in sorted(join))
    )
    return df


def make_importance_dfs(
    plan_df: pd.DataFrame, base_join_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # First, compute the minimum/maximum runtimes and the top 25% threshold for each query
    rt_summary = (
        plan_df.groupby(  # don't use base_join_df here, it contains duplicates which skew the quantile!
            "label", as_index=False, observed=False
        )
        .agg(
            min_rt=pd.NamedAgg(column="runtime", aggfunc="min"),
            max_rt=pd.NamedAgg(column="runtime", aggfunc=lambda rts: rts.quantile(0.9)),
        )
        .assign(
            top25_rt=lambda sample: 0.75 * sample["min_rt"] + 0.25 * sample["max_rt"]
        )
    )  # unrolled form: min + 0.25 * (max - min)

    # Now, determine how many of the execution plans are in the top 25% for each query
    top25_plans = (
        plan_df.merge(  # see comment above: never use base_join_df here
            rt_summary, on="label"
        )
        .query("runtime <= top25_rt")
        .groupby(
            ["label", "top25_rt"], as_index=False, observed=True
        )  # top25_rt is dependent, we just carry it along
        .size()
        .rename(columns={"size": "total_top25_plans"})
    )

    # We are ready to compute the F1 score for each base join
    join_importance_df = (
        base_join_df.merge(top25_plans, on="label")
        .assign(top25_indicator=lambda sample: sample["runtime"] <= sample["top25_rt"])
        .groupby(
            ["label", "base_join", "join_label", "total_top25_plans"],
            as_index=False,
            observed=True,
        )  # total_top25_plans is dependent, we just carry it along
        .agg(
            base_join_plans=pd.NamedAgg(
                column="plan_hash", aggfunc="nunique"
            ),  # how many plans do we have for this base join?
            # (Theoretically, we don't need to do nunique here since the plans should be unique anyway,
            # but its more expressive..)
            top25_plans=pd.NamedAgg(column="top25_indicator", aggfunc="sum"),
        )  # how many of these plans are in the top 25%?
        .assign(
            precision=lambda sample: sample["top25_plans"] / sample["base_join_plans"],
            recall=lambda sample: sample["top25_plans"] / sample["total_top25_plans"],
            f1_score=lambda sample: 2
            * sample["precision"]
            * sample["recall"]
            / (sample["precision"] + sample["recall"]),
        )
        .sort_values(by=["label", "join_label"])
    )

    # Prepare for the aggregated F1 scores: determine the weighting factor for each base join
    importance_weigths = (
        join_importance_df.sort_values(by="f1_score", ascending=False)
        .groupby("label", as_index=False, observed=True)["join_label"]
        .transform(lambda sample: np.arange(len(sample)) + 1)
        .to_frame()
        .rename(columns={"join_label": "harmonic_weight"})
    )

    # Also, we will need to know which base join was actually the best for each query, so let's just compute this here as well
    max_f1s = (
        join_importance_df.assign(
            max_f1=(
                join_importance_df.groupby("label", observed=True)[
                    "f1_score"
                ].transform("max")
            )
        )
        .query("f1_score == max_f1")
        .rename(columns={"base_join": "best_join"})[["label", "best_join"]]
    )

    # Finally, all that's left to do is aggregate
    harmonic_importance = (
        join_importance_df.merge(importance_weigths, left_index=True, right_index=True)
        .merge(max_f1s, on="label")
        .assign(
            f1_harmonic=lambda sample: 1
            / sample["harmonic_weight"]
            * sample["f1_score"]
        )
        .groupby(["label", "best_join"], as_index=False, observed=True)
        .agg(f1_harmonic=pd.NamedAgg(column="f1_harmonic", aggfunc="sum"))
    )

    return join_importance_df, harmonic_importance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=["job", "stats"], default="job")
    parser.add_argument("--out")

    args = parser.parse_args()

    workloads.workloads_base_dir = "/ari/postbound/workloads"
    workload = workloads.job() if args.workload == "job" else workloads.stats()

    plan_space_df = load_plan_space_df(workload)
    base_joins_df = make_base_join_df(plan_space_df)
    _, harmonic_importance = make_importance_dfs(plan_space_df, base_joins_df)

    resample_df = (
        harmonic_importance[
            harmonic_importance["f1_harmonic"]
            >= harmonic_importance["f1_harmonic"].quantile(0.9)
        ]
        .sort_values(by="f1_harmonic", ascending=False)
        .assign(base_join=lambda sample: sample["best_join"].apply(jsonize.to_json))
    )

    resample_df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
