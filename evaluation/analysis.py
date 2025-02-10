from __future__ import annotations

import json
import math
from collections.abc import Iterable
from typing import Optional

import pandas as pd
import natsort
import numpy as np
from matplotlib import axes

from postbound.db import postgres
from postbound.experiments import workloads
from postbound.optimizer import jointree


def extract_family(label: str) -> int:
    return int(label[:-1])


def label_sort(df: pd.DataFrame, *, col_names: str | Iterable[str] = "label") -> pd.DataFrame:
    return df.sort_values(by=col_names, key=lambda series: np.argsort(natsort.index_natsorted(series)))


def family_labels(g: axes.Axes, workload: workloads.Workload) -> None:
    g.xaxis.set_ticks(workload.labels())
    g.set_xticklabels([label if label.get_text()[-1] == "a" else "" for label in g.get_xticklabels()])


def load_pg_explain(raw_explain: str) -> Optional[postgres.PostgresExplainPlan]:
    plan_json = json.loads(raw_explain)
    if not plan_json:
        return None
    return postgres.PostgresExplainPlan(plan_json)


def parse_pg_plan(sample: pd.Series, workload: workloads.Workload,
                  label_col: str = "label", plan_col: str = "query_result", **kwargs) -> jointree.PhysicalQueryPlan:
    query = workload[sample[label_col]]
    if not isinstance(sample[plan_col], postgres.PostgresExplainPlan):
        raw_plan = json.loads(sample[plan_col])
        postgres_plan = postgres.PostgresExplainPlan(raw_plan)
    else:
        postgres_plan = sample[plan_col]
    return jointree.PhysicalQueryPlan.load_from_query_plan(postgres_plan.as_query_execution_plan(), query, **kwargs)


def determine_fake_medians(df: pd.DataFrame, *, label_col: str = "label", median_col: str = "execution_time") -> pd.DataFrame:
    def _select_medians(grp: pd.DataFrame) -> pd.Series:
        grp = grp.sort_values(by=median_col)
        fake_median_idx = math.floor(len(grp) / 2)
        fake_median_sample = grp.iloc[fake_median_idx]
        return fake_median_sample

    return (df
            .groupby(label_col, as_index=False)
            .apply(_select_medians, include_groups=False)
            .pipe(label_sort, col_name=label_col))
