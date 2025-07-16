
from __future__ import annotations

import argparse
import os

from postbound.db import postgres
from postbound.experiments import runner, workloads, analysis
from postbound.util import logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment to determine how much query plans are influenced by "
                                     "different statistics samples.")
    parser.add_argument("--benchmark", "-b", type=str, choices=["job", "stats", "stack"], default="job",
                        help="Benchmark to use")
    parser.add_argument("--min-tables", type=int, default=0, help="Only include queries with at least that many tables")
    parser.add_argument("--repetitions", "-r", type=int, default=10, help="Number of repetitions")
    parser.add_argument("--suffix", "-s", type=str, default="", help="Output file suffix")
    parser.add_argument("--out-dir", "-o", type=str, default="results/local", help="Output directory")
    parser.add_argument("--explain-only", action="store_true", help="Only gather query plans, do not execute them")
    parser.add_argument("--with-geqo", action="store_true", help="Enable the GEQO optimizer")
    parser.add_argument("--workloads-dir", action="store", help="The directory where the workloads are stored.")

    args = parser.parse_args()
    outfile_suffix = f"_{args.suffix}" if args.suffix else ""
    repetitions = args.repetitions
    bench_name = args.benchmark
    out_dir = args.out_dir

    if args.workloads_dir:
        workloads.workloads_base_dir = args.workloads_dir

    logger = logging.make_logger(enabled=True, prefix=logging.timestamp)

    # The Stackoverflow benchmark does not always contain queries with explicit table references
    # (i.e. it contains queries of the form SELECT * FROM R WHERE id < 42 rather than SELECT * FROM R WHERE R.id < 42)
    # In order to parse these tables correctly, we need a working database connection when loading the benchmark. This allows
    # the parser to infer the necessary information directly from the database schema. Therefore, our benchmark-specific
    # database- and workload-setup is a bit repetitive.
    match bench_name:
        case "job":
            pg_instance = postgres.connect(config_file=".psycopg_connection_job")
            benchmark = workloads.job()
        case "stats":
            pg_instance = postgres.connect(config_file=".psycopg_connection_stats")
            benchmark = workloads.stats()
        case "stack":
            pg_instance = postgres.connect(config_file=".psycopg_connection_stack")
            benchmark = workloads.stack()
        case _:
            raise ValueError(f"Unknown benchmark {bench_name}")

    if args.min_tables:
        benchmark = benchmark.filter_by(lambda __, query: len(query.tables()) >= args.min_tables)

    use_explain: bool = args.explain_only
    use_analyze = not use_explain
    use_prewarm = not use_explain
    prep_statements = [] if args.with_geqo else ["SET geqo TO off;"]
    query_config = runner.QueryPreparationService(explain=use_explain, analyze=use_analyze, prewarm=use_prewarm,
                                                  preparatory_statements=prep_statements)

    def update_stats(repetition: int) -> None:
        logger("Updating statistics for repetition", repetition)
        pg_instance.statistics().update_statistics(perfect_mcv=True)
        target_file = f"{out_dir}/stats_dump_{bench_name}_{repetition+1}{outfile_suffix}.tar"
        os.system(f"pg_dump --table=pg_statistic --format=t --file={target_file} {pg_instance.database_name()}")

    os.makedirs(out_dir, exist_ok=True)
    update_stats(-1)  # -1 results in index 0
    result_df = runner.execute_workload(benchmark, pg_instance, workload_repetitions=repetitions, include_labels=True,
                                        query_preparation=query_config, post_repetition_callback=update_stats, logger=logger)
    result_df = analysis.prepare_export(result_df)
    result_df.to_csv(f"{out_dir}/pg-{bench_name}-analyze-stability{outfile_suffix}.csv", index=False)


if __name__ == "__main__":
    main()
