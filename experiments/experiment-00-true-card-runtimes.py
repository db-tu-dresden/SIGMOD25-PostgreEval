from __future__ import annotations

import argparse

from postbound import postbound as pb
from postbound.db import postgres
from postbound.experiments import analysis, runner, workloads
from postbound.optimizer.policies import cardinalities
from postbound.util import logging

NoGeQO = postgres.PostgresSetting("geqo", "off")


def main() -> None:
    parser = argparse.ArgumentParser(description="Determine the runtime of query plans with perfect cardinality estimates.")
    parser.add_argument("--benchmark", action="store", choices=["job", "stats"], help="The benchmark to execute")
    parser.add_argument("--cards-file", action="store", default="", help="File containing the cardinality estimates")
    parser.add_argument("--repetitions", action="store", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--prewarm", action="store_true", help="Use perfect warm buffers")
    parser.add_argument("--db-config", action="store", default="", help="Path to the database configuration file")
    parser.add_argument("--out", action="store", default="", help="Name of the output file")
    parser.add_argument("--workloads-dir", action="store", help="The directory where the workloads are stored.")

    args = parser.parse_args()

    if args.workloads_dir:
        workloads.workloads_base_dir = args.workloads_dir

    db_config = args.db_config if args.db_config else ""

    match args.benchmark:
        case "job":
            workload = workloads.job()
            db_config = db_config if db_config else ".psycopg_connection_job"
            cards_file = args.cards_file if args.cards_file else "results/job/job-intermediate-cardinalities.csv"
            outfile = args.out if args.out else "results/job/job-true-card-runtimes.csv"
        case "stats":
            workload = workloads.stats()
            db_config = db_config if db_config else ".psycopg_connection_stats"
            cards_file = args.cards_file if args.cards_file else "results/stats-ceb/stats-intermediate-cardinalities.csv"
            outfile = args.out if args.out else "results/stats-ceb/stats-true-card-runtimes.csv"

    log = logging.make_logger(prefix=logging.timestamp)
    pg_instance = postgres.connect(config_file=db_config)
    pg_instance.apply_configuration(NoGeQO)

    true_card_predictor = cardinalities.PreComputedCardinalities(workload, cards_file)
    optimization_pipeline = (pb.TwoStageOptimizationPipeline(pg_instance)
                             .setup_plan_parameterization(true_card_predictor)
                             .build())

    query_prep = runner.QueryPreparationService(analyze=True, prewarm=args.prewarm)
    results = runner.optimize_and_execute_workload(workload, optimization_pipeline,
                                                   workload_repetitions=args.repetitions, per_query_repetitions=1,
                                                   query_preparation=query_prep,
                                                   logger=log)

    export_df = analysis.prepare_export(results)
    export_df.to_csv(outfile, index=False)


if __name__ == "__main__":
    main()
