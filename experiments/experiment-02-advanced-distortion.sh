#!/bin/bash

set -e

BENCH=job
ENABLE_COUT=off
CM_SUFFIX=vanilla

show_help() {
    RET=$1
    echo "Usage: $0 [options]"
    echo "Allowed Options:"
    echo -e "--bench <name>\t\tBenchmark name (default: job)"
    echo -e "--enable-cout <on|off>\tEnable or disable Cout cost model (default: off)"
    exit $RET
}

while [ $# -gt 0 ]; do
    case $1 in
        --bench)
            BENCH=$2
            shift
            shift
            ;;
        --enable-cout)
            ENABLE_COUT=$2
            shift
            shift
            ;;
        -h|--help)
            show_help 0
            ;;
        *)
            show_help 1
            ;;
    esac
done


if [ "$ENABLE_COUT" == "on" ]; then
    CM_SUFFIX=cout
fi

echo "--- Evaluating $BENCH benchmark"

echo "--- $(date +'%Y-%m-%d %H:%M:%S') - Stage 1"
python3 -m experiments.experiment-01-cardinality-distortion \
    --benchmark $BENCH \
    --include-vanilla \
    --include-default-underest \
    --include-default-overest \
    --base-cards actual \
    --explain-only \
    --pg-conf "
        LOAD 'coutstar';
        SET enable_cout TO $ENABLE_COUT;

        SET enable_seqscan TO on;
        SET enable_nestloop TO on;

        SET enable_indexscan TO off;
        SET enable_hashjoin TO off;
        SET enable_mergejoin TO off;
        SET enable_sort TO off;

        SET enable_indexonlyscan TO off;
        SET enable_bitmapscan TO off;
        SET enable_hashagg TO off;
        SET enable_incremental_sort TO off;

        SET enable_material TO off;
        SET enable_memoize TO off;

        SET max_parallel_workers_per_gather TO 0;
        SET enable_gathermerge TO off;
        SET enable_parallel_hash TO off;
    " \
    --out results/$BENCH/$BENCH-distortion-$CM_SUFFIX-cost-stage-1.csv


echo "--- $(date +'%Y-%m-%d %H:%M:%S') - Stage 2"
python3 -m experiments.experiment-01-cardinality-distortion \
    --benchmark $BENCH \
    --include-vanilla \
    --include-default-underest \
    --include-default-overest \
    --base-cards actual \
    --explain-only \
    --pg-conf "
        LOAD 'coutstar';
        SET enable_cout TO $ENABLE_COUT;

        SET enable_seqscan TO on;
        SET enable_nestloop TO on;

        SET enable_indexscan TO on;
        SET enable_hashjoin TO on;
        SET enable_mergejoin TO on;
        SET enable_sort TO on;

        SET enable_indexonlyscan TO off;
        SET enable_bitmapscan TO off;
        SET enable_hashagg TO off;
        SET enable_incremental_sort TO off;

        SET enable_material TO off;
        SET enable_memoize TO off;

        SET max_parallel_workers_per_gather TO 0;
        SET enable_gathermerge TO off;
        SET enable_parallel_hash TO off;
    " \
    --out results/$BENCH/$BENCH-distortion-$CM_SUFFIX-cost-stage-2.csv

echo "--- $(date +'%Y-%m-%d %H:%M:%S') - Stage 3"
python3 -m experiments.experiment-01-cardinality-distortion \
    --benchmark $BENCH \
    --include-vanilla \
    --include-default-underest \
    --include-default-overest \
    --base-cards actual \
    --explain-only \
    --pg-conf "
        LOAD 'coutstar';
        SET enable_cout TO $ENABLE_COUT;

        SET enable_seqscan TO on;
        SET enable_nestloop TO on;

        SET enable_indexscan TO on;
        SET enable_hashjoin TO on;
        SET enable_mergejoin TO on;
        SET enable_sort TO on;

        SET enable_indexonlyscan TO on;
        SET enable_bitmapscan TO on;
        SET enable_hashagg TO on;
        SET enable_incremental_sort TO on;

        SET enable_material TO off;
        SET enable_memoize TO off;

        SET max_parallel_workers_per_gather TO 0;
        SET enable_gathermerge TO off;
        SET enable_parallel_hash TO off;
    " \
    --out results/$BENCH/$BENCH-distortion-$CM_SUFFIX-cost-stage-3.csv

echo "--- $(date +'%Y-%m-%d %H:%M:%S') - Stage 4"
python3 -m experiments.experiment-01-cardinality-distortion \
    --benchmark $BENCH \
    --include-vanilla \
    --include-default-underest \
    --include-default-overest \
    --base-cards actual \
    --explain-only \
    --pg-conf "
        LOAD 'coutstar';
        SET enable_cout TO $ENABLE_COUT;

        SET enable_seqscan TO on;
        SET enable_nestloop TO on;

        SET enable_indexscan TO on;
        SET enable_hashjoin TO on;
        SET enable_mergejoin TO on;
        SET enable_sort TO on;

        SET enable_indexonlyscan TO on;
        SET enable_bitmapscan TO on;
        SET enable_hashagg TO on;
        SET enable_incremental_sort TO on;

        SET enable_material TO on;
        SET enable_memoize TO on;

        SET max_parallel_workers_per_gather TO 0;
        SET enable_gathermerge TO off;
        SET enable_parallel_hash TO off;
    " \
    --out results/$BENCH/$BENCH-distortion-$CM_SUFFIX-cost-stage-4.csv

echo "--- $(date +'%Y-%m-%d %H:%M:%S') - Stage 5"
python3 -m experiments.experiment-01-cardinality-distortion \
    --benchmark $BENCH \
    --include-vanilla \
    --include-default-underest \
    --include-default-overest \
    --base-cards actual \
    --explain-only \
    --pg-conf "
        LOAD 'coutstar';
        SET enable_cout TO $ENABLE_COUT;

        SET enable_seqscan TO on;
        SET enable_nestloop TO on;

        SET enable_indexscan TO on;
        SET enable_hashjoin TO on;
        SET enable_mergejoin TO on;
        SET enable_sort TO on;

        SET enable_indexonlyscan TO on;
        SET enable_bitmapscan TO on;
        SET enable_hashagg TO on;
        SET enable_incremental_sort TO on;

        SET enable_material TO on;
        SET enable_memoize TO on;

        -- SET max_parallel_workers_per_gather TO 0;
        SET enable_gathermerge TO on;
        SET enable_parallel_hash TO on;
    " \
    --out results/$BENCH/$BENCH-distortion-$CM_SUFFIX-cost-stage-5.csv
