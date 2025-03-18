#!/bin/bash

set -e

FILL_FACTORS=(0.05 0.1 0.25 0.5 0.75)

for fill_factor in ${FILL_FACTORS[@]}; do
    mkdir -p "results/job/job-stats-stability/fill-factor-$fill_factor"

    echo "$(date --rfc-3339=s) Preparing IMDB instance for fill factor $fill_factor"
    python3 -m experiments.experiment-07-optimizer-architectures \
        --benchmark job \
        --fill-factor "$fill_factor" \
        --out-dir "results/job/job-stats-stability/fill-factor-$fill_factor" \
        shift-only

    echo "$(date --rfc-3339=s) Running stability experiment for fill factor $fill_factor"
    python3 -m experiments.experiment-pg-analyze-stability \
        --benchmark job \
        --suffix "fill_factor_$fill_factor" \
        --out-dir "results/job/job-stats-stability/fill-factor-$fill_factor" \
        --repetitions 10
done
