# SIGMOD ARI Submission

This directory contains all artifacts that are required to reproduce the results of our paper as part of the SIGMOD ARI 2025.
Detailed instructions for reviewers follow below.
Please read this document "cover to cover" to understand certain design decisions.


# Reproducibility Strategy

Reproducing all data from the original paper would require executing more than one million queries across three benchmarks
and two hardware platforms.
In total, it took us several months of raw compute time to obtain the original measurements.

We argue that a complete reproduction is infeasible for the community.
At the same time, we also think that a complete reproduction is not necessary to confirm the core findings of the paper.
Therefore, we limit the scope of our reproducibility efforts to a subset of the original experiments.
We describe and justify our approach in the remainder of this section.
Please note that the reproducibility pipeline is in theory capable of reproducing all the results, but the default ARI
configuration just focuses on the core contributions.

## Core Results

TODO

## Reproduced Results

In the ARI pipeline, we perform the following experiments to reproduce the core results

TODO


# System Requirements

TODO

Running the entire ARI pipeline took us about X time on a machine with Y.


# Setup

The entire pipeline is implemented in a Docker environment and does not require any user intervention once started.
To create the Docker container, please run the following commands in the `ari` working directory
(`💻 $` refers to a shell on your host machine whereas `🐳 $` refers to a shell within the Docker container ):

> [!caution]
> Replace the _SHM_SIZE_ placeholder with the desired amount of shared memory. We heavily recommend using at least 64GB, since
> shared memory determines the size of the shared page buffer used by Postgres
> (see [System Requirements](#system-requirements)). If the databases do not fit into RAM, performance measurements become
> unreliable (see Section 3.3 of the paper).

```sh
💻 $ git clone https://github.com/db-tu-dresden/SIGMOD25-PostgreEval.git
💻 $ cd SIGMOD25-PostgreEval/ari
💻 $ docker build -t ari-elephant .
💻 $ docker run -dt \
    --name ari-elephant \
    --volume $PWD/docker-volume:/ari \
    --shm-size=<SHM_SIZE> \
    -e SETUP_JOB=true \
    ari-elephant
```

The initial container start will take some time, since the setup needs to download and compile a new Postgres instance, download
and fetch the workloads, etc.
Use `docker logs -f ari-elephant` to monitor the setup progress and wait for the
_Setup done. You can now reproduce individual experiments from the paper._ message.
Once this message appears, use

```sh
💻 $ docker exec -it ari-elephant /bin/bash
🐳 $ cd /ari && ari/xctl.py --benchmark job --mode full all
```

to start the pipeline.

Running the pipeline will take a _lot_ of time (think weeks, see [Reproducibility Strategy](#reproducibility-strategy)).
You can monitor the progress via the `results/progress.log` file in the Docker volume, or via stdout in the terminal.

The `xctl` utility can be used to adapt the pipeline configuration and re-run specific experiments, even beyond the necessities
for ARI. See [below](#pipeline-arguments) for the complete set of arguments.


# Interpreting the Results

All results are available in the Docker volume (by default `docker-volume`) in the `results/` directory.
The pipeline creates three kinds of artifacts:

1. Raw result data from all experiments. These are different CSV files located in subdirectories like
   `experiment-01-cardinality-distortion`. See [below](#detailed-structure-of-the-raw-results) for a detailed list.
2. Plots and summaries that should (hopefully) confirm all core results of the paper. These are available as PDF files in the
   `eval/summaries` subdirectory. See [below](#detailed-structure-of-the-core-results-summaries) for a detailed list.
3. Plots for all core results that used individual queries for illustration (e.g., Figure 3 in the paper). These are placed
   in individual subdirectories (similar to the raw results), but in the `eval/` subdirectory (similar to the sumary PDFs).
   See [below](#detailed-structure-of-the-core-plots) for a detailed list.


# Appendix

## Pipeline Arguments

## Detailed Structure of the Core Results Summaries

## Detailed Structure of the Core Plots

## Detailed Structure of the Raw Results
