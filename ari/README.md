# SIGMOD ARI Submission

This directory contains all artifacts that are required to reproduce the results of our paper as part of the SIGMOD ARI 2025.
Detailed instructions for reviewers follow below.
Please read this document "cover to cover" to understand certain design decisions.
This readme serves as a handbook for ARI reviewers.


# Reproducibility Strategy

Reproducing all data from the original paper would require executing more than half a million queries across three benchmarks
and two hardware platforms.
In total, it took us several months of raw compute time to obtain the original measurements.

We argue that a complete reproduction is infeasible for the community.
At the same time, we also think that a complete reproduction is not necessary to confirm the core findings of the paper.
Therefore, we limit the scope of our reproducibility efforts to a subset of the original experiments.
We describe and justify our approach in the remainder of this section.
Please note that the reproducibility pipeline is capable of reproducing all the results, but the default ARI
configuration (as detailed in the [Setup](#setup)) just focuses on the core contributions.
Most importantly, by default we only reproduce results on the Join Order Benchmark.
The Stats benchmark can be activated if desired, but this will increase the runtime of the entire pipeline by a significant
amount.

The pipeline generates plots for all original figures from the paper.
However, some figures contained plots for different hardware platforms or different workloads.
We do not re-create these figures on a 1:1 basis.
Most importantly, we only re-create figures on a single benchmark (JOB), instead of all benchmarks that have been used in the
paper (JOB, Stats).
The different benchmarks were used to aid in generalizing the results.
We argue that this is still possible if the results have be reproduced on an entirely different hardware platform.
Hence, the core results are not affected by this decision.

## Core Results

We consider the following results from our paper to be "core results" which are (hopefully) reproduced in the ARI pipeline.
Experiments and evaluations that are not covered by the core results mainly serve illustrative purposes to understand the core
results better or are ablations of the core experiments.

> [!note]
> For ARI reviewers: If you feel like another result from our paper should be considered a core result, we are happy to add an
> evaluation for it.

**Section 4 (Cardinality Distortion):** this section analyzed when improved cardinality estimates result in new query execution
plans and how the query runtimes changed.
It produced two important observations:

1. plan changes happen without a clear pattern on a per-query basis **(C4.1)**
2. novel execution plans impact the query runtime arbitrarily, i.e. novel plans might improve or degrade performance without a
   predictable pattern **(C4.2)**

As part of an ablation we also showed that neither a simplified cost model nor a restricted enumerator search space improved
this situation substantially **(C4.3)**.

**Section 5 (Analyzing Candidate Plans):** this section analyzed the correlation between estimated plan cost and actual plan
runtime and searched for patterns of good candidate plans.
It produced the following important observations:

1. the correlation between cost estimate and plan runtime is still rather low **(C5.1)**
2. many good candidate plans of a query have the same base join and many queries have obvious base join candidates **(C5.2)**
3. the effect of good base joins is not negated even if the optimizer faces errors in the later optimization process **(C5.3)**

**Section 6 (Non-deterministic Components):** analyzed how non-deterministic parts of the optimization process, specifically
randomized search and sampled statistics catalogs influence the selected query plans and their runtimes.
Core results include:

1. both the randomized GEQO optimizer and different statistics samples cause different execution plans to be selected **(C6.1)**
2. the different plans also impact the execution time by a significant amount **(C6.2)**

We do not consider the data shift experiment performed for Section 6 to be a core result and don't evaluate the raw data in the
pipeline.

**Section 7 (Beyond Textbook Optimizers):** studied whether other optimizer architectures can be more resilient towards
estimation errors.
This section produced the following key insights:

1. gradual increasing cardinality estimation errors lead to jumps in the query execution time **(C7.1)**
2. jumps happen more frequently when the textbook optimizer underestimates the true cardinalities **(C7.2)**
3. robust optimizers can compensate for these errors much better (i.e. they show fewer jumps) **(C7.3)**

## Reproduced Results

In the ARI pipeline, we try to reproduce the core results as follows:
For each of the experiments, we re-run the entire data generation step.
Afterwards, we perform an evaluation specific to the current core result.
This includes re-generating the plots for each query (if the paper contained plots on a per-query basis).
If the paper contained an aggregated plot for a core result, we also generate this plot.
Lastly, we determine _replacement plots_ that are shown in the final PDF.
This last step is necessary, because the original per-query plots were manually picked to illustrate specific phenomena.
For example, Figure 3(b) in the paper showed an example of a query with an unstable plan selection behavior.
However, the individual queries that are affected differ between systems (as discussed in the paper).
Therefore, a 1:1 reproduction of the plots does not make sense.
Instead, we try to model the manual selection process with heuristics and present the most likely replacement plots in the
final PDF.
If the heuristic selects an inappropriate plot, please look at the other plots for potential alternatives.
The evaluation also includes re-generating all aggregated numbers that correspond to the core results.


# System Requirements

To run the ARI pipeline, please use a machine with the following hardware specifications:

- At least 128GB of RAM
- A recent server-class Intel CPU (i.e. Cascade Lake or newer). We did not test on other CPU vendors.
- At least 256GB of available SSD storage. Make sure to mount the Docker volume on the SSD.

The server should run a recent Linux system with Docker v28.3.0 (other installations should work, we simply could not test
them).

Running the entire ARI pipeline took us about 16 days on a machine with an Intel Xeon Gold 6240R CPU (24 cores @ 2.4GHz) and
755GB of RAM.
If you want to reproduce results on the Stack benchmark, you should have at least 500GB of available SSD storage.

# Setup

The entire pipeline is implemented in a Docker environment and does not require any user intervention once started.
To create the Docker container, please run the following commands in the `ari` working directory
(`💻 $` refers to a shell on your host machine whereas `🐳 $` refers to a shell within the Docker container ):

> [!caution]
> Replace the _SHM_SIZE_ placeholder with the desired amount of shared memory. We heavily recommend using at least 128GB, or
> 1/4th of the available RAM. This is because shared memory determines the size of the shared page buffer used by Postgres
> (see [System Requirements](#system-requirements)). If the databases do not fit into RAM, performance measurements might
> become unreliable (see Section 3.3 of the paper).

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
   `experiment-01-cardinality-distortion`.
2. Plots and summaries that should (hopefully) confirm all core results of the paper. These are available as PDF files in the
   `eval/summaries` subdirectory. **Ideally, the PDFs are the only results that reviewers need to take a look at.**
3. Plots for all core results that used individual queries for illustration (e.g., Figure 3 in the paper). These are placed
   in individual subdirectories (similar to the raw results), but in the `eval/` subdirectory (similar to the sumary PDFs).

Below is a detailed list of all relevant file locations:

| Core result | Raw experiment data | Individual plots | Final PDF |
| ----------- | ------------------- | ---------------- | --------- |
| **C4.1** | `results/experiment-01-cardinality-distortion` | `results/eval/experiment-01-cardinality-distortion` | `results/eval/summaries/01-Card-Distortion.pdf` |
| **C4.2** | `results/experiment-01-cardinality-distortion` | `results/eval/experiment-01-cardinality-distortion` | `results/eval/summaries/01-Card-Distortion.pdf` |
| **C4.3** | `results/experiment-02-distortion-ablation` | `results/eval/experiment-02-distortion-ablation` | `results/eval/summaries/02-Distortion-Ablation.pdf` |
| **C5.1** | `results/experiment-03-plan-space-analysis` | `results/eval/experiment-03-plan-space` | `results/eval/summaries/03-Plan-Space.pdf` |
| **C5.2** | `results/experiment-03-plan-space-analysis` | `results/eval/experiment-03-plan-space` | `results/eval/summaries/03-Plan-Space.pdf` |
| **C5.3** | `results/experiment-03-plan-space-analysis` | `results/eval/experiment-03-plan-space` | `results/eval/summaries/03-Plan-Space.pdf` |
| **C6.1** | `results/experiment-05-analyze-stability` | `results/eval/experiment-05-analyze-stability` | `results/eval/summaries/05-Analyze-Stability.pdf` |
| **C6.2** | `results/experiment-05-analyze-stability` | `results/eval/experiment-05-analyze-stability` | `results/eval/summaries/05-Analyze-Stability.pdf` |
| **C7.1** | `results/experiment-07-beyond-textbook` | `results/eval/experiment-07-beyond-textbook` | `results/eval/summaries/07-Beyond-Textbook.pdf` |

All paths are relative to the docker volume root. Results from the data shift in Section 6 are stored in `results/experiment-06-analyze-stability-shift`, but do not contribute to the core results.


# Appendix

## Pipeline Arguments

The `xctl.py` utility accepts the following arguments:

| Argument | Allowed values | Description | ARI setting |
| -------- | -------------- | ----------- | ----------- |
| `--mode` | `full`, `experiments`, `eval` | Controls whether the pipeline should run only the experiments (`experiments`), only evaluate results from a previous run (`eval`), or do both (`full`). | `full` |
| `--benchmark` | `all`, `job`, `stats`, `stack` | The benchmark to execute in `experiments` mode. Note that Stack is only used for experiment 4. | `job` |
| `<experiments>` | `all`, `0-base`, `1-card-distortion`, `2-distortion-ablation`, `3-plan-space`, `4-analyze-stability`, `5-analyze-shift`, `6-beyond-textbook` | The experiments to run and evaluate. Multiple experiments can be listed. And experiment 0 should always be run first to generate important base data (e.g., vanilla benchmark runtimes on the current system). This data is re-used by other experiments! | `all` |
