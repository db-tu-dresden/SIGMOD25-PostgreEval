# Experiments

This directory contains the scripts used to obtain the raw data for our analysis.

All scripts are assumed to be executed from the root directory of the repo and as Python modules (e.g.
`python3 -m experiments.experiment-00-native-runtimes`). Additionally, all scripts need to be able to import the PostBOUND
sources directly from the Python Path. Lastly, the experiments need connections to the database, as specified in PostBOUND's
`postgres` module.

> [!note]
> Currently, the scripts cannot be used in a 1:1 manner to reproduce the actual result files, since we prepared them slightly
> Most significantly, we added columns describing the current database server, or we combined different experiment stages into
> a single file (e.g. for Section 4.2 or Section 7).
> We will automate the entire experiment pipeline in time for the [SIGMOD ARI](https://reproducibility.sigmod.org/).
> The same applies to some of the leftover hard-coded locations of input files or output artifacts.


## Section 4.1

The input cardinalities for the different workloads can be computed using `experiment-00-intermediate-cardinalities.py`.
Based on these cardinalities, `experiment-01-cardinality-distortion.py` obtains the actual results.


## Section 4.2

Results are computed using the `experiment-02-advanced-distortion.sh`, which essentially repeats the experiments of
[Section 4.1](#section-41) with different input data to the distortion script.


## Section 5

In order to run the experiments, the true cardinalities of all intermediates are required again (see
[Section 4.1](#section-41)). Additionally, the query runtimes of the native Postgres optimizer are used to set dynamic timeouts
for all queries. These can be computed via `experiment-00-native-runtimes.py`.

Once all required input data has been computed, the actual experiments can be started using
`experiment-03-plan-space-analysis.py`. For reference, we also provide the legacy implementation of this experiment that
performs a pure uniform sampling. This script was used to check, whether the opionated sampling introduces its own bias.


## Section 5.2

To judge the impact of optimal base join selection, the important base joins have to be determined using the analysis notebook
(but we are going to refactor the implementation for the ARI). Once the dataset has been computed, the
`experiment-04-base-join-impact.py` script performs the adapted experiment of Section 5.


## Section 6

The main experiment data can be computed using `experiment-05-analyze-stability.py`. For our purposes, this script has to be
executed twice: once with GEQO enabled and a second time with GEQO disabled.

In addition, `experiment-06-analyze-stability-datashift.sh` can be used to repeat the same experiment on different data sizes
for JOB. Notice that this script requires the data shift implementation from [Section 7](#section-7)


## Section 7

The data shift can be executed using `experiment-07-optimizer-architectures.py`. This script requires a lot of PostBOUND's
utilities to manage and configure PostgreSQL servers.
