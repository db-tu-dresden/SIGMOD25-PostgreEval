# Supplementary Files for "An Elephant Under the Microscope" (SIGMOD 2025)

This repository contains the experiment scripts, evaluation notebooks and links to the datasets that are used in our
SIGMOD 2025 publication "An Elephant Under the Microscope: Analyzing the Interaction of Optimizer Components in PostgreSQL"
([DOI 10.1145/3709659](https://doi.org/10.1145/3709659)).


## Structure

On a high level, the repository is structured as follows:

| Folder | Description |
| ------ | ----------- |
| `datasets/` | Contains references to the actual datasets. All datasets are open access and archived in the [OPARA repository](https://opara.zih.tu-dresden.de/items/caf1add1-a309-4956-8291-9f3d9cb932dc). |
| `evaluation/` | Contains various Jupyter notebooks that generate the plots used throughout the paper as well as calculations that determine other key numbers. |
| `experiments/` | Contains the Python scripts that compute the datasets. These scripts can be used to reproduce our results. See [Reproducibility](#reproducibility) for details. |
| `pg_lab/` | Links to the *pg_lab* repository that is used to interact with a PostgreSQL instance in our paper. The submodule is checked out at the latest state that was used for publication.
| `postbound/` | Links to the *PostBOUND* respository that is used for optimizer hinting, plan analysis, etc. in our paper. The submodule is checked out at the latests state that was used for publication. |

Please note that both [pg_lab](https://github.com/rbergm/pg_lab) as well as [PostBOUND](https://github.com/rbergm/PostBOUND)
are frameworks under active development. If you are interested in using them for your own experiments, we kindly ask you to use
the latest versions available on Github. Both have been much improved since their original publication.


## Reproducibility

> [!note]
> We will add further information regarding reproducibility, as well as an end-to-end experiment pipeline in time for the
> SIGMOD ARI.


## Errata

### Section 4.2

In the original paper we mistakenly duplicated two numbers in the enumeration on page 13. Specifically, the last bullet point
mentioned that "_73 (simplified) and 62 queries (vanilla) produced the maximum number of jumps at an intermediate stage_".
These numbers are the same as on the previous bullet point. The correct numbers are 11 (simplified) and 12 queries (vanilla).

### Section 6

In the original paper we mentioned that Stack contains 200 queries which pass the GEQO threshold.
In fact, there is just a single context for which this is the case and which contains just 100 queries.
The error was caused by the data frame containing results for both experiment repetitions (once with GEQO enabled and once with
GEQO disabled).  As a consequence, all of these queries actually change their execution plan at least once while GEQO is
active (instead of just 50% as claimed in the paper).
This does not affect the 36 queries that change their execution plan even if GEQO is disabled.
