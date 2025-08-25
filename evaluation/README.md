# Evaluation Notebooks

These notebooks are used to generate the plots and summary statistics that have been used in the actual paper.
In contrast to the notebooks stored in the `ari/` subdirectory, the notebooks here are tailored to the full (static) data sets
that have been used in the paper.
Therefore, the evaluation code needs to consider a couple of things that are not important for ARI, such as evaluation on
different servers.
That is why there is some duplication in the notebooks.

Each notebook correponds to a specific section of the paper, as indicated by its file name.
In order to run the notebooks, make sure to setup PostBOUND first and adjust the `workload_base_dir` according to your setup.
Details are in the PostBOUND git submodule.
But to be honest, it is probably easier to use the environment that is automatically created as part of the ARI pipeline:
setup the Docker container as outlined in the ARI README. Afterwards, login to the container and proceed as normal (i.e. boot
a Jupyter server and load the notebooks that you are interested in).
Please note that the Jupyter server has to be started in this directory in order for all relative paths to work correctly.
