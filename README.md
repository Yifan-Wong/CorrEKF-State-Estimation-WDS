# CorrEKF-State-Estimation-WDS

This repository provides the code for the paper **"Correlation-aware State Estimation in Real Water Distribution Systems with Various Sources of Uncertainty"**.

## Quick Start

To make the main workflow easier to run, we provide a Google Colab version of the main notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yifan-Wong/CorrEKF-State-Estimation-WDS/blob/main/EKFs_Main_Results_colab_version.ipynb)

The Colab notebook automatically installs the required packages, clones this repository into the Colab runtime, and sets the working directory. Readers do not need to manually upload the local Python files or the `pipedream_solver/` folder.

If you prefer to run the notebooks locally, create the environment with one of the following commands:

```bash
conda env create -f environment.yml
```

or:

```bash
pip install -r requirements-local.txt
```

## File Overview

1. [EKFs_Main_Results_colab_version.ipynb](https://github.com/Yifan-Wong/CorrEKF-State-Estimation-WDS/blob/main/EKFs_Main_Results_colab_version.ipynb) contains the main code for the paper, including the comparison among the three EKF variants and the corresponding visualization results.

2. [EKFs_Main_Results.ipynb](https://github.com/Yifan-Wong/CorrEKF-State-Estimation-WDS/blob/main/EKFs_Main_Results.ipynb) contains the same main workflow as the Colab notebook, but in a standard Jupyter Notebook format. It requires a local environment setup before running.

3. [Model_vs_truth.ipynb](https://github.com/Yifan-Wong/CorrEKF-State-Estimation-WDS/blob/main/Model_vs_truth.ipynb) compares the imperfect hydraulic model with the ground-truth hydraulic dynamics.

4. [Multiple_uncertainty_sources.ipynb](https://github.com/Yifan-Wong/CorrEKF-State-Estimation-WDS/blob/main/Multiple_uncertainty_sources.ipynb) compares the effects of different uncertainty sources on the hydraulic dynamics.

5. [Pipedream validation.ipynb](https://github.com/Yifan-Wong/CorrEKF-State-Estimation-WDS/blob/main/Pipedream%20validation.ipynb) validates the state-space model used in this paper against EPANET.

6. The remaining Python files and the `pipedream_solver/` folder provide helper functions and solver routines used by the notebooks.

## Citation

Citation information will be added after publication.

## License

MIT
