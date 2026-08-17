# Two-Area Capacity Market

This repository contains the data and Python code used to reproduce the numerical analyses presented in Chapters 4 and 5 of an MSc dissertation on capacity value in a GB–Ireland two-area power system.

## Repository Structure

* `data/` contains the input data used in the adequacy calculations. `InterconnectionData_Rescaled.txt` contains the hourly demand and wind series used to construct GB and Irish net demand; `GB_anonymised_conv.txt` and `I_conv.txt` contain conventional generation capacities and availabilities for GB and Ireland, respectively; `InterconnectionData_peak.txt` is retained as part of the original input data set and is checked when the data are loaded.
* `code/chapter4_reproduce.py` reproduces the Chapter 4 numerical analysis, including conventional-resource EFC calculations, controlled-background adequacy results, additional Irish wind capacity value, and whole-fleet Irish wind capacity value.
* `code/chapter5_reproduce.py` reproduces the Chapter 5 portfolio analysis, including resource-location comparisons, controlled adequacy-background cases, portfolio non-additivity calculations, and simplified-model diagnostics.
* `requirements.txt` lists the Python packages required to run the reproduction scripts.

## Installation

Clone or download the repository and install the required Python packages from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Reproducing Chapter 4

From the repository root, run:

```bash
python code/chapter4_reproduce.py
```

The script reads the four input files from `data/` by default and creates a `chapter4_outputs/` directory containing the reproduced numerical results, figures, and summary output.

A different data or output directory can be specified using:

```bash
python code/chapter4_reproduce.py --data-dir PATH --output-dir PATH
```

## Reproducing Chapter 5

From the repository root, run:

```bash
python code/chapter5_reproduce.py
```

By default, the script computes the Chapter 5 numerical results and generates the corresponding figures. Numerical outputs are written to `chapter5_outputs/` and figures to `chapter5_figures/`.

The script also supports separate computation, plotting, and verification modes:

```bash
python code/chapter5_reproduce.py --mode compute
python code/chapter5_reproduce.py --mode plot
python code/chapter5_reproduce.py --mode verify
```

Verification against locked reference outputs is optional and requires the corresponding reference-output directory. If no reference directory is present, the default run completes the calculation and plotting stages and skips verification.

## Reproducibility

The scripts implement the adequacy and capacity-value conventions used in the dissertation and are intended to reproduce the numerical results reported in Chapters 4 and 5 from the supplied input data.

All commands above assume that they are executed from the root directory of this repository.
