# Two-Area Capacity Market

This repository contains the data and Python code used to reproduce the numerical analyses presented in Chapters 4 and 5 of an MSc dissertation on capacity value in a GB–Ireland two-area power system.

The analysis evaluates the contribution of resources located in Ireland to Great Britain (GB) generation adequacy, with particular emphasis on Equivalent Firm Capacity (EFC), interconnector capacity, background adequacy conditions, wind generation, and portfolio interactions.

## Repository Structure

```text
Two-Area-Capacity-Market/
│
├── code/
│   ├── chapter4_reproduce.py
│   └── chapter5_reproduce.py
│
├── data/
│   ├── GB_anonymised_conv.txt
│   ├── I_conv.txt
│   ├── InterconnectionData_Rescaled.txt
│   └── InterconnectionData_peak.txt
│
├── requirements.txt
└── README.md
```

### `data/`

The `data/` directory contains the input data used in the GB–Ireland adequacy model.

* `GB_anonymised_conv.txt` contains the conventional generation data for Great Britain.
* `I_conv.txt` contains the conventional generation data for Ireland.
* `InterconnectionData_Rescaled.txt` contains the demand and wind-generation time series used in the main adequacy calculations.
* `InterconnectionData_peak.txt` is retained as part of the input dataset used in the analysis.

### `code/chapter4_reproduce.py`

This script reproduces the numerical analysis presented in Chapter 4.

The Chapter 4 analysis focuses on the capacity value of individual resources located in Ireland with respect to GB adequacy, including:

* EFC calculations for conventional resources;
* sensitivity to interconnector capacity;
* controlled GB background adequacy conditions;
* the capacity value of additional Irish wind generation; and
* the capacity value of the existing Irish wind fleet.

### `code/chapter5_reproduce.py`

This script reproduces the numerical analysis presented in Chapter 5.

The Chapter 5 analysis focuses on portfolio effects and the mechanisms underlying the capacity-value results, including:

* portfolio EFC calculations;
* comparison of resource-location configurations;
* portfolio non-additivity and subadditivity;
* sensitivity to interconnector capacity and GB background adequacy;
* limiting-case behaviour; and
* simplified numerical diagnostics used to investigate the drivers of the observed results.

## Requirements

The reproduction scripts require Python and the packages listed in `requirements.txt`.

The main external Python dependencies are:

* NumPy
* pandas
* Matplotlib
* Numba

From the root directory of the repository, install the required packages using:

```bash
python -m pip install -r requirements.txt
```

## Reproducing Chapter 4

From the root directory of the repository, run:

```bash
python code/chapter4_reproduce.py
```

The script reads the required input data from `data/` and writes the reproduced numerical results, figures, and summary output to:

```text
chapter4_outputs/
```

Alternative data and output directories can be specified using:

```bash
python code/chapter4_reproduce.py --data-dir PATH --output-dir PATH
```

## Reproducing Chapter 5

From the root directory of the repository, run:

```bash
python code/chapter5_reproduce.py
```

The script reproduces the Chapter 5 numerical results and generates the corresponding figures.

By default, numerical outputs are written to:

```text
chapter5_outputs/
```

and figures are written to:

```text
chapter5_figures/
```

Alternative data, output, and figure directories can be specified using:

```bash
python code/chapter5_reproduce.py --data-dir PATH --output-dir PATH --figure-dir PATH
```

The script also contains optional computation, plotting, and verification functionality for development and reproducibility checks. The default run performs the numerical calculations and figure generation required to reproduce the Chapter 5 results.

## Reproducibility

The scripts implement the adequacy and capacity-value methodology used in the dissertation and are provided to support reproduction of the numerical results reported in Chapters 4 and 5.

The principal adequacy metric is Great Britain Loss of Load Expectation (LOLE), while resource capacity value is evaluated using Equivalent Firm Capacity (EFC). The EFC comparator is firm capacity located in Great Britain.

All commands in this README assume that they are executed from the root directory of the repository.

## Dissertation

This repository accompanies an MSc dissertation undertaken at the University of Edinburgh.

The repository is intended to provide the data and computational material required to reproduce the principal numerical analyses presented in Chapters 4 and 5.
