# Two-Area Capacity Market

This repository contains the data, Python code, and selected final results used to support the numerical analyses presented in Chapters 4 and 5 of an MSc dissertation on capacity value in a GB–Ireland two-area power system.

The analysis considers how resources located in Ireland contribute to Great Britain (GB) generation adequacy, with particular emphasis on Equivalent Firm Capacity (EFC), interconnector capacity, background adequacy conditions, wind generation, and portfolio interactions.

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
├── results/
│   ├── chapter4/
│   │   ├── ch4_additional_wind_efc.png
│   │   ├── ch4_controlled_background_efc.png
│   │   ├── ch4_conventional_location_efc.png
│   │   ├── ch4_whole_wind_efc.png
│   │   └── ch4_wind_accessible_fraction.png
│   │
│   └── chapter5/
│       ├── ch5_efc_decomposition.png
│       ├── ch5_ireland_ireland_backgrounds.png
│       ├── ch5_location_controls.png
│       └── ch5_overlap_complementarity.png
│
├── requirements.txt
└── README.md
```

## Data

The `data/` directory contains the input data used in the GB–Ireland adequacy calculations.

* `GB_anonymised_conv.txt` contains conventional generation data for Great Britain.
* `I_conv.txt` contains conventional generation data for Ireland.
* `InterconnectionData_Rescaled.txt` contains the demand and wind-generation time series used in the main adequacy calculations.
* `InterconnectionData_peak.txt` is retained as part of the input dataset used in the analysis.

## Code

### Chapter 4

`code/chapter4_reproduce.py` reproduces the numerical analysis presented in Chapter 4.

The analysis includes:

* EFC calculations for conventional resources;
* sensitivity to interconnector capacity;
* controlled GB background adequacy conditions;
* capacity value of additional Irish wind generation; and
* capacity value of the existing Irish wind fleet.

### Chapter 5

`code/chapter5_reproduce.py` reproduces the numerical analysis presented in Chapter 5.

The analysis includes:

* portfolio EFC calculations;
* comparison of different resource-location configurations;
* portfolio non-additivity and subadditivity;
* sensitivity to interconnector capacity and GB background adequacy;
* limiting-case behaviour; and
* simplified numerical diagnostics used to investigate the mechanisms underlying the observed portfolio effects.

## Results

The `results/` directory contains the final figures reported in Chapters 4 and 5 of the dissertation.

The Chapter 4 figures cover the capacity value of conventional resources and Irish wind generation under different interconnector and adequacy conditions.

The Chapter 5 figures cover portfolio interactions, resource-location comparisons, EFC decomposition, and the diagnostics used to investigate portfolio non-additivity.

These files are provided as archived final results corresponding to the submitted dissertation. Running the reproduction scripts generates new numerical outputs and figures from the supplied input data, which can be compared with the archived results in this directory.

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

The principal adequacy metric is Great Britain Loss of Load Expectation (LOLE), while resource capacity value is evaluated using Equivalent Firm Capacity (EFC). The EFC comparator is perfectly reliable firm capacity located in Great Britain.

All commands in this README assume that they are executed from the root directory of the repository.
