# ANN models for benchmark and hybrid ecGEM-based prediction

This repository contains two Python scripts for fitting artificial neural network
(ANN) regression models using data from `Supplementary Data 3.xlsx`.

## Contents

```text
ecGEM_hybrid_ANN_models/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── results/
│   └── README.md
└── scripts/
    ├── benchmark_model_fit.py
    └── Hybird_model_fit.py
```

## Models

### 1. Benchmark ANN model

Script:

```bash
python scripts/benchmark_model_fit.py
```

Target variable:

```text
qP_mmol_L_h
```

Input data:

- Sheet 1 of `Supplementary Data 3.xlsx`
- Benchmark fermentation process variables
- Engineered temporal and batch-position features

Output directory:

```text
ann_benchmark_model_results/
```

Main outputs:

- `qP_hyperparameter_search.xlsx`
- `qP_metrics.xlsx`
- `qP_predictions.xlsx`
- `qP_ann_model.joblib`
- `qP_actual_vs_predicted.png`

### 2. Hybrid ANN model

Script:

```bash
python scripts/Hybird_model_fit.py
```

Target variable:

```text
r_0013
```

Input data:

- Sheet 1: benchmark model data, excluding `qP_mmol_L_h`
- Sheet 2: eciFX1172 core metabolic reactions and partial flux data
- Sample-number and total-sample-count features from both sheets

Output directory:

```text
ann_hybrid_model_results/
```

Main outputs:

- `r0013_hyperparameter_search.xlsx`
- `r0013_metrics.xlsx`
- `r0013_predictions.xlsx`
- `r0013_ann_model.joblib`
- `r0013_actual_vs_predicted.png`

## Data setup

Place the Excel file in the repository root:

```text
Supplementary Data 3.xlsx
```

The scripts first look for this exact filename in the current working directory.
If it is not found, they fall back to the largest `.xlsx` file in the current
directory.

Recommended execution from the repository root:

```bash
python scripts/benchmark_model_fit.py
python scripts/Hybird_model_fit.py
```

## Environment

Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Reproducibility

Both scripts use:

```text
RANDOM_STATE = 42
```

The ANN models are implemented with `scikit-learn`'s `MLPRegressor`. Outputs are
written as Excel tables, fitted model files, and actual-vs-predicted figures.

## Notes

- The benchmark model reports both all-data fitting performance and random
  train/test split performance.
- The hybrid model selects the best ANN configuration using test-set R2 from a
  small hyperparameter grid.
- Large input data files and generated model/result files are ignored by Git by
  default. Upload data separately if required by journal or repository policy.
