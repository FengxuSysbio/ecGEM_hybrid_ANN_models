# Scripts

## benchmark_model_fit.py

Fits an ANN model for `qP_mmol_L_h` using benchmark model data from the first
sheet of `Supplementary Data 3.xlsx`.

The script performs:

- batch and sample-position feature engineering
- ANN hyperparameter comparison
- all-data fitting evaluation
- random train/test split evaluation
- model, table, and figure export

## Hybird_model_fit.py

Fits an ANN model for `r_0013` using integrated benchmark and flux data.

The script performs:

- row-wise integration of sheet 1 and sheet 2
- removal of `qP_mmol_L_h` from benchmark inputs
- removal of `r_0013` from flux inputs
- feature selection with `SelectKBest`
- ANN hyperparameter comparison
- model, table, and figure export

Note: the filename keeps the original spelling `Hybird_model_fit.py` for
consistency with the manuscript working files.
