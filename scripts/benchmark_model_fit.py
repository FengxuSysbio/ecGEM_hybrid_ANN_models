from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")

RANDOM_STATE = 42
OUTPUT_DIR = Path("ann_benchmark_model_results")
TARGET = "qP_mmol_L_h"


def find_workbook():
    preferred = Path.cwd() / "Supplementary Data 3.xlsx"
    if preferred.exists():
        return preferred
    candidates = sorted(Path.cwd().glob("*.xlsx"), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        raise FileNotFoundError("No Excel workbook found.")
    return candidates[0]


def add_qp_features(base):
    data = base.copy()
    data["batch_id"] = (data["Time_h"].diff().fillna(1) <= 0).cumsum() + 1
    data["sample_no"] = np.arange(1, len(data) + 1)
    data["batch_sample_no"] = data.groupby("batch_id").cumcount() + 1
    data["batch_n_samples"] = data.groupby("batch_id")["Time_h"].transform("size")
    data["batch_time_fraction"] = data["batch_sample_no"] / data["batch_n_samples"]

    dynamic_columns = [
        "Time_h",
        "DO_%",
        "pH",
        "Tail_CO2_%",
        "OUR_online_mmol/L/h",
        "CER_online_mmol/L/h",
        "RQ_online",
        "Glucose_g_L",
        "Biomass_%",
        "NH4_g_L",
    ]
    for column in dynamic_columns:
        data[f"{column}__sq"] = data[column] ** 2
        data[f"{column}__lag1"] = data.groupby("batch_id")[column].shift(1)
        data[f"{column}__diff1"] = data.groupby("batch_id")[column].diff(1)
        data[f"{column}__roll3_mean"] = (
            data.groupby("batch_id")[column]
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return data


def build_preprocessor(X):
    categorical_features = ["batch_id", "sample_no", "batch_sample_no"]
    numeric_features = [
        column
        for column in X.select_dtypes(include=[np.number]).columns
        if column not in categorical_features
    ]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )


def make_model(X, hidden_layer_sizes, activation, solver, alpha):
    ann_parameters = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": activation,
        "solver": solver,
        "alpha": alpha,
        "max_iter": 5000,
        "random_state": RANDOM_STATE,
    }
    if solver == "adam":
        ann_parameters.update(
            {
                "learning_rate_init": 1e-3,
                "early_stopping": False,
                "n_iter_no_change": 300,
            }
        )

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X)),
            ("ann", MLPRegressor(**ann_parameters)),
        ]
    )
    return TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())


def metrics(y_true, y_pred, prefix):
    return {
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
        f"{prefix}_rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
    }


def plot_fit(predictions, output_path):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.scatter(predictions["actual"], predictions["predicted"], s=18, alpha=0.75, color="#2878b5")
    low = min(predictions["actual"].min(), predictions["predicted"].min())
    high = max(predictions["actual"].max(), predictions["predicted"].max())
    ax.plot([low, high], [low, high], color="#333333", lw=1)
    ax.set_xlabel("Actual qP (mmol/L/h)")
    ax.set_ylabel("Predicted qP (mmol/L/h)")
    ax.set_title("ANN fitting result for qP")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    workbook = find_workbook()
    base = pd.read_excel(workbook, sheet_name=0)
    data = add_qp_features(base)
    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    hyperparameter_grid = [
        {"hidden_layer_sizes": (128, 64), "activation": "relu", "solver": "lbfgs", "alpha": 1e-6},
        {"hidden_layer_sizes": (256, 128), "activation": "relu", "solver": "lbfgs", "alpha": 1e-7},
        {"hidden_layer_sizes": (256, 128), "activation": "tanh", "solver": "lbfgs", "alpha": 1e-7},
        {"hidden_layer_sizes": (256, 128, 64), "activation": "relu", "solver": "adam", "alpha": 1e-6},
    ]

    search_rows = []
    fitted_models = []
    for config_id, params in enumerate(hyperparameter_grid, start=1):
        model = make_model(X, **params)
        model.fit(X, y)
        fitted = model.predict(X)
        row = {
            "config_id": config_id,
            **params,
            **metrics(y, fitted, "fit"),
            "n_iter": model.regressor_.named_steps["ann"].n_iter_,
        }
        search_rows.append(row)
        fitted_models.append((row["fit_r2"], model, params, fitted))

    search_table = pd.DataFrame(search_rows).sort_values("fit_r2", ascending=False)
    best_r2, best_model, best_params, fitted = fitted_models[
        int(search_table.iloc[0]["config_id"]) - 1
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )
    split_model = make_model(X_train, **best_params)
    split_model.fit(X_train, y_train)
    train_pred = split_model.predict(X_train)
    test_pred = split_model.predict(X_test)

    final_metrics = {
        "target": TARGET,
        "n_samples": len(X),
        "n_features_after_engineering": X.shape[1],
        **best_params,
        **metrics(y, fitted, "all_data_fit"),
        **metrics(y_train, train_pred, "train_split"),
        **metrics(y_test, test_pred, "test_split"),
    }

    predictions = data[["sample_no", "batch_id", "batch_sample_no", "Time_h"]].copy()
    predictions["actual"] = y
    predictions["predicted"] = fitted
    predictions["residual"] = predictions["actual"] - predictions["predicted"]

    search_table.to_excel(OUTPUT_DIR / "qP_hyperparameter_search.xlsx", index=False)
    pd.DataFrame([final_metrics]).to_excel(OUTPUT_DIR / "qP_metrics.xlsx", index=False)
    predictions.to_excel(OUTPUT_DIR / "qP_predictions.xlsx", index=False)
    joblib.dump(best_model, OUTPUT_DIR / "qP_ann_model.joblib")
    plot_fit(predictions, OUTPUT_DIR / "qP_actual_vs_predicted.png")

    print("qP ANN fitting complete")
    print(pd.DataFrame([final_metrics]).to_string(index=False))
    print(f"Outputs: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
