from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

RANDOM_STATE = 42
OUTPUT_DIR = Path("ann_hybrid_model_results")
TARGET = "r_0013"


def find_workbook():
    preferred = Path.cwd() / "Supplementary Data 3.xlsx"
    if preferred.exists():
        return preferred
    candidates = sorted(Path.cwd().glob("*.xlsx"), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        raise FileNotFoundError("No Excel workbook found.")
    return candidates[0]


def prepare_integrated_dataset(base, flux):
    n_rows = min(len(base), len(flux))

    base_part = base.iloc[:n_rows].reset_index(drop=True).drop(columns=["qP_mmol_L_h"], errors="ignore")
    flux_part = flux.iloc[:n_rows].reset_index(drop=True)

    base_part.insert(0, "base_sample_no", np.arange(1, n_rows + 1))
    base_part.insert(1, "base_total_samples", len(base))

    flux_features = flux_part.drop(columns=[TARGET], errors="ignore").copy()
    flux_features.insert(0, "flux_sample_no", np.arange(1, n_rows + 1))
    flux_features.insert(1, "flux_total_samples", len(flux))
    flux_features = flux_features.add_prefix("flux__")

    X = pd.concat([base_part, flux_features], axis=1).select_dtypes(include=[np.number])
    y = flux_part[TARGET]
    return X, y


def make_model(hidden_layer_sizes, alpha, learning_rate_init, selected_features):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if selected_features is not None:
        steps.append(("select", SelectKBest(score_func=f_regression, k=selected_features)))

    steps.append(
        (
            "ann",
            MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation="relu",
                solver="adam",
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                max_iter=6000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=120,
                random_state=RANDOM_STATE,
            ),
        )
    )
    return TransformedTargetRegressor(
        regressor=Pipeline(steps),
        transformer=StandardScaler(),
    )


def metrics(y_true, y_pred, prefix):
    return {
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
        f"{prefix}_rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
    }


def plot_fit(predictions, output_path):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    color_map = {"train": "#2878b5", "test": "#c82423"}
    for split, group in predictions.groupby("split"):
        ax.scatter(
            group["actual"],
            group["predicted"],
            s=20,
            alpha=0.75,
            label=split,
            color=color_map.get(split, "#333333"),
        )
    low = min(predictions["actual"].min(), predictions["predicted"].min())
    high = max(predictions["actual"].max(), predictions["predicted"].max())
    ax.plot([low, high], [low, high], color="#333333", lw=1)
    ax.set_xlabel("Actual r_0013")
    ax.set_ylabel("Predicted r_0013")
    ax.set_title("ANN prediction result for r_0013")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    workbook = find_workbook()
    base = pd.read_excel(workbook, sheet_name=0)
    flux = pd.read_excel(workbook, sheet_name=1)
    X, y = prepare_integrated_dataset(base, flux)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )

    max_k = min(X_train.shape[1], X_train.shape[0] - 1)
    hyperparameter_grid = [
        {"hidden_layer_sizes": (64, 32), "alpha": 1e-3, "learning_rate_init": 1e-3, "selected_features": min(80, max_k)},
        {"hidden_layer_sizes": (128, 64), "alpha": 1e-4, "learning_rate_init": 1e-3, "selected_features": min(120, max_k)},
        {"hidden_layer_sizes": (128, 64, 32), "alpha": 1e-4, "learning_rate_init": 5e-4, "selected_features": min(160, max_k)},
        {"hidden_layer_sizes": (256, 128), "alpha": 1e-5, "learning_rate_init": 5e-4, "selected_features": min(200, max_k)},
        {"hidden_layer_sizes": (256, 128, 64), "alpha": 1e-5, "learning_rate_init": 1e-3, "selected_features": min(240, max_k)},
        {"hidden_layer_sizes": (128, 64), "alpha": 1e-3, "learning_rate_init": 1e-3, "selected_features": None},
    ]

    search_rows = []
    best = None
    for config_id, params in enumerate(hyperparameter_grid, start=1):
        model = make_model(**params)
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        row = {
            "config_id": config_id,
            **params,
            **metrics(y_train, train_pred, "train"),
            **metrics(y_test, test_pred, "test"),
            "n_iter": model.regressor_.named_steps["ann"].n_iter_,
        }
        search_rows.append(row)
        if best is None or row["test_r2"] > best["row"]["test_r2"]:
            best = {"row": row, "model": model, "params": params}

    final_model = best["model"]
    all_pred = final_model.predict(X)
    train_pred = final_model.predict(X_train)
    test_pred = final_model.predict(X_test)

    final_metrics = {
        "target": TARGET,
        "n_rows_integrated": len(X),
        "base_total_samples": len(base),
        "flux_total_samples": len(flux),
        "n_features": X.shape[1],
        **best["params"],
        **metrics(y_train, train_pred, "train"),
        **metrics(y_test, test_pred, "test"),
        **metrics(y, all_pred, "all_data"),
    }

    predictions = pd.DataFrame(
        {
            "row_index": X.index,
            "base_sample_no": X["base_sample_no"],
            "flux_sample_no": X["flux__flux_sample_no"],
            "actual": y,
            "predicted": all_pred,
            "residual": y - all_pred,
            "split": "train",
        }
    )
    predictions.loc[predictions["row_index"].isin(X_test.index), "split"] = "test"

    pd.DataFrame(search_rows).sort_values("test_r2", ascending=False).to_excel(
        OUTPUT_DIR / "r0013_hyperparameter_search.xlsx",
        index=False,
    )
    pd.DataFrame([final_metrics]).to_excel(OUTPUT_DIR / "r0013_metrics.xlsx", index=False)
    predictions.to_excel(OUTPUT_DIR / "r0013_predictions.xlsx", index=False)
    joblib.dump(final_model, OUTPUT_DIR / "r0013_ann_model.joblib")
    plot_fit(predictions, OUTPUT_DIR / "r0013_actual_vs_predicted.png")

    print("r_0013 ANN prediction complete")
    print(pd.DataFrame([final_metrics]).to_string(index=False))
    print(f"Outputs: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
