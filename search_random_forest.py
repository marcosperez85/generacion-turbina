import time
import warnings

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_validate, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

import json
from turbines_common import (DATASET_PATH, PARAMS_PATH, RESULTS_PATH, FEATURES, TARGET,
                             RANDOM_STATE, N_SPLITS, load_data, split_temporal)


def build_models():
    models = {
        "Baseline (mediana)": Pipeline([("model", DummyRegressor(strategy="median"))]),
        "Random Forest": Pipeline([("model", RandomForestRegressor(
            n_estimators=1000, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("model", GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            random_state=RANDOM_STATE))]),
        "XGBoost": Pipeline([("model", XGBRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            objective="reg:squarederror", random_state=RANDOM_STATE,
            n_jobs=-1, verbosity=0))]),
    }
    scaled = {
        "Regresión lineal": LinearRegression(),
        "SVR (RBF)": SVR(kernel="rbf", C=100, epsilon=0.1),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        "Red neuronal (MLP)": MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True,
            random_state=RANDOM_STATE),
    }
    models.update({name: Pipeline([("scaler", StandardScaler()), ("model", model)])
                   for name, model in scaled.items()})
    return models

def tune_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    param_distributions = {
        "n_estimators": [600, 800, 1000, 1200],
        "max_depth": [15, 20, 25, None],
        "min_samples_split": [2, 4, 6, 10],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": ["sqrt", 0.7, 0.85, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=40,
        scoring="neg_mean_absolute_percentage_error",
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=2,
    )

    search.fit(X_train, y_train)

    print("\nMejores hiperparámetros encontrados:")
    print(search.best_params_)

    print(
        f"Mejor MAPE de validación temporal: "
        f"{-search.best_score_ * 100:.3f} %"
    )

    return search.best_estimator_

def compare_models(models, X_train, y_train):
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "mape": "neg_mean_absolute_percentage_error",
        "r2": "r2",
    }

    rows = []
    print(f"Comparando modelos con TimeSeriesSplit ({N_SPLITS} particiones)...")
    for name, model in models.items():
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            scores = cross_validate(model, X_train, y_train,
                                    cv=TimeSeriesSplit(n_splits=N_SPLITS),
                                    scoring=scoring, n_jobs=1, error_score="raise")
        row = {
            "Modelo": name,
            "MAE_CV_MW": -scores["test_mae"].mean(),
            "MAE_CV_STD_MW": (-scores["test_mae"]).std(ddof=1),

            "RMSE_CV_MW": -scores["test_rmse"].mean(),

            "MAPE_CV_PCT": -scores["test_mape"].mean() * 100,
            "MAPE_CV_STD_PCT": (-scores["test_mape"] * 100).std(ddof=1),

            "R2_CV": scores["test_r2"].mean(),
            "Tiempo_s": time.perf_counter() - start,
        }

        rows.append(row)
        print(
            f"  {name:<24} "
            f"MAE: {row['MAE_CV_MW']:.4f} MW | "
            f"RMSE: {row['RMSE_CV_MW']:.4f} MW | "
            f"MAPE: {row['MAPE_CV_PCT']:.3f} % | "
            f"R²: {row['R2_CV']:.4f}"
        )
    return (
        pd.DataFrame(rows)
        .sort_values("MAPE_CV_PCT")
        .reset_index(drop=True)
    )


def main():
    df = load_data(DATASET_PATH)
    train, _ = split_temporal(df)
    if len(train) <= N_SPLITS:
        raise ValueError("Datos insuficientes para TimeSeriesSplit.")
    X_train, y_train = train[FEATURES], train[TARGET]
    results = compare_models(build_models(), X_train, y_train)
    results.to_csv(RESULTS_PATH, index=False, float_format="%.6f")
    model = tune_random_forest(X_train, y_train)
    PARAMS_PATH.write_text(
        json.dumps(model.get_params(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Hiperpar?metros guardados: {PARAMS_PATH}")


if __name__ == "__main__":
    main()
