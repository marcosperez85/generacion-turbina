from pathlib import Path
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset_CC02.csv"
MODEL_PATH = BASE_DIR / "best_turbines_model.joblib"
RESULTS_PATH = BASE_DIR / "resultados_modelos.csv"
PLOT_PATH = BASE_DIR / "comparacion_modelos.png"
FEATURES = ["MW_TG", "Humedad", "Presion", "Temp", "Viento",
            "Viento_cos", "Viento_sin", "VIGV"]
TARGET, DATE_COLUMN = "MW_ST", "fecha"
RANDOM_STATE, TEST_SIZE, N_SPLITS = 42, 0.20, 5


def load_data(path):
    print("Cargando y validando datos...")
    df = pd.read_csv(path)
    required = [DATE_COLUMN, *FEATURES, TARGET]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    if df[DATE_COLUMN].isna().any():
        raise ValueError(f"Hay {df[DATE_COLUMN].isna().sum()} fechas inválidas.")
    if df[required].isna().any().any():
        counts = df[required].isna().sum()
        raise ValueError(f"Se encontraron datos faltantes:\n{counts[counts > 0]}")
    if df[DATE_COLUMN].duplicated().any():
        raise ValueError(f"Hay {df[DATE_COLUMN].duplicated().sum()} timestamps duplicados.")
    df = df.sort_values(DATE_COLUMN, kind="stable").reset_index(drop=True)
    print(f"Dimensiones: {df.shape}")
    print(f"Período: {df[DATE_COLUMN].min()} a {df[DATE_COLUMN].max()}")
    print(f"Target continuo: {TARGET} ({df[TARGET].min():.3f} a {df[TARGET].max():.3f} MW)\n")
    return df


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


def create_plots(results, dates, actual, predicted, model_name):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.barplot(data=results, x="MAE_CV_MW", y="Modelo",
                ax=axes[0], color="skyblue")
    axes[0].set(title="MAE medio en validación temporal",
                xlabel="MAE (MW; menor es mejor)", ylabel="")
    axes[1].plot(dates, actual, label="Real", linewidth=1.2)
    axes[1].plot(dates, predicted, label="Predicción", linewidth=1, alpha=.85)
    axes[1].set(title=f"Prueba temporal — {model_name}", ylabel="MW_ST (MW)")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=30)
    residuals = actual.to_numpy() - predicted
    axes[2].scatter(predicted, residuals, alpha=.45, s=18)
    axes[2].axhline(0, color="red", linestyle="--")
    axes[2].set(title="Residuos en prueba", xlabel="Predicción (MW)",
                ylabel="Real − predicción (MW)")
    for axis in axes:
        axis.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_data(DATASET_PATH)
    split = int(len(df) * (1 - TEST_SIZE))
    train, test = df.iloc[:split], df.iloc[split:]
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    print(f"Entrenamiento: {len(train)} filas, hasta {train[DATE_COLUMN].iloc[-1]}")
    print(f"Prueba: {len(test)} filas, desde {test[DATE_COLUMN].iloc[0]}\n")

    models = build_models()
    results = compare_models(models, X_train, y_train)
    results.to_csv(RESULTS_PATH, index=False, float_format="%.6f")
    winner = str(results.iloc[0]["Modelo"])
    evaluation_model = clone(models[winner])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        evaluation_model.fit(X_train, y_train)
    predicted = np.asarray(evaluation_model.predict(X_test))
    if len(predicted) != len(y_test) or not np.isfinite(predicted).all():
        raise RuntimeError("Las predicciones de prueba son inválidas.")

    # Error porcentual de cada predicción respecto del valor real
    percentage_errors = (
        np.abs(y_test.to_numpy() - predicted)
        / np.abs(y_test.to_numpy())
    ) * 100

    # Metricas finales
    metrics = {
        "mae_mw": mean_absolute_error(y_test, predicted),
        "rmse_mw": mean_squared_error(y_test, predicted) ** .5,
        "r2": r2_score(y_test, predicted),
        "mape_pct": percentage_errors.mean(),
        "within_1pct": np.mean(percentage_errors < 1) * 100,
    }

    print(f"\nMejor modelo por MAPE de validación: {winner}")
    print("\nEvaluación final sobre el 20 % más reciente:")
    print(f"  MAE:  {metrics['mae_mw']:.4f} MW")
    print(f"  RMSE: {metrics['rmse_mw']:.4f} MW")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  MAPE: {metrics['mape_pct']:.3f} %")
    print(f"  Predicciones con error < 1 %: {metrics['within_1pct']:.2f} %")

    if metrics["mape_pct"] < 1:
        print("  Cumple objetivo de error porcentual promedio < 1 %")
    else:
        print("  No cumple objetivo de error porcentual promedio < 1 %")

    final_model = clone(models[winner])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        final_model.fit(df[FEATURES], df[TARGET])
    artifact = {"model": final_model, "model_name": winner, "features": FEATURES,
                "target": TARGET, "trained_until": df[DATE_COLUMN].max().isoformat(),
                "test_metrics": metrics}
    joblib.dump(artifact, MODEL_PATH)
    loaded = joblib.load(MODEL_PATH)
    check = loaded["model"].predict(df[FEATURES].iloc[[0]])
    if len(check) != 1 or not np.isfinite(check).all():
        raise RuntimeError("El modelo guardado no superó la prueba de carga.")
    print("\nArchivos generados:")
    print(f"  Modelo: {MODEL_PATH}")
    print(f"  Resultados: {RESULTS_PATH}")
    print(f"  Gráficos: {PLOT_PATH}")


if __name__ == "__main__":
    main()
