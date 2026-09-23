"""Shared data validation, chronological split and original metric definitions."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset_CC02.csv"
PARAMS_PATH = BASE_DIR / "best_rf_params.json"
VALIDATION_PATH = BASE_DIR / "rf_validation.joblib"
PRODUCTION_PATH = BASE_DIR / "rf_production.joblib"
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


def split_temporal(df):
    split = int(len(df) * (1 - TEST_SIZE))
    if split == 0 or split == len(df):
        raise ValueError("Se necesitan datos en ambos conjuntos temporales.")
    return df.iloc[:split], df.iloc[split:]


def calculate_metrics(actual, predicted):
    actual = pd.Series(np.asarray(actual))
    predicted = np.asarray(predicted)
    if len(predicted) != len(actual) or not np.isfinite(predicted).all():
        raise RuntimeError("Las predicciones de prueba son inválidas.")

    # Error porcentual de cada predicción respecto del valor real
    percentage_errors = (
        np.abs(actual.to_numpy() - predicted)
        / np.abs(actual.to_numpy())
    ) * 100

    metrics = {
        "mae_mw": mean_absolute_error(actual, predicted),
        "rmse_mw": mean_squared_error(actual, predicted) ** .5,
        "r2": r2_score(actual, predicted),
        "mape_pct": percentage_errors.mean(),
        "within_1pct": np.mean(percentage_errors < 1) * 100,
    }

    return metrics
