"""Train independent validation and production Random Forest artifacts."""
import json
import joblib
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from turbines_common import (
    DATASET_PATH, PARAMS_PATH, VALIDATION_PATH, PRODUCTION_PATH,
    FEATURES, TARGET, DATE_COLUMN, RANDOM_STATE,
    load_data, split_temporal, calculate_metrics,
)


def make_artifact(model, training, test, metrics, role):
    return {
        "model": model,
        "model_name": "Random Forest optimizado",
        "role": role,
        "features": list(FEATURES),
        "target": TARGET,
        "trained_from": training[DATE_COLUMN].iloc[0].isoformat(),
        "trained_until": training[DATE_COLUMN].iloc[-1].isoformat(),
        "training_rows": len(training),
        "hyperparameters": model.get_params(),
        "test_metrics": dict(metrics),
        "metrics_source": "rf_validation: holdout temporal del 20 % final",
        "test_from": test[DATE_COLUMN].iloc[0].isoformat(),
        "test_until": test[DATE_COLUMN].iloc[-1].isoformat(),
        "test_rows": len(test),
    }


def train_models(dataset_path=DATASET_PATH, params_path=PARAMS_PATH,
                 validation_path=VALIDATION_PATH, production_path=PRODUCTION_PATH):
    params = json.loads(params_path.read_text(encoding="utf-8"))
    config = {"random_state": RANDOM_STATE, "n_jobs": -1, **params}
    df = load_data(dataset_path)
    train, test = split_temporal(df)
    model = RandomForestRegressor(**config)
    model.fit(train[FEATURES], train[TARGET])
    metrics = calculate_metrics(test[TARGET], model.predict(test[FEATURES]))
    validation = make_artifact(model, train, test, metrics, "validation")
    joblib.dump(validation, validation_path)
    production_model = clone(model)
    production_model.fit(df[FEATURES], df[TARGET])
    production = make_artifact(production_model, df, test, metrics, "production")
    joblib.dump(production, production_path)
    return validation, production


def main():
    validation, _ = train_models()
    print("Evaluaci?n sobre el 20 % final:")
    for name, value in validation["test_metrics"].items():
        print(f"  {name}: {value:.6f}")
    print(f"Modelos guardados: {VALIDATION_PATH}, {PRODUCTION_PATH}")


if __name__ == "__main__":
    main()
