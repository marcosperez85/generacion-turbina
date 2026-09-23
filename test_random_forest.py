import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import search_random_forest as search
from train_random_forest import train_models
from turbines_common import FEATURES, TARGET, DATE_COLUMN, load_data, split_temporal, calculate_metrics


def synthetic_data():
    df = pd.DataFrame({name: np.arange(20, dtype=float) + 1 for name in FEATURES})
    df[TARGET] = np.arange(20, dtype=float) + 10
    df[DATE_COLUMN] = pd.date_range("2024-01-01", periods=20)
    return df


class WorkflowTests(unittest.TestCase):
    def test_loading_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            df = synthetic_data()
            df.iloc[::-1].to_csv(path, index=False)
            loaded = load_data(path)
            pd.testing.assert_frame_equal(loaded, df)
            train, test = split_temporal(loaded)
            self.assertEqual((len(train), len(test)), (16, 4))
            self.assertLess(train.fecha.max(), test.fecha.min())
            invalid = [df.drop(columns=[TARGET]), df.assign(fecha="invalid"),
                       df.assign(fecha=df.fecha.iloc[0]), df.assign(MW_TG=np.nan)]
            for frame in invalid:
                with self.subTest(columns=list(frame.columns)):
                    frame.to_csv(path, index=False)
                    with self.assertRaises(ValueError):
                        load_data(path)
            with self.assertRaises(ValueError):
                split_temporal(df.iloc[:1])

    def test_original_metrics_and_strict_threshold(self):
        metrics = calculate_metrics([100, 100, 100], [100, 101, 102])
        self.assertEqual(metrics["mae_mw"], 1)
        self.assertAlmostEqual(metrics["rmse_mw"], np.sqrt(5 / 3))
        self.assertEqual(metrics["mape_pct"], 1)
        self.assertAlmostEqual(metrics["within_1pct"], 100 / 3)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.assertTrue(np.isinf(calculate_metrics([0, 1], [1, 1])["mape_pct"]))
        with self.assertRaises(RuntimeError):
            calculate_metrics([1, 2], [np.nan, 2])
        with self.assertRaises(RuntimeError):
            calculate_metrics([1, 2], [1])

    def test_persisted_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, params = root / "data.csv", root / "params.json"
            validation, production = root / "validation.joblib", root / "production.joblib"
            df = synthetic_data()
            df.to_csv(data, index=False)
            config = {"n_estimators": 3, "max_depth": 2, "random_state": 42, "n_jobs": 1}
            params.write_text(json.dumps(config), encoding="utf-8")
            train_models(data, params, validation, production)
            val, prod = joblib.load(validation), joblib.load(production)
            self.assertEqual((val["training_rows"], prod["training_rows"]), (16, 20))
            self.assertEqual(val["trained_until"], df.fecha.iloc[15].isoformat())
            self.assertEqual(prod["trained_until"], df.fecha.iloc[-1].isoformat())
            self.assertEqual(val["hyperparameters"], prod["hyperparameters"])
            for artifact, rows in [(val, 16), (prod, 20)]:
                expected = RandomForestRegressor(**config).fit(df[FEATURES].iloc[:rows], df[TARGET].iloc[:rows])
                np.testing.assert_array_equal(artifact["model"].predict(df[FEATURES]), expected.predict(df[FEATURES]))
                self.assertEqual(artifact["features"], FEATURES)
                self.assertEqual(artifact["target"], TARGET)
            self.assertEqual(val["test_metrics"], calculate_metrics(df[TARGET].iloc[16:], val["model"].predict(df[FEATURES].iloc[16:])))
            self.assertEqual(prod["test_metrics"], val["test_metrics"])

    def test_search_uses_only_initial_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            df = synthetic_data()
            with patch.object(search, "load_data", return_value=df), patch.object(search, "build_models", return_value={}), patch.object(search, "compare_models", return_value=pd.DataFrame()) as compare, patch.object(search, "tune_random_forest", return_value=RandomForestRegressor(n_estimators=3)) as tune, patch.object(search, "RESULTS_PATH", root / "results.csv"), patch.object(search, "PARAMS_PATH", root / "params.json"):
                search.main()
                pd.testing.assert_frame_equal(tune.call_args.args[0], df[FEATURES].iloc[:16])
                pd.testing.assert_series_equal(tune.call_args.args[1], df[TARGET].iloc[:16])
                pd.testing.assert_frame_equal(compare.call_args.args[1], df[FEATURES].iloc[:16])
                self.assertEqual(json.loads((root / "params.json").read_text())["n_estimators"], 3)

    def test_search_configuration(self):
        df = synthetic_data()
        with patch.object(search, "RandomizedSearchCV") as constructor:
            constructor.return_value.best_score_ = -0.01
            search.tune_random_forest(df[FEATURES], df[TARGET])
            options = constructor.call_args.kwargs
            self.assertEqual(options["n_iter"], 40)
            self.assertEqual(options["scoring"], "neg_mean_absolute_percentage_error")
            self.assertEqual(options["cv"].n_splits, 5)
            for train, test in options["cv"].split(df):
                self.assertLess(train.max(), test.min())


if __name__ == "__main__":
    unittest.main()
