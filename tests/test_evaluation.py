"""
Unit tests for src/evaluation.py
Run: python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.evaluation import (
    compute_metrics,
    aggregate_cv_metrics,
    wilcoxon_comparison,
    run_wilcoxon_table,
)


class TestComputeMetrics:

    def test_perfect_predictions(self):
        y    = np.array([0, 0, 1, 1, 0, 1])
        prob = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        m    = compute_metrics(y, prob)
        assert m["AUC"]   == pytest.approx(1.0)
        assert m["OA"]    == pytest.approx(1.0)
        assert m["RMSE"]  == pytest.approx(0.0)
        assert m["Kappa"] == pytest.approx(1.0)

    def test_random_auc_near_half(self):
        rng  = np.random.default_rng(0)
        y    = rng.integers(0, 2, 1000)
        prob = rng.uniform(0, 1, 1000)
        m    = compute_metrics(y, prob)
        assert 0.4 < m["AUC"] < 0.6

    def test_all_metrics_in_range(self):
        rng  = np.random.default_rng(1)
        y    = rng.integers(0, 2, 200)
        prob = rng.uniform(0, 1, 200)
        m    = compute_metrics(y, prob)
        assert 0.0 <= m["OA"]          <= 1.0
        assert 0.0 <= m["AUC"]         <= 1.0
        assert m["RMSE"]               >= 0.0
        assert -1.0 <= m["Kappa"]      <= 1.0
        assert 0.0 <= m["Sensitivity"] <= 1.0
        assert 0.0 <= m["Specificity"] <= 1.0

    def test_required_keys_present(self):
        y    = np.array([0, 1, 0, 1])
        prob = np.array([0.1, 0.9, 0.2, 0.8])
        m    = compute_metrics(y, prob)
        for key in ["OA","AUC","RMSE","Kappa",
                    "Sensitivity","Specificity",
                    "TP","TN","FP","FN"]:
            assert key in m

    def test_confusion_matrix_sums_to_n(self):
        y    = np.array([0, 0, 1, 1, 0, 1])
        prob = np.array([0.2, 0.8, 0.9, 0.1, 0.3, 0.7])
        m    = compute_metrics(y, prob)
        assert m["TP"] + m["TN"] + m["FP"] + m["FN"] == len(y)

    def test_worst_case_rmse(self):
        y    = np.array([0, 0, 1, 1])
        prob = np.array([1.0, 1.0, 0.0, 0.0])
        m    = compute_metrics(y, prob)
        assert m["RMSE"] == pytest.approx(1.0)


class TestAggregateCVMetrics:

    def _folds(self, n=5):
        rng = np.random.default_rng(0)
        return [compute_metrics(
                    rng.integers(0,2,100),
                    rng.uniform(0,1,100))
                for _ in range(n)]

    def test_output_has_correct_columns(self):
        df = aggregate_cv_metrics(self._folds())
        for col in ["Mean","Std","Min","Max"]:
            assert col in df.columns

    def test_mean_in_valid_range(self):
        df = aggregate_cv_metrics(self._folds(10))
        assert 0.0 <= df.loc["AUC","Mean"] <= 1.0
        assert 0.0 <= df.loc["OA","Mean"]  <= 1.0
        assert df.loc["RMSE","Mean"]       >= 0.0

    def test_std_non_negative(self):
        df = aggregate_cv_metrics(self._folds())
        for m in ["OA","AUC","RMSE","Kappa"]:
            assert df.loc[m,"Std"] >= 0.0


class TestWilcoxon:

    def test_identical_not_significant(self):
        a = [0.82, 0.84, 0.81, 0.83, 0.85]
        r = wilcoxon_comparison(a, a)
        assert not r["significant"]

    def test_clearly_different_is_significant(self):
        # Need enough varied pairs for Wilcoxon to detect significance
        a = [0.90, 0.91, 0.88, 0.92, 0.89, 0.93, 0.87, 0.91, 0.90, 0.92]
        b = [0.60, 0.58, 0.62, 0.57, 0.61, 0.59, 0.63, 0.56, 0.60, 0.58]
        r = wilcoxon_comparison(a, b)
        assert r["significant"], f"Expected significant, got p={r['p_value']}"
        assert r["p_value"] < 0.05

    def test_required_keys(self):
        a = [0.80, 0.82, 0.81]
        b = [0.75, 0.77, 0.76]
        r = wilcoxon_comparison(a, b)
        for k in ["p_value","significant","mean_diff",
                  "statistic","model_a","model_b"]:
            assert k in r

    def test_unequal_length_returns_nan(self):
        import math
        r = wilcoxon_comparison([0.80, 0.82], [0.75])
        assert math.isnan(r["p_value"])

    def test_wilcoxon_table_shape(self):
        aucs = {
            "RF":      [0.81,0.80,0.82,0.83,0.79],
            "GBM":     [0.83,0.82,0.84,0.85,0.81],
            "XGBoost": [0.85,0.84,0.86,0.87,0.83],
        }
        df = run_wilcoxon_table(aucs)
        assert len(df) == 9    # 3×3 matrix
        for col in ["Model A","Model B","p-value","Significant"]:
            assert col in df.columns
