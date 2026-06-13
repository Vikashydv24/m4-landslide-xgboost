"""
Evaluation metrics for LSM — matches Sahin (2020) Table 5.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              cohen_kappa_score, mean_squared_error,
                              confusion_matrix)
from scipy.stats import wilcoxon
import warnings


def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    oa     = accuracy_score(y_true, y_pred)
    auc    = roc_auc_score(y_true, y_pred_proba)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred_proba))
    kappa  = cohen_kappa_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "OA":          oa,
        "AUC":         auc,
        "RMSE":        rmse,
        "Kappa":       kappa,
        "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def aggregate_cv_metrics(fold_metrics):
    df   = pd.DataFrame(fold_metrics)
    cols = ["OA", "AUC", "RMSE", "Kappa", "Sensitivity", "Specificity"]
    summ = df[cols].agg(["mean", "std", "min", "max"]).T
    summ.columns = ["Mean", "Std", "Min", "Max"]
    summ.index.name = "Metric"
    return summ.round(4)


def wilcoxon_comparison(aucs_a, aucs_b, name_a="A", name_b="B"):
    if len(aucs_a) != len(aucs_b):
        return {"p_value": float("nan"), "significant": False,
                "model_a": name_a, "model_b": name_b,
                "statistic": float("nan"), "mean_diff": float("nan")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p = wilcoxon(aucs_a, aucs_b)
    return {
        "model_a":    name_a,
        "model_b":    name_b,
        "statistic":  stat,
        "p_value":    p,
        "significant": p < 0.05,
        "mean_diff":  np.mean(aucs_a) - np.mean(aucs_b),
    }


def run_wilcoxon_table(model_aucs):
    names = list(model_aucs.keys())
    rows  = []
    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if j <= i:
                rows.append({"Model A": na, "Model B": nb,
                             "p-value": "—", "Significant": "—"})
                continue
            res = wilcoxon_comparison(
                model_aucs[na], model_aucs[nb], na, nb)
            rows.append({
                "Model A":     na,
                "Model B":     nb,
                "p-value":     f"{res['p_value']:.4f}",
                "Significant": "YES" if res["significant"] else "no",
                "Mean AUC A":  f"{np.mean(model_aucs[na]):.4f}",
                "Mean AUC B":  f"{np.mean(model_aucs[nb]):.4f}",
            })
    return pd.DataFrame(rows)
