"""카테고리 공통 학습 도구. 시간순 분할과 단순 기준선 비교를 기본으로 한다."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error, median_absolute_error,
                             precision_recall_curve, precision_score, r2_score, recall_score, roc_auc_score,
                             root_mean_squared_error)

MAX_CATS = 250  # HistGradientBoosting은 범주 수가 max_bins(255) 이하여야 함 -> 나머지는 OTHER


def fit_params():
    return dict(categorical_features="from_dtype", max_iter=300, learning_rate=0.1, max_leaf_nodes=31, random_state=0)


def encode(train, others, cat_cols, num_cols):
    """범주 목록은 학습 데이터로만 만든다. 처음 보는 값과 결측은 OTHER."""
    cats = {c: train[c].astype("string").value_counts().index[:MAX_CATS].tolist() for c in cat_cols}

    def one(d):
        X = pd.DataFrame({c: pd.to_numeric(d[c], errors="coerce").astype("float32") for c in num_cols}, index=d.index)
        for c in cat_cols:
            s = d[c].astype("string")
            X[c] = pd.Categorical(s.where(s.isin(cats[c]), "OTHER").fillna("OTHER"), categories=cats[c] + ["OTHER"])
        return X

    return [one(d) for d in (train, *others)]


def binary_result(X_tr, y_tr, X_val, y_val, X_te, y_te, class_weight=None):
    """임계값은 검증 구간에서 F1 최대로 정하고, 평가 구간은 평가에만 쓴다."""
    model = HistGradientBoostingClassifier(class_weight=class_weight, **fit_params()).fit(X_tr, y_tr)
    p, r, t = precision_recall_curve(y_val, model.predict_proba(X_val)[:, 1])
    threshold = float(t[np.argmax((2 * p * r / np.maximum(p + r, 1e-12))[:-1])])
    proba = model.predict_proba(X_te)[:, 1]
    pred = proba >= threshold
    prevalence = float(np.mean(y_te))
    pr_auc = float(average_precision_score(y_te, proba))
    return {"kind": "binary", "n_train": len(X_tr), "n_test": len(X_te), "roc_auc": float(roc_auc_score(y_te, proba)),
            "pr_auc": pr_auc, "baseline_pr_auc": prevalence, "lift": pr_auc / prevalence,
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall": float(recall_score(y_te, pred, zero_division=0)), "f1": float(f1_score(y_te, pred, zero_division=0))}


def regression_scores(y, pred):
    return {"mae": float(mean_absolute_error(y, pred)), "rmse": float(root_mean_squared_error(y, pred)),
            "medae": float(median_absolute_error(y, pred)), "r2": float(r2_score(y, pred))}


def regression_result(X_tr, y_tr, X_te, y_te, persistence=None):
    """기준선: 학습 평균과 (있으면) 과거 값 그대로 쓰기 중 오차가 작은 쪽."""
    model = HistGradientBoostingRegressor(**fit_params()).fit(X_tr, y_tr)
    scores = regression_scores(y_te, model.predict(X_te))
    mean_mae = float(np.abs(y_te - y_tr.mean()).mean())
    out = {"kind": "regression", "n_train": len(X_tr), "n_test": len(X_te), **scores, "mean_mae": mean_mae}
    if persistence is not None:
        out["persistence_mae"] = float(np.abs(y_te - persistence).mean())
    out["baseline_mae"] = min(mean_mae, out.get("persistence_mae", mean_mae))
    out["improve"] = 1 - scores["mae"] / out["baseline_mae"]  # 양수면 기준선보다 나음
    return out


def add_lags(d, keys, target, lags):
    d = d[keys + ["t", target] + [c for c in d.columns if c.startswith("x_")]].dropna(subset=[target])
    for lag in lags:
        prev = d[keys + ["t", target]].assign(t=d.t + lag).rename(columns={target: f"lag{lag}"})
        d = d.merge(prev, on=keys + ["t"], how="left")
    return d


def lag_regression(df, keys, target, train_range, test_range, lags, extra_num=()):
    """같은 대상의 과거 값으로 다음 기간을 예측. 기준선은 lags[0] 만큼 이전 값 그대로."""
    d = add_lags(df, keys, target, lags)
    tr = d[d.t.between(*train_range)]
    te = d[d.t.between(*test_range) & d[f"lag{lags[0]}"].notna()]
    num = [f"lag{lag}" for lag in lags] + list(extra_num)
    X_tr, X_te = encode(tr, [te], keys, num)
    return regression_result(X_tr, tr[target].to_numpy(float), X_te, te[target].to_numpy(float), te[f"lag{lags[0]}"].to_numpy(float))


def model_row(title, result, train_period, test_period, group="", note=""):
    return {"title": title, "group": group, "train_period": train_period, "test_period": test_period, "note": note, **result}


def self_check():
    d = pd.DataFrame({"k": ["a", "a", "b"], "t": [1, 2, 2], "y": [10.0, 20.0, 5.0]})
    out = add_lags(d, ["k"], "y", [1]).sort_values(["k", "t"])
    assert out.lag1.iloc[1] == 10 and out.lag1.isna().sum() == 2, "과거 값은 같은 대상의 이전 값이어야 함"
