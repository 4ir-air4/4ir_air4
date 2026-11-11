"""ldh/datasets/raw/bts_zip/ 다중 타깃 학습. 출발 전에 알 수 있는 일정 정보만 특성으로 사용합니다.

학습: TRAIN_YEARS 전체에서 파일(월)마다 SAMPLE_FRAC 표본 추출, 평가: TEST_YEAR 전체
사용법: python -m ldh.scripts.train [task ...]   인자 없으면 TASKS 전부. 예) python -m ldh.scripts.train cancelled diverted
결과: models/<task>.pkl, models/metrics/<task>.json
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report, confusion_matrix,
                             f1_score, mean_absolute_error, median_absolute_error, precision_recall_curve,
                             r2_score, roc_auc_score, root_mean_squared_error)
from sklearn.model_selection import train_test_split

from ldh.core.data import START_YEAR

TRAIN_YEARS = range(START_YEAR, 2026)
TEST_YEAR = 2026
SAMPLE_FRAC = 0.15  # 학습 기간 전체(약 3,800만 행)에서 월마다 15%씩 -> 약 570만 행
CAT = ["Reporting_Airline", "Origin", "Dest"]
NUM = ["Year", "Month", "DayOfWeek", "dep_hour", "arr_hour", "CRSElapsedTime", "Distance"]
MAX_CATS = 250  # HistGradientBoosting은 범주 수가 max_bins(255) 이하여야 함 -> 나머지는 OTHER
MODEL_DIR = Path("ldh/results/legacy_models")  # 예전 train.py 모델. models/ 는 팀원 폴더
METRIC_DIR = MODEL_DIR / "metrics"

CAUSES = {"CarrierDelay": "항공사", "WeatherDelay": "날씨", "NASDelay": "관제(NAS)",
          "SecurityDelay": "보안", "LateAircraftDelay": "앞편 지연"}
CANCEL_CODES = {"A": "항공사", "B": "날씨", "C": "관제(NAS)", "D": "보안"}


def operated(d):
    return (d.Cancelled == 0) & (d.Diverted == 0)


def delay_cause(d):
    """15분 이상 지연편에서 지연 시간이 가장 큰 원인 (2003년 6월 이후만 기록됨)."""
    c = d[list(CAUSES)]
    ok = (d.ArrDel15 == 1) & c.notna().all(axis=1) & (c.sum(axis=1) > 0)
    return c[ok].idxmax(axis=1).map(CAUSES)


# task: (제목, 종류, 타깃 함수(df -> 대상 행만 담은 Series, NaN은 제외됨), class_weight)
TASKS = {
    "arr_del15": ("도착 지연 여부 (15분 이상)", "binary", lambda d: d.ArrDel15[operated(d)], None),
    "arr_delay": ("도착 지연 시간 (분)", "regression", lambda d: d.ArrDelay[operated(d)], None),
    "dep_del15": ("출발 지연 여부 (15분 이상)", "binary", lambda d: d.DepDel15[d.Cancelled == 0], None),
    "dep_delay": ("출발 지연 시간 (분)", "regression", lambda d: d.DepDelay[d.Cancelled == 0], None),
    "cancelled": ("결항 여부", "binary", lambda d: d.Cancelled, "balanced"),
    "cancel_code": ("결항 사유 (결항된 경우)", "multiclass",
                    lambda d: d.CancellationCode[d.Cancelled == 1].map(CANCEL_CODES), "balanced"),
    "diverted": ("회항 여부", "binary", lambda d: d.Diverted[d.Cancelled == 0], "balanced"),
    "delay_cause": ("주요 지연 원인 (15분 이상 지연된 경우)", "multiclass", delay_cause, "balanced"),
    "taxi_out": ("택시아웃 시간 (분)", "regression", lambda d: d.TaxiOut[d.Cancelled == 0], None),
    "taxi_in": ("택시인 시간 (분)", "regression", lambda d: d.TaxiIn[operated(d)], None),
    "air_time": ("실제 비행 시간 (분)", "regression", lambda d: d.AirTime[operated(d)], None),
}


def load(years, frac=1.0):
    files = [f for y in years for f in sorted(Path("ldh/datasets/processed/bts_parquet").glob(f"ontime_{y}_*.parquet"))]
    if not files:
        raise SystemExit(f"ldh/datasets/processed/bts_parquet/ 에 {list(years)} parquet 파일이 없습니다. ldh/scripts/prepare_data.py 를 먼저 실행하세요.")
    # 파일마다 읽자마자 표본 추출 (전부 읽은 뒤 추출하면 메모리 부족)
    parts = [pd.read_parquet(f).sample(frac=frac, random_state=0) if frac < 1 else pd.read_parquet(f) for f in files]
    return pd.concat(parts, ignore_index=True)


def features(df, cats):
    """원본 컬럼(prepare_data.DTYPES) -> 모델 입력. 학습과 앱에서 같이 사용."""
    X = pd.DataFrame({
        "Year": df.Year, "Month": df.Month, "DayOfWeek": df.DayOfWeek,
        "dep_hour": df.CRSDepTime // 100, "arr_hour": df.CRSArrTime // 100,
        "CRSElapsedTime": df.CRSElapsedTime, "Distance": df.Distance,
    })
    for c in CAT:
        X[c] = pd.Categorical(df[c].where(df[c].isin(cats[c]), "OTHER"), categories=cats[c] + ["OTHER"])
    return X[CAT + NUM]


def binary_metrics(model, X_val, y_val, X_te, y_te):
    # 임계값은 검증셋에서 F1 최대가 되는 값 (테스트셋은 평가에만 사용)
    p, r, t = precision_recall_curve(y_val, model.predict_proba(X_val)[:, 1])
    threshold = float(t[np.argmax((2 * p * r / np.maximum(p + r, 1e-12))[:-1])])
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= threshold).astype(int)
    p, r, _ = precision_recall_curve(y_te, proba)
    idx = np.linspace(0, len(p) - 1, 200).astype(int)  # 차트용으로 200개 점만 저장
    by_threshold = []
    for th in sorted({0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, round(threshold, 3)}):
        flag = proba >= th
        tp = int((flag & (y_te == 1)).sum())
        by_threshold.append({"임계값": th, "양성 예측 비율": float(flag.mean()),
                             "정밀도": tp / max(int(flag.sum()), 1), "재현율": tp / max(int(y_te.sum()), 1)})
    return {
        "positive_rate": float(y_te.mean()),  # PR-AUC 기준선(무작위 예측 시 값)
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "pr_auc": float(average_precision_score(y_te, proba)),
        "threshold": threshold,
        "report": classification_report(y_te, pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "by_threshold": by_threshold,
        "pr_curve": {"precision": p[idx].tolist(), "recall": r[idx].tolist()},
    }


def regression_scores(y, pred):
    return {"mae": float(mean_absolute_error(y, pred)), "rmse": float(root_mean_squared_error(y, pred)),
            "medae": float(median_absolute_error(y, pred)), "r2": float(r2_score(y, pred))}


def regression_metrics(model, train_mean, X_te, y_te, airline):
    y = y_te.to_numpy()
    pred = model.predict(X_te)
    edges = np.unique(np.quantile(y, [0, .2, .4, .6, .8, .95, 1]))  # 실제값 분위 구간 (상위 5% 따로)
    bins = pd.cut(y, edges, include_lowest=True)
    err = pd.DataFrame({"구간": bins.rename_categories([f"{iv.left:.0f}~{iv.right:.0f}" for iv in bins.categories]),
                        "actual": y, "pred": pred, "abs_err": np.abs(y - pred), "airline": airline.to_numpy()})
    agg = dict(건수=("actual", "size"), 실제평균=("actual", "mean"), 예측평균=("pred", "mean"), MAE=("abs_err", "mean"))
    return {
        "model": regression_scores(y, pred),
        "baseline_mean": regression_scores(y, np.full_like(y, train_mean)),  # 학습 평균으로 항상 예측
        "actual_dist": {k: float(v) for k, v in pd.Series(y).describe(percentiles=[.1, .25, .5, .75, .9, .99]).items()},
        "by_bin": err.groupby("구간", observed=False).agg(**agg).reset_index().astype({"구간": str}).to_dict("records"),
        "by_airline": err.groupby("airline").agg(**agg).reset_index().to_dict("records"),
    }


def multiclass_metrics(model, X_te, y_te):
    pred = model.predict(X_te)
    labels = list(model.classes_)
    return {
        "classes": labels,
        "majority_rate": float(y_te.value_counts(normalize=True).iloc[0]),  # 최빈 클래스만 찍을 때 정확도
        "accuracy": float(accuracy_score(y_te, pred)),
        "macro_f1": float(f1_score(y_te, pred, average="macro")),
        "report": classification_report(y_te, pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_te, pred, labels=labels).tolist(),
        "class_share": {"실제": y_te.value_counts(normalize=True).reindex(labels).fillna(0).tolist(),
                        "예측": pd.Series(pred).value_counts(normalize=True).reindex(labels).fillna(0).tolist()},
    }


def run(name, train, test):
    title, kind, target, class_weight = TASKS[name]
    y_all, y_te = target(train).dropna(), target(test).dropna()
    rows = train.loc[y_all.index]
    cats = {c: rows[c].value_counts().index[:MAX_CATS].tolist() for c in CAT}
    X_all, X_te = features(rows, cats), features(test.loc[y_te.index], cats)
    if kind == "binary":
        y_all, y_te = y_all.astype(int), y_te.astype(int)

    print(f"[{name}] 학습 시작: {title} ({len(X_all):,}행)", flush=True)
    params = dict(categorical_features="from_dtype", max_iter=500, learning_rate=0.1, max_leaf_nodes=63,
                  random_state=0, verbose=2)  # verbose=2: 반복(iteration)마다 손실·소요 시간 출력
    start = time.time()
    if kind == "regression":
        model = HistGradientBoostingRegressor(**params).fit(X_all, y_all)
        metrics = regression_metrics(model, float(y_all.mean()), X_te, y_te, test.Reporting_Airline.loc[y_te.index])
        threshold = None
    else:
        model = HistGradientBoostingClassifier(class_weight=class_weight, **params)
        if kind == "binary":
            X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.1, stratify=y_all, random_state=0)
            model.fit(X_tr, y_tr)
            metrics = binary_metrics(model, X_val, y_val, X_te, y_te)
            threshold = metrics["threshold"]
        else:
            model.fit(X_all, y_all)
            metrics = multiclass_metrics(model, X_te, y_te)
            threshold = None

    years = rows.Year
    metrics = {"task": name, "title": title, "kind": kind, "class_weight": class_weight,
               "train_years": [int(years.min()), int(years.max())], "test_year": TEST_YEAR,
               "n_train": len(X_all), "n_test": len(X_te), "n_iter": int(model.n_iter_),
               "seconds": round(time.time() - start), **metrics}
    joblib.dump({"model": model, "cats": cats, "threshold": threshold, "title": title, "kind": kind},
                MODEL_DIR / f"{name}.pkl")
    (METRIC_DIR / f"{name}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    keys = {"binary": ["roc_auc", "pr_auc", "positive_rate"], "multiclass": ["accuracy", "macro_f1", "majority_rate"]}
    summary = {k: round(metrics[k], 4) for k in keys[kind]} if kind != "regression" else \
        {k: round(v, 3) for k, v in metrics["model"].items()} | {"baseline_mae": round(metrics["baseline_mean"]["mae"], 3)}
    print(f"[{name}] {title}  train {len(X_all):,} ({metrics['train_years']}) / test {len(X_te):,}  "
          f"{metrics['seconds']}s  {summary}", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or list(TASKS)
    unknown = set(names) - set(TASKS)
    if unknown:
        raise SystemExit(f"알 수 없는 task: {unknown}. 가능: {list(TASKS)}")
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load(TRAIN_YEARS, SAMPLE_FRAC), load([TEST_YEAR])
    print(f"loaded train {len(train):,} rows / test {len(test):,} rows", flush=True)
    for name in names:
        run(name, train, test)
