"""인천 수집 자료(collect_icn.py)로 저장된 인천 지연 모델을 실전 채점한다.  사용법: python -m ldh.scripts.predict_icn_live [모델]

입력은 학습 때와 같은 방식(icn())으로 만들되, 과거 이력은 수집 자료만 쓴다(학습 자료는 2026-06에서 끝남).
수집 첫날에 가까울수록 이력 입력(최근 지연률, 편명 30일 등)이 비어 학습 때보다 불리하다.
정답은 두 가지로 본다.
  학습 기준: 수집 시각에 LIVE_OFFSET 을 더한 지연으로 학습 때와 같은 판정(출발은 학습 구간 중앙값을 뺌)
  API 기준:  API 예상 시각 - 계획 시각 > 15분 (운항마감 지연 표시와 일치)
결과: ldh/results/full/korea_ops/icn_live_eval.json, 편별 예측 pred_live_<과제>__<모델>.parquet
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ldh.categories.korea_ops.models import SELECTED_GROUPS, recent_context, recent_delays
from ldh.categories.korea_ops.pipeline import live_flights, load_flights
from ldh.core.models import load_model, prev_indices
from ldh.core.store import Store

TASKS = {"도착": "icn_arr_delay", "출발": "icn_dep_delay"}
CUTS, GRADES = (0.3, 0.5), ["낮음", "보통", "높음"]  # ldh/core/icn_risk.py 와 같은 등급
BASE = pd.Timestamp("2000-01-01")


def prepare(live, train):
    """수집 편에 학습 자료와 같은 파생 열을 붙인다. 국가는 학습 자료의 공항 코드별 마지막 값."""
    country = train.sort_values("sched").drop_duplicates("code", keep="last").set_index("code").country
    d = live.copy()
    d["airline"] = d.flight.astype("string").str[:2]  # 학습 자료와 같은 정의(API 의 한글 항공사명 대신)
    d["country"] = d.code.map(country)
    d["ym"] = d.sched.dt.strftime("%Y-%m")
    d["month"], d["hour"], d["dow"] = d.sched.dt.month, d.sched.dt.hour, d.sched.dt.day_of_week
    d["sched_hour"] = d.groupby(["direction", d.sched.dt.floor("h")]).month.transform("size")
    return d


def inputs(flights, direction, meta, offsets):
    """icn() 과 같은 순서의 입력 행렬. 열 이름이 저장 모델과 다르면 멈춘다."""
    d = flights[flights.direction == direction].sort_values("sched").reset_index(drop=True)
    arrays, columns = [], []
    for c in ["airline", "code", "country"]:
        cats = meta["encoders"][c]
        code = pd.Index(cats).get_indexer(d[c].astype("string")).astype(np.float32)  # 학습 때와 같은 코드, 없는 값은 -1
        code[code < 0] = len(cats)
        arrays.append(code)
        columns.append(c)
    for c in ["hour", "dow", "month", "sched_hour"]:
        arrays.append(d[c].to_numpy(np.float32))
        columns.append(c)
    minutes = ((d.sched - BASE).dt.total_seconds() / 60).to_numpy(np.int64)
    delay = d.delay.to_numpy(np.float32)
    prev = prev_indices(d.flight.astype("string").fillna("").factorize()[0].astype(np.int64), minutes, 5)
    arrays.append(recent_delays(delay, prev, minutes + delay.astype(np.int64), minutes - meta["lead_minutes"]))
    columns += [f"prev_delay_{i}" for i in range(1, 6)] + ["prev_delay_mean", "prev_delay_count"]
    extra, names = recent_context(flights, d, direction, offsets, meta.get("context_groups", SELECTED_GROUPS))
    arrays += extra
    columns += names
    assert columns == meta["columns"], f"입력 열이 저장 모델과 다름: {set(columns) ^ set(meta['columns'])}"
    return d, np.column_stack(arrays).astype(np.float32)


def scores(y, p):
    y = np.asarray(y)
    ok = len(np.unique(y)) == 2
    return {"편수": int(len(y)), "실제 지연률": round(float(y.mean()), 4), "평균 점수": round(float(p.mean()), 4),
            "PR-AUC": round(float(average_precision_score(y, p)), 4) if ok else None,
            "ROC-AUC": round(float(roc_auc_score(y, p)), 4) if ok else None}


def grade_table(y, p):
    g = pd.cut(p, [-np.inf, *CUTS, np.inf], right=False, labels=GRADES)
    t = pd.DataFrame({"등급": g, "y": y}).groupby("등급", observed=False).y.agg(["size", "mean"])
    return {k: {"편수": int(r["size"]), "실제 지연률": None if pd.isna(r["mean"]) else round(float(r["mean"]), 4)} for k, r in t.iterrows()}


def run(model_name="lightgbm"):
    store = Store("korea_ops")
    train = load_flights()
    dep = train[train.direction == "출발"]
    offsets = {"도착": 0.0, "출발": float(dep.delay[dep.ym <= "2024-12"].median())}  # icn() 과 같은 출발 판정 기준
    flights = prepare(live_flights(), train)
    days = sorted(flights.date.unique())
    report = {"수집 기간": f"{days[0]}~{days[-1]} ({len(days)}일)", "모델": model_name, "출발 판정 기준(분)": offsets["출발"], "방향": {}}
    for direction, task in TASKS.items():
        predict, meta = load_model(store, task, model_name)
        d, X = inputs(flights, direction, meta, offsets)
        p = np.asarray(predict(type("D", (), {"X": X})(), np.arange(len(d))), dtype=float)
        y_train_rule = (d.delay - offsets[direction] >= 15).astype(int).to_numpy()
        y_api = (d.api_delay > 15).astype(int).to_numpy()
        out = d[["date", "sched", "flight", "code", "plan", "actual", "api_delay", "delay"]].assign(점수=p, 정답_학습기준=y_train_rule, 정답_API=y_api)
        out.to_parquet(store.dir / f"pred_live_{task}__{model_name}.parquet", index=False)
        later = (d.date >= days[min(3, len(days) - 1)]).to_numpy()  # 이력 입력이 어느 정도 찬 4일째 이후
        report["방향"][direction] = {
            "전체 (학습 기준 정답)": scores(y_train_rule, p),
            "전체 (API 기준 정답)": scores(y_api, p),
            "4일째 이후 (학습 기준 정답)": scores(y_train_rule[later], p[later]),
            "등급별 (학습 기준 정답)": grade_table(y_train_rule, p),
            "날짜별 (학습 기준 정답)": {day: scores(y_train_rule[(d.date == day).to_numpy()], p[(d.date == day).to_numpy()]) for day in days},
            "입력 빈 비율": {c: round(float(np.isnan(X[:, i]).mean()), 3) for i, c in enumerate(meta["columns"]) if "rate" in c or c == "prev_delay_1"},
        }
    (store.dir / "icn_live_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def check():
    y, p = np.array([0, 1, 1, 0]), np.array([0.1, 0.6, 0.4, 0.2])
    assert scores(y, p)["PR-AUC"] == 1.0 and scores(np.ones(3), np.ones(3))["PR-AUC"] is None, "한쪽 정답만 있으면 지표 없음"
    g = grade_table(y, p)
    assert g["낮음"]["편수"] == 2 and g["보통"]["편수"] == 1 and g["높음"]["실제 지연률"] == 1.0, "등급 경계 0.3·0.5"


if __name__ == "__main__":
    check()
    print(json.dumps(run(*sys.argv[1:2]), ensure_ascii=False, indent=1))
