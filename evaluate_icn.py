"""인천 지연 모델의 여러 시점 재평가.  사용법: python evaluate_icn.py [도착|출발] [모델 ...]

각 구간은 그 이전 자료로만 학습하고, 검증 구간으로 조기 종료와 확률 보정을 한 뒤 평가 구간에 적용한다.
기존 모델과 예측 파일은 바꾸지 않고, 결과만 outputs/korea_ops/icn_eval_<방향>.json 에 쓴다.
범주 코드표는 2024년까지의 자료로 만든 것을 모든 구간에 함께 쓰므로, 첫 구간에는 이후 자료가 조금 반영된다.
"""
import json
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from categories.korea_ops.models import icn
from core.models import MODEL_NAMES, MODELS, TEST, TRAIN, VAL, train_one
from core.store import Store

# (이름, 학습 끝, 검증 시작, 검증 끝, 평가 시작, 평가 끝). 2024년은 원본 공백이 있어 자료가 있는 달로 잡는다
FOLDS = [("2024-11~2025-02", "2023-12", "2024-05", "2024-06", "2024-11", "2025-02"),
         ("2025-03~2025-06", "2024-12", "2025-01", "2025-02", "2025-03", "2025-06"),
         ("2025-09~2025-12", "2025-06", "2025-07", "2025-08", "2025-09", "2025-12"),
         ("2026-03~2026-06", "2025-12", "2026-01", "2026-02", "2026-03", "2026-06")]
BASE = "단순 기준"
GRADES = ["낮음", "보통", "높음"]
CUTS = (0.3, 0.5)
SHOWN = ["PR-AUC", "ROC-AUC", "Brier", "보정 Brier", "평균 점수", "보정 평균"]


def fold_split(period, train_end, val_start, val_end, test_start, test_end):
    split = np.full(len(period), -1, dtype=np.int8)
    split[period <= train_end] = TRAIN
    split[(period >= val_start) & (period <= val_end)] = VAL
    split[(period >= test_start) & (period <= test_end)] = TEST
    return split


def recent_rate(history, offset, fallback):
    """예측 시점에 알려진 같은 편명 이력의 지연률. 이력이 없으면 fallback."""
    known = np.isfinite(history)
    count = known.sum(axis=1)
    late = (np.nan_to_num(history, nan=-1e9) - offset >= 15) & known
    return np.where(count > 0, late.sum(axis=1) / np.maximum(count, 1), fallback)


def grades(prob):
    return np.select([prob >= CUTS[1], prob >= CUTS[0]], [GRADES[2], GRADES[1]], default=GRADES[0])


def score(y, raw, val_y, val_raw):
    """보정은 검증 구간으로만 맞춘다."""
    calibrated = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(val_raw, val_y).predict(raw)
    level = grades(calibrated)
    row = {"건수": int(len(y)), "지연률": float(y.mean()), "PR-AUC": float(average_precision_score(y, raw)), "ROC-AUC": float(roc_auc_score(y, raw)),
           "Brier": float(brier_score_loss(y, np.clip(raw, 0, 1))), "보정 Brier": float(brier_score_loss(y, calibrated)), "평균 점수": float(raw.mean()), "보정 평균": float(calibrated.mean()),
           "검증 PR-AUC": float(average_precision_score(val_y, val_raw))}
    for prefix, labels in (("", level), ("원점수 ", grades(raw))):
        for g in GRADES:
            hit = labels == g
            row[f"{prefix}{g} 비중"], row[f"{prefix}{g} 지연률"] = float(hit.mean()), float(y[hit].mean()) if hit.any() else None
        row[f"{prefix}높음 포착률"] = float(y[labels == GRADES[2]].sum() / y.sum())
        row[f"{prefix}보통 이상 포착률"] = float(y[labels != GRADES[0]].sum() / y.sum())
    return row


def self_check():
    period = np.array(["2023-12", "2024-05", "2024-07", "2024-11", "2025-03"])
    assert fold_split(period, "2023-12", "2024-05", "2024-06", "2024-11", "2025-02").tolist() == [TRAIN, VAL, -1, TEST, -1], "구간 분할"
    assert fold_split(period, "2024-05", "2024-07", "2024-07", "2024-11", "2025-03").tolist() == [TRAIN, TRAIN, VAL, TEST, TEST], "구간 겹침 없음"
    history = np.array([[20.0, 0.0], [np.nan, np.nan], [np.nan, 30.0]])
    assert recent_rate(history, 0.0, 0.1).tolist() == [0.5, 0.1, 1.0], "알려진 이력만 사용"
    assert recent_rate(np.array([[20.0]]), 23.0, 0.1).tolist() == [0.0], "판정 기준 반영"
    assert grades(np.array([0.1, 0.3, 0.49, 0.5])).tolist() == ["낮음", "보통", "보통", "높음"], "등급 경계"


if __name__ == "__main__":
    self_check()
    args = sys.argv[1:]
    direction = args[0] if args else "도착"
    names = args[1:] or MODELS
    data = icn(direction)
    period = np.asarray(data.period, dtype=str)
    y_all = np.asarray(data.y)
    history = np.asarray(data.X)[:, [data.columns.index(f"prev_delay_{i}") for i in range(1, 6)]]
    offset = float(data.meta["label"].split()[2])
    rows = []
    started = time.time()
    for fold in FOLDS:
        split = fold_split(period, *fold[1:])
        val, test = np.flatnonzero(split == VAL), np.flatnonzero(split == TEST)
        y, val_y = y_all[test], y_all[val]
        fallback = y_all[split == TRAIN].mean()
        print(f"=== {fold[0]}: 학습 {(split == TRAIN).sum():,}, 검증 {len(val):,}, 평가 {len(test):,}, 지연률 {y.mean():.3f}", flush=True)
        rows.append({"구간": fold[0], "모델": BASE, "걸린 시간": 0, **score(y, recent_rate(history[test], offset, fallback), val_y, recent_rate(history[val], offset, fallback))})
        base_pr = rows[-1]["PR-AUC"]
        for name in names:
            start = time.time()
            predict = train_one(replace(data, split=split), "binary", name)
            rows.append({"구간": fold[0], "모델": MODEL_NAMES[name], "걸린 시간": round(time.time() - start), **score(y, predict(test), val_y, predict(val))})
            print(f"--- {fold[0]} {MODEL_NAMES[name]}: PR-AUC {rows[-1]['PR-AUC']:.3f} (기준 {base_pr:.3f}), {rows[-1]['걸린 시간']}초, 누적 {(time.time() - started) / 60:.1f}분", flush=True)
    frame = pd.DataFrame(rows)
    pd.set_option("display.width", 240, "display.max_columns", 30)
    print("\n=== 구간별 PR-AUC")
    print(frame.pivot(index="구간", columns="모델", values="PR-AUC").loc[[f[0] for f in FOLDS]].round(3).to_string())
    print("\n=== 모델별 평균 (4개 구간)")
    print(frame.groupby("모델")[[*SHOWN, "낮음 지연률", "보통 지연률", "높음 지연률", "높음 비중", "걸린 시간"]].mean().sort_values("PR-AUC", ascending=False).round(3).to_string())
    print("\n=== 구간별 원점수 등급: 실제 지연률, 비중, 포착률")
    print(frame[["구간", "모델", "지연률", "평균 점수", "원점수 낮음 지연률", "원점수 보통 지연률", "원점수 높음 지연률", "원점수 높음 비중", "원점수 높음 포착률", "원점수 보통 이상 포착률"]].round(3).to_string(index=False))
    path = Store("korea_ops").dir / f"icn_eval_{direction}.json"
    path.write_text(json.dumps({"direction": direction, "folds": FOLDS, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {path}, 총 {(time.time() - started) / 60:.1f}분")
