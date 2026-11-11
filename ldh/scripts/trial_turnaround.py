"""turnaround(같은 기체의 직전 도착 예정 ~ 이번 출발 예정) 처리 방식 비교. 검증 구간 성능으로만 고른다.

BTS 기체번호는 실제 운항 기체라 당일 기체 교체가 짧은 간격으로 드러난다(WORK_LOG 2026-11-11 '피처 누수 점검').
캐시를 다시 만들지 않고 읽을 때 열 값을 바꿔 비교한다: 그대로 / 30분 아래를 30으로 / 빼기(전부 NaN이면 LightGBM이 쓰지 않음).
학습 행은 시드 고정 표본(기본 800만 행), 검증 행은 전부 쓴다. 결과: ldh/results/trial_turnaround.json

사용법: python -m ldh.scripts.trial_turnaround [학습 표본 행 수]
"""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ldh.core.models import TRAIN, VAL, fit_lightgbm, tuned_params

OUT = Path("ldh/results/trial_turnaround.json")
TASKS = [("us_flights", "bts_arr_del15"), ("us_weather", "wx_arr_del15")]
VARIANTS = ("as_is", "clip30", "drop")


class EditedColumn:
    """X 를 읽을 때 한 열만 바꿔 돌려준다 (디스크 캐시는 그대로)."""

    def __init__(self, X, col, variant):
        self.X, self.col, self.variant = X, col, variant
        self.shape = X.shape

    def __getitem__(self, rows):
        a = np.array(self.X[rows], dtype=np.float32)
        if self.variant == "clip30":
            a[:, self.col] = np.where(np.isfinite(a[:, self.col]), np.maximum(a[:, self.col], 30), np.nan)
        elif self.variant == "drop":
            a[:, self.col] = np.nan
        return a


def check():
    X = np.array([[1, 10], [2, -5], [3, np.nan]], dtype=np.float32)
    clip = EditedColumn(X, 1, "clip30")[np.arange(3)]
    assert clip[0, 1] == 30 and clip[1, 1] == 30 and np.isnan(clip[2, 1]) and clip[0, 0] == 1, "30분 자르기, NaN 유지"
    assert np.isnan(EditedColumn(X, 1, "drop")[np.arange(3)][:, 1]).all(), "빼기"
    assert EditedColumn(X, 1, "as_is")[np.arange(3)][1, 1] == -5, "그대로"


def main(sample):
    import importlib
    check()
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for slug, task in TASKS:
        t = next(t for t in importlib.import_module(f"ldh.categories.{slug}.models").tasks() if t["task"] == task)
        d = t["load"]()
        col = d.columns.index("turnaround")
        tr_all, val = np.flatnonzero(d.split == TRAIN), np.flatnonzero(d.split == VAL)
        tr = np.sort(np.random.default_rng(0).choice(tr_all, size=min(sample, len(tr_all)), replace=False))
        y_val = np.asarray(d.y[val])
        for v in VARIANTS:
            key = f"{task}:{v}"
            if key in results and results[key].get("train_rows") == len(tr):
                print(f"{key} 이미 있음, 건너뜀", flush=True)
                continue
            start = time.time()
            predict = fit_lightgbm(replace(d, X=EditedColumn(d.X, col, v)), t["kind"], tr, val, tuned=tuned_params(task))
            p = predict(val)
            results[key] = {"val_pr_auc": float(average_precision_score(y_val, p)), "val_roc_auc": float(roc_auc_score(y_val, p)),
                            "train_rows": len(tr), "val_rows": len(val), "seconds": round(time.time() - start)}
            print(key, results[key], flush=True)
            OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return results


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8_000_000)
