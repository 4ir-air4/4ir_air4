"""큰 과제(미국 운항·기상·OpenSky)의 LightGBM 설정 고르기. 검증 구간으로만 고른다.

카테고리마다 대표 과제(이진 1개, 회귀 1개)에서 설정 6개를 비교하고, 기본 설정보다 나은 것을
같은 카테고리·같은 종류 과제에 넣는다(ldh/config/tuned_params.json). 드문 사건(결항·회항) 과제는 넣지 않는다.
시간을 줄이려고 학습 400만 행, 검증 200만 행 표본(시드 고정)을 쓴다.

사용법: python -m ldh.scripts.tune_big
"""
import importlib
import json
import time

import numpy as np
from sklearn.metrics import average_precision_score, mean_absolute_error

from ldh.core.models import TRAIN, TUNED_PATH, VAL, fit_lightgbm

REPS = {"us_flights": ["bts_arr_del15", "bts_arr_delay"], "us_weather": ["wx_arr_del15", "wx_arr_delay"], "world_traffic": ["opensky_duration"]}
CONFIGS = {
    "default": None,
    "lr05_l127": {"learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 100},
    "lr05_l255_ff": {"learning_rate": 0.05, "num_leaves": 255, "min_data_in_leaf": 200, "feature_fraction": 0.8},
    "lr05_l127_reg": {"learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 100, "feature_fraction": 0.8,
                      "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0},
    "lr10_l255": {"learning_rate": 0.1, "num_leaves": 255, "min_data_in_leaf": 500},
    "lr03_l255_reg": {"learning_rate": 0.03, "num_leaves": 255, "min_data_in_leaf": 200, "feature_fraction": 0.8, "lambda_l2": 5.0},
}
N_TRAIN, N_VAL = 4_000_000, 2_000_000
RARE = ("cancel", "divert")  # 가중치 학습하는 드문 사건 과제: 대표 과제와 성격이 달라 설정을 넣지 않는다


def score(kind, y, p):
    """높을수록 좋은 점수."""
    return float(average_precision_score(y, p)) if kind == "binary" else -float(mean_absolute_error(y, p))


def main():
    tuned = json.loads(TUNED_PATH.read_text(encoding="utf-8")) if TUNED_PATH.exists() else {}
    rng = np.random.default_rng(0)
    for slug, reps in REPS.items():
        tasks = importlib.import_module(f"ldh.categories.{slug}.models").tasks()
        for rep in reps:
            t = next(t for t in tasks if t["task"] == rep)
            d = t["load"]()
            tr_all, val_all = np.flatnonzero(d.split == TRAIN), np.flatnonzero(d.split == VAL)
            tr = np.sort(rng.choice(tr_all, min(N_TRAIN, len(tr_all)), replace=False))
            val = np.sort(rng.choice(val_all, min(N_VAL, len(val_all)), replace=False))
            y_val = np.asarray(d.y[val])
            scores, start = {}, time.time()
            for name, params in CONFIGS.items():
                s0 = time.time()
                scores[name] = score(t["kind"], y_val, fit_lightgbm(d, t["kind"], tr, val, tuned=params)(val))
                print(f"{rep} {name}: 검증 {abs(scores[name]):.4f} ({time.time() - s0:.0f}초)", flush=True)
            best = max(scores, key=scores.get)
            print(f"{rep}: 고른 설정 {best} (기본 {abs(scores['default']):.4f} -> {abs(scores[best]):.4f})", flush=True)
            if best == "default":
                continue
            kind = t["kind"]
            for task in [x["task"] for x in tasks if x["kind"] == kind and not any(w in x["task"] for w in RARE)]:
                tuned[task] = {"slug": slug, "kind": kind, "metric": "pr_auc" if kind == "binary" else "mae", "params": CONFIGS[best],
                               "chosen_on": rep, "val_default": abs(scores["default"]), "val_tuned": abs(scores[best]),
                               "trials": len(CONFIGS), "seconds": round(time.time() - start), "sample": [len(tr), len(val)]}
            TUNED_PATH.write_text(json.dumps(tuned, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
