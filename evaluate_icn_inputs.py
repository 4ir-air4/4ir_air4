"""인천 지연 모델의 입력 묶음 비교.  사용법: python evaluate_icn_inputs.py [도착|출발]

LightGBM 으로 입력 묶음별 4개 구간 성능을 잰다. 입력 선택은 검증 PR-AUC 로 하고, 평가 지표는 확인에만 쓴다.
결과는 outputs/korea_ops/icn_inputs_<방향>.json 에 쓴다.
"""
import json
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from categories.korea_ops.models import CONTEXT_GROUPS, icn
from core.models import TEST, VAL, train_one
from core.store import Store
from evaluate_icn import FOLDS, fold_split, score

SETS = {"기존 입력": [], "+인천 최근": ["icn"], "+출발 공항 최근": ["route"], "+항공사 최근": ["airline"], "+편명 장기": ["flight"],
        "+인천·출발 공항": ["icn", "route"], "전체 추가": CONTEXT_GROUPS}
SHOWN = ["검증 PR-AUC", "PR-AUC", "ROC-AUC", "Brier", "원점수 낮음 지연률", "원점수 보통 지연률", "원점수 높음 지연률", "원점수 높음 포착률", "원점수 보통 이상 포착률"]


def columns_for(columns, groups):
    prefixes = tuple(f"{g}_" for g in CONTEXT_GROUPS)
    wanted = tuple(f"{g}_" for g in groups)
    return [i for i, c in enumerate(columns) if not c.startswith(prefixes) or c.startswith(wanted)]


assert columns_for(["airline", "icn_rate_60", "route_rate_1440"], ["route"]) == [0, 2], "입력 묶음 선택"
assert columns_for(["airline", "icn_rate_60"], []) == [0], "기존 입력만 선택"

if __name__ == "__main__":
    direction = sys.argv[1] if len(sys.argv) > 1 else "도착"
    data = icn(direction, groups=CONTEXT_GROUPS)
    print("추가 입력:", data.columns[14:], "결측 비율:", np.round(np.isnan(np.asarray(data.X)[:, 14:]).mean(axis=0), 3).tolist(), flush=True)
    period = np.asarray(data.period, dtype=str)
    y_all = np.asarray(data.y)
    rows, started = [], time.time()
    for set_name, groups in SETS.items():
        keep = columns_for(data.columns, groups)
        X = np.asarray(data.X)[:, keep]
        for fold in FOLDS:
            split = fold_split(period, *fold[1:])
            val, test = np.flatnonzero(split == VAL), np.flatnonzero(split == TEST)
            predict = train_one(replace(data, X=X, columns=[data.columns[i] for i in keep], split=split), "binary", "lightgbm")
            rows.append({"입력": set_name, "구간": fold[0], "입력 수": len(keep), **score(y_all[test], predict(test), y_all[val], predict(val))})
        mean = pd.DataFrame(rows[-len(FOLDS):])[SHOWN].mean()
        print(f"--- {set_name}: 검증 PR-AUC {mean['검증 PR-AUC']:.3f}, 평가 PR-AUC {mean['PR-AUC']:.3f}, 누적 {(time.time() - started) / 60:.1f}분", flush=True)
    frame = pd.DataFrame(rows)
    pd.set_option("display.width", 240, "display.max_columns", 30)
    print("\n=== 입력별 4개 구간 평균")
    print(frame.groupby("입력", sort=False)[SHOWN].mean().round(3).to_string())
    print("\n=== 구간별 검증 PR-AUC")
    print(frame.pivot(index="입력", columns="구간", values="검증 PR-AUC").loc[list(SETS)].round(3).to_string())
    print("\n=== 구간별 평가 PR-AUC")
    print(frame.pivot(index="입력", columns="구간", values="PR-AUC").loc[list(SETS)].round(3).to_string())
    path = Store("korea_ops").dir / f"icn_inputs_{direction}.json"
    path.write_text(json.dumps({"direction": direction, "sets": SETS, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {path}, 총 {(time.time() - started) / 60:.1f}분")
