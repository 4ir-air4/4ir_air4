"""NTSB 사고 START_YEAR 이후의 사망 발생 여부 4개 모델. 사고 뒤의 분류이며 사고 발생 예측이 아니다."""
import numpy as np
import pandas as pd

from ldh.categories.safety.pipeline import load_events
from ldh.core.data import START_YEAR
from ldh.core.models import TEST, TRAIN, VAL, Data, prev_indices


def fatal():
    d = load_events().sort_values(["ev_year", "ev_month"]).reset_index(drop=True)
    y = d.ev_highest_injury.eq("FATL").fillna(False).to_numpy(np.float32)
    split = np.full(len(d), -1, dtype=np.int8)
    split[(d.ev_year <= 2022).to_numpy()] = TRAIN
    split[(d.ev_year == 2023).to_numpy()] = VAL
    split[(d.ev_year >= 2024).to_numpy()] = TEST
    cat = ["ev_state", "light_cond", "wx_cond_basic", "far_part", "acft_category", "homebuilt"]
    arrays = []
    for c in cat:
        cats = d.loc[split == TRAIN, c].astype("string").value_counts().index[:250]
        code = pd.Categorical(d[c].astype("string"), categories=cats).codes.astype(np.float32)
        code[code < 0] = len(cats)
        arrays.append(code)
    hour = pd.to_numeric(d.ev_time, errors="coerce") // 100
    for values in (hour, d.ev_month, d.wind_vel_kts, d.vis_sm, d.num_eng):
        arrays.append(pd.to_numeric(values, errors="coerce").to_numpy(np.float32))
    t = (d.ev_year * 12 + d.ev_month).to_numpy(np.int64)
    entity = d.ev_state.astype("string").fillna("").factorize()[0].astype(np.int64)  # 같은 주의 이전 사고
    return Data(X=np.column_stack(arrays).astype(np.float32), columns=cat + ["hour", "ev_month", "wind_vel_kts", "vis_sm", "num_eng"], y=y, split=split, cat_idx=list(range(len(cat))),
                keys={"연도": d.ev_year.to_numpy(), "주": d.ev_state.astype("string").to_numpy(), "항공기": d.acft_category.astype("string").to_numpy()},
                period=(d.ev_year.astype(str) + "-" + d.ev_month.astype(str).str.zfill(2)).to_numpy(), S=y.reshape(-1, 1), prev=prev_indices(entity, t, 5), known_t=t.reshape(-1, 1), pred_t=t - 1,
                balanced=True, train_period=f"{START_YEAR}~2022", test_period="2024~2026-10", group="사고 환경, 기체 정보", note="사고가 난 뒤의 분류이며 사고 발생 예측이 아닙니다")


def tasks():
    return [{"task": "ntsb_fatal", "title": "NTSB 사고의 사망 발생 여부 (사후 분석)", "kind": "binary", "rows": 8_000, "load": fatal}]
