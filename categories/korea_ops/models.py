"""한국 운항 4개 모델: 인천 편별 지연(전체 편), 한국공항공사 시간대별 지연 편수."""
import numpy as np
import pandas as pd

from categories.korea_ops.pipeline import labels, load_flights, load_kac
from core.data import START_YEAR
from core.models import TEST, TRAIN, VAL, Data, prev_indices
from core.panel import month_label, panel_data


def recent_delays(delay, prev, known_t, pred_t):
    """같은 편명의 직전 지연(최근 순)과 평균, 건수. 예측 시점에 결과가 알려진 운항만 쓴다."""
    known = (prev >= 0) & (known_t[prev] <= pred_t[:, None])
    values = np.where(known, delay[prev], np.nan)
    count = known.sum(axis=1)
    mean = np.divide(np.nansum(values, axis=1), count, out=np.full(len(prev), np.nan), where=count > 0)
    return np.column_stack([values, mean, count]).astype(np.float32)


def icn(direction):
    d = load_flights()
    d = d[d.direction == direction].sort_values("sched").reset_index(drop=True)
    tr, val, te = d.ym <= "2024-12", d.ym.between("2025-01", "2025-06"), d.ym >= "2025-07"
    y = labels(d, tr).to_numpy(np.float32)
    split = np.full(len(d), -1, dtype=np.int8)
    split[tr.to_numpy()], split[val.to_numpy()], split[te.to_numpy()] = TRAIN, VAL, TEST
    cat = ["airline", "code", "country"]
    columns, arrays, encoders = [], [], {}
    for c in cat:
        cats = d.loc[tr, c].astype("string").value_counts().index[:250]
        code = pd.Categorical(d[c].astype("string"), categories=cats).codes.astype(np.float32)
        code[code < 0] = len(cats)
        arrays.append(code)
        columns.append(c)
        encoders[c] = cats.tolist()
    for c in ["hour", "dow", "month", "sched_hour"]:
        arrays.append(d[c].to_numpy(np.float32))
        columns.append(c)
    minutes = ((d.sched - pd.Timestamp("2000-01-01")).dt.total_seconds() / 60).to_numpy(np.int64)
    entity = d.flight.astype("string").fillna("").factorize()[0].astype(np.int64)  # 같은 편명의 이전 운항
    delay = d.delay.to_numpy(np.float32)
    prev = prev_indices(entity, minutes, 5)
    known_t, pred_t = minutes + np.nan_to_num(delay).astype(np.int64), minutes - 120
    # LSTM 시퀀스와 같은 정보를 표 입력으로도 넣어 모든 모델의 입력 정보를 맞춘다
    arrays.append(recent_delays(delay, prev, known_t, pred_t))
    columns += [f"prev_delay_{i}" for i in range(1, 6)] + ["prev_delay_mean", "prev_delay_count"]
    offset = float(d.delay[tr].median()) if direction == "출발" else 0.0
    return Data(X=np.column_stack(arrays).astype(np.float32), columns=columns, y=y, split=split, cat_idx=[0, 1, 2],
                keys={"운항일": d.date.astype("string").to_numpy(), "편명": d.flight.astype("string").to_numpy(), "공항": d.code.astype("string").to_numpy(), "계획": d.plan.astype("string").to_numpy()},
                period=d.ym.to_numpy(), S=delay.reshape(-1, 1), prev=prev, known_t=known_t.reshape(-1, 1), pred_t=pred_t,
                train_period=f"{START_YEAR}-01~2024-12", test_period="2025-07~2026-06", group="일정·혼잡·같은 편명 이력",
                meta={"encoders": encoders, "label": f"delay - {offset:g} >= 15", "lead_minutes": 120})


def kac():
    d = load_kac()
    t_min, t_max = int(d.t.min()), int(d.t.max())
    return panel_data(d, ["국내_국제", "출도착", "시간대"], "지연(편)", (t_min, t_max - 11), (t_max - 10, t_max - 5), (t_max - 4, t_max), [1, 12], label=month_label, seq_len=12,
                      group="과거 값", train_period=f"{month_label(t_min)}~{month_label(t_max - 11)}", test_period=f"{month_label(t_max - 4)}~{month_label(t_max)}")


def tasks():
    return [{"task": "icn_arr_delay", "title": "인천 도착 지연 여부", "kind": "binary", "rows": 770_000, "load": lambda: icn("도착")},
            {"task": "icn_dep_delay", "title": "인천 출발 지연 여부", "kind": "binary", "rows": 770_001, "load": lambda: icn("출발")},
            {"task": "kac_delay_hour", "title": "시간대별 지연 편수 (월 단위)", "kind": "regression", "rows": 5_900, "load": kac}]
