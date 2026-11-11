"""한국 운항 4개 모델: 인천 편별 지연(전체 편), 한국공항공사 시간대별 지연 편수."""
import numpy as np
import pandas as pd

from ldh.categories.korea_ops.pipeline import labels, load_flights, load_kac
from ldh.core.data import START_YEAR
from ldh.core.models import TEST, TRAIN, VAL, Data, prev_indices
from ldh.core.panel import month_label, panel_data


def recent_delays(delay, prev, known_t, pred_t):
    """같은 편명의 직전 지연(최근 순)과 평균, 건수. 예측 시점에 결과가 알려진 운항만 쓴다."""
    known = (prev >= 0) & (known_t[prev] <= pred_t[:, None])
    values = np.where(known, delay[prev], np.nan)
    count = known.sum(axis=1)
    mean = np.divide(np.nansum(values, axis=1), count, out=np.full(len(prev), np.nan), where=count > 0)
    return np.column_stack([values, mean, count]).astype(np.float32)


def window_rate(event_t, late, query_t, minutes):
    """query_t 직전 minutes 분 동안 결과가 알려진 운항의 지연률과 건수. query_t 이후의 운항은 쓰지 않는다."""
    order = np.argsort(event_t, kind="stable")
    t = event_t[order]
    total = np.concatenate([[0.0], np.cumsum(late[order], dtype=np.float64)])
    hi, lo = np.searchsorted(t, query_t, side="right"), np.searchsorted(t, query_t - minutes, side="right")
    count = hi - lo
    return np.divide(total[hi] - total[lo], count, out=np.full(len(count), np.nan), where=count > 0), count


def group_window_rate(event_group, event_t, late, query_group, query_t, minutes):
    rate, count = np.full(len(query_t), np.nan), np.zeros(len(query_t))
    for g in np.unique(query_group):
        q = query_group == g
        e = event_group == g
        rate[q], count[q] = window_rate(event_t[e], late[e], query_t[q], minutes)
    return rate, count


CONTEXT_GROUPS = ["icn", "route", "airline", "flight"]
SELECTED_GROUPS = CONTEXT_GROUPS  # ldh/scripts/evaluate_icn_inputs.py 의 검증 PR-AUC 로 고른 입력
LEAD_MINUTES = 120


def recent_context(flights, part, direction, offsets, groups):
    """예측 시점에 알려진 최근 운항에서 만든 입력. groups: CONTEXT_GROUPS 중 일부. (값 배열 목록, 열 이름 목록)을 돌려준다."""
    base = pd.Timestamp("2000-01-01")
    # 예정보다 LEAD_MINUTES 이상 일찍 끝난 기록은 자기 자신이 입력에 섞일 수 있어 뺀다
    events = flights[flights.delay > -LEAD_MINUTES]
    event_t = ((events.sched - base).dt.total_seconds() / 60).to_numpy(np.int64) + events.delay.to_numpy(np.int64)
    late = (events.delay.to_numpy(np.float64) - events.direction.map(offsets).to_numpy(np.float64) >= 15).astype(np.float64)
    same = (events.direction == direction).to_numpy()
    query_t = ((part.sched - base).dt.total_seconds() / 60).to_numpy(np.int64) - LEAD_MINUTES
    arrays, columns = [], []

    def add(name, values):
        arrays.append(np.asarray(values, dtype=np.float32))
        columns.append(name)

    def add_group(prefix, key, windows, count_window):
        for w in windows:
            rate, count = group_window_rate(events.loc[same, key].astype("string").fillna("").to_numpy(dtype=object), event_t[same], late[same], part[key].astype("string").fillna("").to_numpy(dtype=object), query_t, w)
            add(f"{prefix}_rate_{w}", rate)
            if w == count_window:
                add(f"{prefix}_count_{w}", count)

    if "icn" in groups:
        for w in (60, 180, 360, 1440):
            rate, count = window_rate(event_t[same], late[same], query_t, w)
            add(f"icn_rate_{w}", rate)
            if w == 180:
                add("icn_count_180", count)
        for w in (180, 360):
            add(f"icn_other_rate_{w}", window_rate(event_t[~same], late[~same], query_t, w)[0])
    if "route" in groups:
        add_group("route", "code", (1440, 10080), 10080)
    if "airline" in groups:
        add_group("airline", "airline", (1440, 10080), None)
    if "flight" in groups:
        add_group("flight", "flight", (43200,), 43200)
    return arrays, columns


def icn(direction, groups=None):
    flights = load_flights()
    d = flights[flights.direction == direction].sort_values("sched").reset_index(drop=True)
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
    groups = SELECTED_GROUPS if groups is None else groups
    if groups:
        dep = flights[flights.direction == "출발"]
        offsets = {"도착": 0.0, "출발": float(dep.delay[dep.ym <= "2024-12"].median())}
        extra, names = recent_context(flights, d, direction, offsets, groups)
        arrays += extra
        columns += names
    return Data(X=np.column_stack(arrays).astype(np.float32), columns=columns, y=y, split=split, cat_idx=[0, 1, 2],
                keys={"운항일": d.date.astype("string").to_numpy(), "편명": d.flight.astype("string").to_numpy(), "공항": d.code.astype("string").to_numpy(), "계획": d.plan.astype("string").to_numpy()},
                period=d.ym.to_numpy(), S=delay.reshape(-1, 1), prev=prev, known_t=known_t.reshape(-1, 1), pred_t=pred_t,
                train_period=f"{START_YEAR}-01~2024-12", test_period="2025-07~2026-06", group="일정·혼잡·같은 편명 이력",
                meta={"encoders": encoders, "label": f"delay - {offset:g} >= 15", "lead_minutes": LEAD_MINUTES, "context_groups": list(groups)})


def kac():
    d = load_kac()
    t_min, t_max = int(d.t.min()), int(d.t.max())
    return panel_data(d, ["국내_국제", "출도착", "시간대"], "지연(편)", (t_min, t_max - 11), (t_max - 10, t_max - 5), (t_max - 4, t_max), [1, 12], label=month_label, seq_len=12,
                      group="과거 값", train_period=f"{month_label(t_min)}~{month_label(t_max - 11)}", test_period=f"{month_label(t_max - 4)}~{month_label(t_max)}")


def tasks():
    return [{"task": "icn_arr_delay", "title": "인천 도착 지연 여부", "kind": "binary", "rows": 770_000, "load": lambda: icn("도착")},
            {"task": "icn_dep_delay", "title": "인천 출발 지연 여부", "kind": "binary", "rows": 770_001, "load": lambda: icn("출발")},
            {"task": "kac_delay_hour", "title": "시간대별 지연 편수 (월 단위)", "kind": "regression", "rows": 5_900, "load": kac}]
