"""기상 관측이 있는 2023~2026년 BTS 전체 약 2,500만 편에 4개 모델을 학습한다. 표본 추출 없음.

입력: 일정, 혼잡, 직전 운항, 예측 시점 전 공항 기상, 기체 정보. 기상 자료가 2023년부터라 기간은 여기까지다.
"""
import numpy as np
import pandas as pd

from core import bigdata
from core.data import CACHE
from core.models import TEST, TRAIN, VAL, Data, prev_indices

ROOT = CACHE / "us_weather_full"
Y_COLS = ["ArrDel15", "ArrDelay", "DepDel15", "DepDelay", "Cancelled"]
SEQ = 5


def build():
    meta = bigdata.load_meta(ROOT)
    if meta:
        return meta
    import train_enhanced as enh
    from download_weather import airport_timezones, stations
    from train import load
    print("  2023~2026 BTS와 기상 결합 중", flush=True)
    df = load(range(2023, 2027))
    codes = sorted(set(df.Origin) | set(df.Dest))
    df = enh.add_times(df, airport_timezones())
    df = enh.add_aircraft(enh.add_weather(enh.add_congestion(enh.add_prev_leg(df)), enh.load_weather(), stations(codes)))
    extra = enh.CONGESTION + enh.PREV_LEG + enh.WEATHER + ["aircraft_age"]
    names = enh.CAT + ["aircraft_type"]
    cats = {c: df[c].astype("string").value_counts().index[:enh.MAX_CATS].tolist() for c in names}
    base = enh.features(df, {c: cats[c] for c in enh.CAT})
    n = len(df)
    plain = [c for c in base.columns if c not in enh.CAT]
    all_cols = names + plain + extra
    X = bigdata.create(ROOT, "X", (n, len(all_cols)))
    for i, c in enumerate(names):
        s = df[c].astype("string")
        code = pd.Categorical(s.where(s.isin(cats[c])), categories=cats[c]).codes.astype(np.float32)
        code[code < 0] = len(cats[c])
        X[:, i] = code
    for i, c in enumerate(plain):
        X[:, len(names) + i] = pd.to_numeric(base[c], errors="coerce").to_numpy(np.float32)
    for i, c in enumerate(extra):
        X[:, len(names) + len(plain) + i] = df[c].to_numpy(np.float32)
    y = bigdata.create(ROOT, "Y", (n, len(Y_COLS)))
    for i, c in enumerate(Y_COLS):
        y[:, i] = df[c].to_numpy(np.float32)
    sched = ((df.sched_dep - pd.Timestamp("1980-01-01")).dt.total_seconds() / 60).to_numpy()
    arr = ((df.sched_arr - pd.Timestamp("1980-01-01")).dt.total_seconds() / 60).to_numpy()
    bigdata.create(ROOT, "sched", (n,), "int32")[:] = np.nan_to_num(sched).astype(np.int32)
    S = bigdata.create(ROOT, "S", (n, 2))
    S[:, 0], S[:, 1] = df.DepDelay.to_numpy(np.float32), df.ArrDelay.to_numpy(np.float32)
    known = bigdata.create(ROOT, "known", (n, 2), "int32")
    known[:, 0] = np.nan_to_num(sched + df.DepDelay.fillna(0).to_numpy()).astype(np.int32)
    known[:, 1] = np.nan_to_num(arr + df.ArrDelay.fillna(0).to_numpy()).astype(np.int32)
    tail = pd.util.hash_array(df.Tail_Number.astype("string").fillna("").to_numpy(dtype=object)).astype(np.int64) >> 1
    tail[df.Tail_Number.isna().to_numpy()] = -1
    bigdata.create(ROOT, "prev_tail", (n, SEQ), "int32")[:] = prev_indices(tail, np.nan_to_num(sched), SEQ)
    origin = np.asarray(X[:, enh.CAT.index("Origin")]).astype(np.int64)
    bigdata.create(ROOT, "prev_origin", (n, SEQ), "int32")[:] = prev_indices(origin, np.nan_to_num(sched), SEQ)
    bigdata.create(ROOT, "period", (n,), "int32")[:] = (df.Year * 100 + df.Month).to_numpy(np.int32)
    date = pd.to_datetime(df.FlightDate.astype("string"))
    bigdata.create(ROOT, "date", (n,), "int32")[:] = (date.dt.year * 10000 + date.dt.month * 100 + date.dt.day).to_numpy(np.int32)
    # 결항 모델에서 뺄 열: 기체번호가 결항편에서 일부 비어 직전 운항·기체 정보로 정답이 샌다
    meta = {"columns": all_cols, "codes": cats, "excluded": enh.PREV_LEG + ["aircraft_age", "aircraft_type"], "n": n}
    bigdata.save_meta(ROOT, meta)
    return meta


def load_task(target, balanced=False, no_tail=False):
    meta = build()
    all_cols = meta["columns"]
    X, ys = bigdata.read(ROOT, "X"), bigdata.read(ROOT, "Y")
    y = ys[:, Y_COLS.index(target)]
    cancelled = np.asarray(ys[:, Y_COLS.index("Cancelled")])
    eligible = np.isfinite(np.asarray(y)) & (np.ones(len(y), bool) if target == "Cancelled" else (cancelled == 0))
    period = np.asarray(bigdata.read(ROOT, "period"))
    split = np.full(len(y), -1, dtype=np.int8)
    split[eligible & (period < 202510)] = TRAIN
    split[eligible & (period >= 202510) & (period < 202601)] = VAL
    split[eligible & (period >= 202601)] = TEST
    keep = [i for i, c in enumerate(all_cols) if not (no_tail and c in meta["excluded"])]
    columns = [all_cols[i] for i in keep]
    cats = list(meta["codes"])
    decode = {c: meta["codes"][c] + ["OTHER"] for c in cats[:3]}
    keys = {"운항일": bigdata.read(ROOT, "date"), **{c: X[:, all_cols.index(c)] for c in cats[:3]}}
    return Data(X=np.asarray(X[:, keep]) if no_tail else X, columns=columns, y=y, split=split, cat_idx=[i for i, c in enumerate(columns) if c in cats], keys=keys, decode=decode, period=period, balanced=balanced,
                S=bigdata.read(ROOT, "S"), prev=bigdata.read(ROOT, "prev_origin" if no_tail else "prev_tail"), known_t=bigdata.read(ROOT, "known"),
                pred_t=np.asarray(bigdata.read(ROOT, "sched")) - 120, train_period="2023-01~2025-09", test_period="2026-01~08",
                group="일정·혼잡·기상" if no_tail else "일정·혼잡·직전 운항·기상·기체")


def tasks():
    def t(task, title, kind, target, **kw):
        return {"task": task, "title": title, "kind": kind, "rows": 25_000_001, "load": lambda: load_task(target, **kw)}
    return [t("wx_arr_del15", "도착 지연 여부 (15분 이상)", "binary", "ArrDel15"), t("wx_arr_delay", "도착 지연 시간 (분)", "regression", "ArrDelay"),
            t("wx_dep_del15", "출발 지연 여부 (15분 이상)", "binary", "DepDel15"), t("wx_dep_delay", "출발 지연 시간 (분)", "regression", "DepDelay"),
            t("wx_cancelled", "결항 여부", "binary", "Cancelled", balanced=True, no_tail=True)]
