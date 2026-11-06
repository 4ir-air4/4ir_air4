"""BTS START_YEAR 이후 약 4,300만 편에 4개 모델을 학습한다. 표본 추출 없음.

입력: 일정 정보, 공항 시간대 혼잡도, 같은 기체의 직전 운항(예측 시점 2시간 전에 끝난 것만).
LSTM: 같은 기체의 이전 5개 운항. 결항·결항 사유는 결항편의 기체번호가 일부 비어 정답이 새므로 같은 출발 공항의 이전 5편을 쓴다.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from core import bigdata
from core.data import CACHE, START_YEAR
from core.models import TEST, TRAIN, VAL, Data, prev_indices

ROOT = CACHE / f"bts_{START_YEAR}"
SOURCE = Path("data")
SEQ = 5
CUTOFF_MIN = 120
EPOCH = pd.Timestamp("1980-01-01")
CAT = ["Reporting_Airline", "Origin", "Dest"]
BASE = CAT + ["Year", "Month", "DayOfWeek", "dep_hour", "arr_hour", "CRSElapsedTime", "Distance", "origin_sched_hour", "dest_sched_hour"]
PREV = ["prev_dep_delay", "prev_arr_delay", "turnaround"]
COLUMNS = BASE + PREV
CAUSES = {"CarrierDelay": "항공사", "WeatherDelay": "날씨", "NASDelay": "관제(NAS)", "SecurityDelay": "보안", "LateAircraftDelay": "앞편 지연"}
CANCEL = {"A": "항공사", "B": "날씨", "C": "관제(NAS)", "D": "보안"}
Y = ["ArrDel15", "ArrDelay", "DepDel15", "DepDelay", "Cancelled", "Diverted", "TaxiOut", "TaxiIn", "AirTime", "cancel_code", "delay_cause"]
READ = ["FlightDate", "Year", "Month", "DayOfWeek", "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "Distance", "Tail_Number", *CAT,
        "ArrDel15", "ArrDelay", "DepDel15", "DepDelay", "Cancelled", "Diverted", "TaxiOut", "TaxiIn", "AirTime", "CancellationCode", *CAUSES]


def utc_minutes(df, tz):
    """예정 출발의 UTC 분. 시간대를 모르는 옛 공항은 현지 시각을 그대로 쓴다 (기체 이력 순서에만 영향)."""
    minutes = df.CRSDepTime // 100 * 60 + df.CRSDepTime % 100
    local = pd.to_datetime(df.FlightDate.astype("string")) + pd.to_timedelta(minutes.fillna(0), unit="min")
    utc = local.copy()
    zones = df.Origin.map(tz)
    for zone, idx in zones.groupby(zones).groups.items():
        part = local.loc[idx]
        utc.loc[idx] = part.dt.tz_localize(zone, ambiguous=np.zeros(len(part), bool), nonexistent="shift_forward").dt.tz_convert(None)
    return ((utc - EPOCH).dt.total_seconds() / 60).to_numpy(np.int64)


def convert(job):
    """파일 하나를 디스크 배열의 제 위치에 쓴다."""
    path, offset, codes, root = job
    from download_weather import airport_timezones
    d = pd.read_parquet(path, columns=READ)
    n = len(d)
    X, Yy, S, known = (bigdata.update(root, a) for a in ("X", "Y", "S", "known"))
    sched = utc_minutes(d, airport_timezones())
    arr = sched + np.nan_to_num(d.CRSElapsedTime.to_numpy(float)).astype(np.int64)
    out = np.full((n, len(COLUMNS)), np.nan, dtype=np.float32)
    for i, c in enumerate(CAT):
        code = pd.Categorical(d[c].astype("string"), categories=codes[c]).codes.astype(np.float32)
        code[code < 0] = len(codes[c])
        out[:, i] = code
    hours = {"dep_hour": d.CRSDepTime // 100, "arr_hour": d.CRSArrTime // 100}
    for c in BASE[3:]:
        if c in hours:
            out[:, BASE.index(c)] = hours[c].to_numpy(np.float32)
        elif c in d:
            out[:, BASE.index(c)] = d[c].to_numpy(np.float32)
    hour = sched // 60
    out[:, BASE.index("origin_sched_hour")] = pd.Series(hour).groupby([d.Origin.to_numpy(), hour]).transform("size").to_numpy(np.float32)
    arr_hour = arr // 60
    out[:, BASE.index("dest_sched_hour")] = pd.Series(arr_hour).groupby([d.Dest.to_numpy(), arr_hour]).transform("size").to_numpy(np.float32)
    X[offset:offset + n] = out

    y = np.full((n, len(Y)), np.nan, dtype=np.float32)
    for i, c in enumerate(Y[:9]):
        y[:, i] = d[c].to_numpy(np.float32)
    y[:, 9] = np.where(d.Cancelled == 1, pd.Categorical(d.CancellationCode.astype("string"), categories=list(CANCEL)).codes, -1)
    cause = d[list(CAUSES)]
    ok = (d.ArrDel15 == 1) & cause.notna().all(axis=1) & (cause.sum(axis=1) > 0)
    y[:, 10] = -1
    if ok.any():
        y[ok.to_numpy(), 10] = np.argmax(cause[ok].to_numpy(), axis=1)
    Yy[offset:offset + n] = y

    dep_delay, arr_delay = d.DepDelay.to_numpy(np.float32), d.ArrDelay.to_numpy(np.float32)
    S[offset:offset + n] = np.column_stack([dep_delay, arr_delay])
    # 결항편은 지연값이 없으므로 예정 시각을 '알게 되는 시각'으로 둔다 (값은 NaN 이라 입력에서 가려짐)
    known[offset:offset + n] = np.column_stack([sched + np.nan_to_num(dep_delay).astype(np.int64), arr + np.nan_to_num(arr_delay).astype(np.int64)]).astype(np.int32)
    bigdata.update(root, "sched")[offset:offset + n] = sched.astype(np.int32)
    bigdata.update(root, "sched_arr")[offset:offset + n] = arr.astype(np.int32)
    tail = d.Tail_Number.astype("string")
    # 옛 자료의 'UNKNOW' 등은 실제 기체가 아니다. 그대로 두면 다른 편들이 같은 기체로 묶여 직전 운항 값이 틀어진다
    clean = tail.str.strip().fillna("")
    missing = ~clean.str.fullmatch(r"[A-Za-z0-9-]{2,}").fillna(False) | clean.str.upper().str.startswith("UNK").fillna(False) | clean.str.upper().eq("NONE")
    code = (pd.util.hash_array(tail.fillna("").to_numpy(dtype=object)) >> 1).astype(np.int64)
    code[missing.to_numpy()] = -1
    bigdata.update(root, "tail")[offset:offset + n] = code
    date = pd.to_datetime(d.FlightDate.astype("string"))
    bigdata.update(root, "date")[offset:offset + n] = (date.dt.year * 10000 + date.dt.month * 100 + date.dt.day).to_numpy(np.int32)
    bigdata.update(root, "period")[offset:offset + n] = (d.Year * 100 + d.Month).to_numpy(np.int32)
    return path.name


def build(files=None, root=ROOT):
    """중간에 끊기면 meta.json 이 없어 처음부터 다시 만든다."""
    meta = bigdata.load_meta(root)
    if meta:
        return meta
    files = files or [f for f in sorted(SOURCE.glob("ontime_*.parquet")) if int(f.stem.split("_")[1]) >= START_YEAR]
    missing = [f.name for f in files if "Tail_Number" not in pq.read_schema(f).names]
    if missing:
        raise SystemExit(f"Tail_Number 없는 parquet {len(missing)}개. python prepare_data.py 를 먼저 실행하세요.")
    counts = [pq.read_metadata(f).num_rows for f in files]
    n = sum(counts)
    print(f"  BTS 디스크 배열 만들기: {len(files)}개 파일, {n:,}행", flush=True)
    freq = {c: {} for c in CAT}
    for f in files:
        d = pd.read_parquet(f, columns=CAT)
        for c in CAT:
            for k, v in d[c].value_counts().items():
                freq[c][k] = freq[c].get(k, 0) + int(v)
    codes = {c: [k for k, _ in sorted(freq[c].items(), key=lambda kv: -kv[1])[:250]] for c in CAT}
    bigdata.create(root, "X", (n, len(COLUMNS)))
    bigdata.create(root, "Y", (n, len(Y)))
    bigdata.create(root, "S", (n, 2))
    bigdata.create(root, "known", (n, 2), "int32")
    for name, dtype in (("sched", "int32"), ("sched_arr", "int32"), ("tail", "int64"), ("date", "int32"), ("period", "int32")):
        bigdata.create(root, name, (n,), dtype)
    jobs, offset = [], 0
    for f, c in zip(files, counts):
        jobs.append((f, offset, codes, root))
        offset += c
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i, name in enumerate(ex.map(convert, jobs), 1):
            if i % 12 == 0 or i == len(jobs):
                print(f"  변환 {i}/{len(jobs)} ({name})", flush=True)

    print("  같은 기체·같은 공항의 이전 운항 찾는 중", flush=True)
    sched = np.asarray(bigdata.read(root, "sched"))
    bigdata.create(root, "prev_tail", (n, SEQ), "int32")[:] = prev_indices(np.asarray(bigdata.read(root, "tail")), sched, SEQ)
    origin = np.asarray(bigdata.read(root, "X")[:, COLUMNS.index("Origin")]).astype(np.int64)
    bigdata.create(root, "prev_origin", (n, SEQ), "int32")[:] = prev_indices(origin, sched, SEQ)
    del origin

    print("  직전 운항 특성 계산 중", flush=True)
    X, S, known, sched_arr = bigdata.update(root, "X"), bigdata.read(root, "S"), bigdata.read(root, "known"), bigdata.read(root, "sched_arr")
    prev = bigdata.read(root, "prev_tail")
    pred_all = sched - CUTOFF_MIN
    for start in range(0, n, 5_000_000):
        end = min(n, start + 5_000_000)
        p = np.asarray(prev[start:end, 0])
        has = p >= 0
        clip = np.maximum(p, 0)
        order = np.argsort(clip)  # memmap 은 정렬해 읽어야 빠르다
        s, k, a = np.empty((end - start, 2), np.float32), np.empty((end - start, 2), np.int32), np.empty(end - start, np.int32)
        s[order], k[order], a[order] = S[clip[order]], known[clip[order]], sched_arr[clip[order]]
        pred = pred_all[start:end]
        X[start:end, COLUMNS.index("prev_dep_delay")] = np.where(has & (k[:, 0] <= pred), s[:, 0], np.nan)
        X[start:end, COLUMNS.index("prev_arr_delay")] = np.where(has & (k[:, 1] <= pred), s[:, 1], np.nan)
        X[start:end, COLUMNS.index("turnaround")] = np.where(has, sched[start:end] - a, np.nan)
    X.flush()
    meta = {"n": n, "codes": codes}
    bigdata.save_meta(root, meta)
    return meta


def filters(y):
    """각 대상에 해당하는 행. 운항편 기준 대상은 결항·회항편을 뺀다."""
    cancelled, diverted = np.asarray(y[:, Y.index("Cancelled")]), np.asarray(y[:, Y.index("Diverted")])
    operated = (cancelled == 0) & (diverted == 0)
    not_cancelled = cancelled == 0
    everyone = np.ones(len(y), bool)
    return {"ArrDel15": operated, "ArrDelay": operated, "DepDel15": not_cancelled, "DepDelay": not_cancelled, "Cancelled": everyone,
            "Diverted": not_cancelled, "TaxiOut": not_cancelled, "TaxiIn": operated, "AirTime": operated, "cancel_code": cancelled == 1, "delay_cause": everyone}


def load_task(target, tail_seq=True, classes=None, balanced=False, root=ROOT, train_until=202412, val_until=202512):
    meta = build(root=root)
    X, y_all = bigdata.read(root, "X"), bigdata.read(root, "Y")
    y = y_all[:, Y.index(target)]
    eligible = filters(y_all)[target] & np.isfinite(y)
    if target in ("cancel_code", "delay_cause"):
        eligible &= np.asarray(y) >= 0
    period = np.asarray(bigdata.read(root, "period"))
    split = np.full(len(y), -1, dtype=np.int8)
    split[eligible & (period <= train_until)] = TRAIN
    split[eligible & (period > train_until) & (period <= val_until)] = VAL
    split[eligible & (period > val_until)] = TEST
    features = X if tail_seq else X[:, :len(BASE)]
    decode = {c: meta["codes"][c] + ["OTHER"] for c in CAT}
    keys = {"운항일": bigdata.read(root, "date"), **{c: X[:, CAT.index(c)] for c in CAT}}
    return Data(X=features, columns=COLUMNS if tail_seq else BASE, y=y, split=split, cat_idx=[0, 1, 2], keys=keys, decode=decode, period=period, classes=classes, balanced=balanced,
                S=bigdata.read(root, "S"), prev=bigdata.read(root, "prev_tail" if tail_seq else "prev_origin"), known_t=bigdata.read(root, "known"), pred_t=np.asarray(bigdata.read(root, "sched")) - CUTOFF_MIN,
                train_period=f"{START_YEAR}-01~2024-12", test_period="2026-01~08", group="일정·혼잡·직전 운항" if tail_seq else "일정·혼잡",
                note="" if tail_seq else "기체번호가 정답을 새게 하므로 기체 이력을 뺐습니다")


def tasks():
    def t(task, title, kind, target, rows, **kw):
        return {"task": task, "title": title, "kind": kind, "rows": rows, "load": lambda: load_task(target, **kw)}
    # 도착 지연 여부를 큰 데이터 중 가장 먼저 끝낸다
    return [t("bts_cancel_code", "결항 사유 (결항된 경우)", "multiclass", "cancel_code", 700_000, tail_seq=False, classes=list(CANCEL.values()), balanced=True),
            t("bts_delay_cause", "주요 지연 원인 (15분 이상 지연된 경우)", "multiclass", "delay_cause", 8_000_000, classes=list(CAUSES.values()), balanced=True),
            t("bts_arr_del15", "도착 지연 여부 (15분 이상)", "binary", "ArrDel15", 43_000_000),
            t("bts_arr_delay", "도착 지연 시간 (분)", "regression", "ArrDelay", 43_000_001),
            t("bts_cancelled", "결항 여부", "binary", "Cancelled", 43_000_002, tail_seq=False, balanced=True),
            t("bts_dep_del15", "출발 지연 여부 (15분 이상)", "binary", "DepDel15", 43_000_003),
            t("bts_dep_delay", "출발 지연 시간 (분)", "regression", "DepDelay", 43_000_004),
            t("bts_diverted", "회항 여부", "binary", "Diverted", 43_000_005, balanced=True),
            t("bts_air_time", "실제 비행 시간 (분)", "regression", "AirTime", 43_000_006),
            t("bts_taxi_out", "택시아웃 시간 (분)", "regression", "TaxiOut", 43_000_007),
            t("bts_taxi_in", "택시인 시간 (분)", "regression", "TaxiIn", 43_000_008)]
