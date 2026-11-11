"""OpenSky START_YEAR 이후 항공편(출발·도착 공항이 확인된 편)의 관측 비행 시간 4개 모델. 표본 추출 없음.

LSTM 입력: 같은 기체의 이전 5개 비행. 기체 식별자는 해시로만 쓰고 결과에 남기지 않는다.
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from categories.world_traffic.pipeline import airports, opensky_paths
from core import bigdata
from core.data import CACHE, START_YEAR
from core.models import TEST, TRAIN, VAL, Data, prev_indices

ROOT = CACHE / f"opensky_{START_YEAR}"
CAT = ["origin", "destination", "typecode"]
EPOCH = pd.Timestamp("2018-01-01", tz="UTC")


def read_counts(path):
    d = pd.read_csv(path, usecols=CAT, dtype="string").dropna(subset=["origin", "destination"])
    return {c: d[c].value_counts() for c in CAT}


def convert(job):
    path, codes, out = job
    d = pd.read_csv(path, usecols=CAT + ["firstseen", "lastseen", "icao24"], dtype="string").dropna(subset=["origin", "destination"])
    first, last = pd.to_datetime(d.firstseen, utc=True), pd.to_datetime(d.lastseen, utc=True)
    y = (last - first).dt.total_seconds() / 60
    keep = y.between(5, 1200).to_numpy()
    d, first, last, y = d[keep], first[keep], last[keep], y[keep]
    oa = airports().set_index("ident")
    lat1, lon1 = np.radians(d.origin.map(oa.latitude_deg).astype(float)), np.radians(d.origin.map(oa.longitude_deg).astype(float))
    lat2, lon2 = np.radians(d.destination.map(oa.latitude_deg).astype(float)), np.radians(d.destination.map(oa.longitude_deg).astype(float))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    x = np.full((len(d), 6), np.nan, dtype=np.float32)
    for i, c in enumerate(CAT):
        code = pd.Categorical(d[c], categories=codes[c]).codes.astype(np.float32)
        code[code < 0] = len(codes[c])
        x[:, i] = code
    x[:, 3], x[:, 4], x[:, 5] = first.dt.hour.to_numpy(), first.dt.day_of_week.to_numpy(), (2 * 6371 * np.arcsin(np.sqrt(a))).to_numpy()
    stem = path.name.split("_")[1][:6]
    np.savez(out / f"{stem}.npz", x=x, y=y.to_numpy(np.float32), t0=((first - EPOCH).dt.total_seconds() / 60).to_numpy(np.int32), t1=((last - EPOCH).dt.total_seconds() / 60).to_numpy(np.int32),
             entity=(pd.util.hash_array(d.icao24.fillna("").to_numpy(dtype=object)) >> 1).astype(np.int64), period=np.full(len(d), int(stem), dtype=np.int32))
    return stem


def build():
    meta = bigdata.load_meta(ROOT)
    if meta:
        return meta
    paths = opensky_paths()
    ROOT.mkdir(parents=True, exist_ok=True)
    print(f"  OpenSky {len(paths)}개월 읽는 중", flush=True)
    totals = {c: pd.Series(dtype="int64") for c in CAT}
    with ProcessPoolExecutor(max_workers=2) as ex:
        for part in ex.map(read_counts, paths):
            for c in CAT:
                totals[c] = totals[c].add(part[c], fill_value=0)
        codes = {c: totals[c].nlargest(250).index.tolist() for c in CAT}
        for stem in ex.map(convert, [(p, codes, ROOT) for p in paths]):
            print(f"  변환 {stem}", flush=True)
    parts = sorted(ROOT.glob("*.npz"))
    n = sum(len(np.load(p)["y"]) for p in parts)
    X, y = bigdata.create(ROOT, "X", (n, 6)), bigdata.create(ROOT, "y", (n,))
    t0, t1 = bigdata.create(ROOT, "t0", (n,), "int32"), bigdata.create(ROOT, "t1", (n,), "int32")
    period, entity = bigdata.create(ROOT, "period", (n,), "int32"), bigdata.create(ROOT, "entity", (n,), "int64")
    at = 0
    for p in parts:
        with np.load(p) as z:  # 열린 채로 두면 Windows에서 아래 unlink가 실패한다
            m = len(z["y"])
            X[at:at + m], y[at:at + m], t0[at:at + m], t1[at:at + m], period[at:at + m], entity[at:at + m] = z["x"], z["y"], z["t0"], z["t1"], z["period"], z["entity"]
        at += m
    bigdata.create(ROOT, "prev", (n, 5), "int32")[:] = prev_indices(np.asarray(entity), np.asarray(t0), 5)
    for p in parts:
        p.unlink()
    meta = {"n": n, "codes": codes}
    bigdata.save_meta(ROOT, meta)
    return meta


def duration():
    meta = build()
    X, y = bigdata.read(ROOT, "X"), bigdata.read(ROOT, "y")
    period = np.asarray(bigdata.read(ROOT, "period"))
    split = np.full(len(y), -1, dtype=np.int8)
    split[period <= 202106] = TRAIN
    split[(period > 202106) & (period <= 202112)] = VAL
    split[period >= 202201] = TEST
    return Data(X=X, columns=CAT + ["hour", "dow", "distance_km"], y=y, split=split, cat_idx=[0, 1, 2], keys={c: X[:, i] for i, c in enumerate(CAT)}, decode={c: meta["codes"][c] + ["OTHER"] for c in CAT},
                period=period, S=np.asarray(y).reshape(-1, 1), prev=bigdata.read(ROOT, "prev"), known_t=np.asarray(bigdata.read(ROOT, "t1")).reshape(-1, 1), pred_t=np.asarray(bigdata.read(ROOT, "t0")),
                train_period=f"{START_YEAR}-01~2021-06", test_period="2022-01~12", group="공항, 기종, 거리, 시각, 같은 기체 이력", note="ADS-B 첫·마지막 관측 사이 시간이며 예정 시각이 없어 지연 예측은 아닙니다")


def tasks():
    return [{"task": "opensky_duration", "title": "관측 비행 시간(분)", "kind": "regression", "rows": 55_000_000, "load": duration}]
