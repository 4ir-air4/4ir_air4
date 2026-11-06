"""일정 정보에 혼잡도, 앞편 지연, 기상, 기체 정보를 더한 지연·결항 모델 (train.py 모델과 앱은 그대로 둡니다).

예측 시점은 예정 출발 CUTOFF 전이며, 그때 알 수 있는 값만 특성으로 씁니다.
학습 2023-01~2025-09(월별 표본), 임계값 검증 2025-10~12, 평가 2026 전체.
준비: python prepare_data.py 2023 -> python download_weather.py
사용법: python train_enhanced.py [task ...]   인자 없으면 ALL_TASKS 전부
결과: models/enhanced/<task>.joblib, models/enhanced/metrics/<task>__<단계>.json
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from download_weather import REF, airport_idents, airport_timezones, stations
from train import CAT, MAX_CATS, MODEL_DIR, TASKS, binary_metrics, features, load, regression_metrics

CUTOFF = pd.Timedelta(hours=2)
WEATHER_MAX_AGE = pd.Timedelta(hours=3)  # 이보다 오래된 관측은 없는 것으로 처리
MIN_TURNAROUND = 30  # 분
SAMPLE_FRAC = 0.25  # 학습 기간 약 2,000만 행에서 월마다 25% -> 약 500만 행 (기존 모델과 비슷한 규모)
OUT_DIR = MODEL_DIR / "enhanced"
ALL_TASKS = ["arr_del15", "dep_del15", "arr_delay", "dep_delay", "cancelled"]

WX = ["tmpf", "sknt", "gust", "vsby", "p01i", "sky_cover", "skyl1", "ts", "sn", "fz", "fg", "ra"]
CONGESTION = ["origin_sched_hour", "dest_sched_hour", "origin_runways", "dest_runways"]
PREV_LEG = ["prev_dep_delay", "prev_arr_delay", "turnaround"]
WEATHER = [f"{side}_{c}" for side in ("origin", "dest") for c in WX]
AIRCRAFT = ["aircraft_type", "aircraft_age"]
STEPS = {"schedule": [], "congestion": CONGESTION, "prev_leg": CONGESTION + PREV_LEG,
         "weather": CONGESTION + PREV_LEG + WEATHER, "aircraft": CONGESTION + PREV_LEG + WEATHER + AIRCRAFT}
SKY = {"CLR": 0, "SKC": 0, "NSC": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4, "VV": 5}


def steps_for(task):
    """결항편은 기체번호가 일부 비어 있어(운항편은 0%) 기체 기반 특성을 쓰면 정답이 샌다."""
    if task == "cancelled":
        return {"schedule": [], "full": CONGESTION + WEATHER}
    if task == "arr_del15":  # 대표 task만 단계별 기여도를 본다
        return STEPS
    return {"schedule": [], "full": STEPS["aircraft"]}


def add_times(df, tz):
    minutes = df.CRSDepTime // 100 * 60 + df.CRSDepTime % 100
    local = (pd.to_datetime(df.FlightDate) + pd.to_timedelta(minutes, unit="min")).astype("datetime64[ns]")
    utc = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    zones = df.Origin.map(tz)
    for zone, idx in zones.groupby(zones).groups.items():
        part = local.loc[idx]
        utc.loc[idx] = part.dt.tz_localize(zone, ambiguous=np.zeros(len(part), bool),
                                           nonexistent="shift_forward").dt.tz_convert(None).astype("datetime64[ns]")
    df["sched_dep"] = utc
    df["sched_arr"] = utc + pd.to_timedelta(df.CRSElapsedTime, unit="min")
    return df


def add_prev_leg(df):
    """같은 기체의 직전 운항편. 실제 시각 = 예정 시각 + 지연이며, 예측 시점 전에 끝난 사건만 값으로 남긴다."""
    legs = df.loc[(df.Cancelled == 0) & df.Tail_Number.notna(),
                  ["Tail_Number", "sched_dep", "sched_arr", "DepDelay", "ArrDelay"]]
    legs = legs.assign(Tail_Number=legs.Tail_Number.astype("category")).sort_values(["Tail_Number", "sched_dep"])
    prev = legs.groupby("Tail_Number", observed=True)[["sched_dep", "sched_arr", "DepDelay", "ArrDelay"]].shift(1)
    cutoff = legs.sched_dep - CUTOFF
    dep_known = prev.sched_dep + pd.to_timedelta(prev.DepDelay, unit="min") <= cutoff
    arr_known = prev.sched_arr + pd.to_timedelta(prev.ArrDelay, unit="min") <= cutoff
    df["prev_dep_delay"] = prev.DepDelay.where(dep_known)
    df["prev_arr_delay"] = prev.ArrDelay.where(arr_known)
    # BTS 기체번호는 실제 운항 기체라 30분 미만(2026년 지연율 73%)은 지연으로 기체를 바꾼 흔적이다
    df["turnaround"] = ((legs.sched_dep - prev.sched_arr).dt.total_seconds() / 60).clip(lower=MIN_TURNAROUND)
    return df


def add_congestion(df):
    for side, col, when in (("origin", "Origin", "sched_dep"), ("dest", "Dest", "sched_arr")):
        hour = df[when].dt.floor("h").rename("hour")
        df[f"{side}_sched_hour"] = df.groupby([df[col], hour]).DayOfWeek.transform("size")
    runways = pd.read_csv(REF / "ourairports/runways.csv", usecols=["airport_ident", "closed"])
    counts = runways[runways.closed == 0].groupby("airport_ident").size()
    by_iata = airport_idents().map(counts)
    df["origin_runways"], df["dest_runways"] = df.Origin.map(by_iata), df.Dest.map(by_iata)
    return df


def load_weather():
    cols = {"station": "string", "valid": "string", "tmpf": "float32", "sknt": "float32", "gust": "float32",
            "vsby": "float32", "p01i": "float32", "skyc1": "string", "skyl1": "float32", "wxcodes": "string"}
    wx = pd.concat(pd.read_csv(f, usecols=list(cols), dtype=cols, na_values=["M", "T"])
                   for f in sorted((REF / "weather_asos").glob("all_*.csv")))
    wx["valid"] = pd.to_datetime(wx.valid, format="%Y-%m-%d %H:%M").astype("datetime64[ns]")
    wx["sky_cover"] = wx.skyc1.map(SKY).astype("float32")
    for flag, pattern in {"ts": "TS", "sn": "SN", "fz": "FZ", "fg": "FG|BR", "ra": "RA|DZ"}.items():
        wx[flag] = wx.wxcodes.str.contains(pattern, na=False).astype("float32")
    return wx.drop(columns=["skyc1", "wxcodes"]).drop_duplicates(["station", "valid"]).sort_values("valid")


def add_weather(df, wx, ids):
    """예측 시점 이전의 가장 최근 관측만 붙인다. 예보는 없어 예측 시점 이후 날씨 변화는 반영되지 않는다."""
    for side, col in (("origin", "Origin"), ("dest", "Dest")):
        left = pd.DataFrame({"station": df[col].map(ids).astype("string"), "t": df.sched_dep - CUTOFF}).dropna()
        hit = pd.merge_asof(left.sort_values("t").reset_index(names="row"), wx, left_on="t", right_on="valid",
                            by="station", tolerance=WEATHER_MAX_AGE, direction="backward")
        assert (hit.valid.dropna() <= hit.t[hit.valid.notna()]).all(), "예측 시점 이후 기상이 섞임"
        hit = hit.set_index("row")
        for c in WX:
            df[f"{side}_{c}"] = hit[c].reindex(df.index)
    return df


def add_aircraft(df):
    air = pd.read_csv(REF / "opensky/aircraftDatabase.csv", usecols=["registration", "typecode", "built"], dtype="string")
    air = air.dropna(subset=["registration"]).drop_duplicates("registration").set_index("registration")
    built = pd.to_datetime(air.built, errors="coerce").dt.year
    df["aircraft_type"] = df.Tail_Number.map(air.typecode)
    age = df.Year - df.Tail_Number.map(built)
    df["aircraft_age"] = age.where(age >= 0)  # 등록번호 재사용으로 제작 연도가 운항 뒤인 경우 제외
    return df


def model_input(df, cats, cols):
    X = features(df, cats)
    for c in cols:
        if c == "aircraft_type":
            X[c] = pd.Categorical(df[c].where(df[c].isin(cats[c]), "OTHER").fillna("OTHER"), categories=cats[c] + ["OTHER"])
        else:
            X[c] = df[c].astype("float32")
    return X


def run(task, step, cols, train, val, test, save):
    title, kind, target, class_weight = TASKS[task]
    y_tr, y_val, y_te = target(train).dropna(), target(val).dropna(), target(test).dropna()
    rows = train.loc[y_tr.index]
    names = CAT + ["aircraft_type"] if "aircraft_type" in cols else CAT
    cats = {c: rows[c].value_counts().index[:MAX_CATS].tolist() for c in names}
    X_tr, X_val, X_te = (model_input(d.loc[y.index], cats, cols) for d, y in ((train, y_tr), (val, y_val), (test, y_te)))
    params = dict(categorical_features="from_dtype", max_iter=500, learning_rate=0.1, max_leaf_nodes=63, random_state=0)
    start = time.time()
    threshold = None
    if kind == "regression":
        model = HistGradientBoostingRegressor(**params).fit(X_tr, y_tr)
        metrics = regression_metrics(model, float(y_tr.mean()), X_te, y_te, test.Reporting_Airline.loc[y_te.index])
        score = {k: round(v, 3) for k, v in metrics["model"].items()} | {"baseline_mae": round(metrics["baseline_mean"]["mae"], 3)}
    else:
        y_tr, y_val, y_te = y_tr.astype(int), y_val.astype(int), y_te.astype(int)
        model = HistGradientBoostingClassifier(class_weight=class_weight, **params).fit(X_tr, y_tr)
        metrics = binary_metrics(model, X_val, y_val, X_te, y_te)
        threshold = metrics["threshold"]
        score = {k: round(metrics[k], 4) for k in ("roc_auc", "pr_auc", "positive_rate")}
    seconds = round(time.time() - start)
    metrics = {"task": task, "title": title, "kind": kind, "step": step, "columns": list(X_tr.columns),
               "class_weight": class_weight, "cutoff_hours": CUTOFF.total_seconds() / 3600,
               "n_train": len(X_tr), "n_val": len(X_val), "n_test": len(X_te), "n_iter": int(model.n_iter_),
               "seconds": seconds, **metrics}
    (OUT_DIR / "metrics" / f"{task}__{step}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if save:
        joblib.dump({"model": model, "cats": cats, "columns": cols, "threshold": threshold, "title": title, "kind": kind,
                     "cutoff_hours": metrics["cutoff_hours"]}, OUT_DIR / f"{task}.joblib")
    print(f"[{task}/{step}] train {len(X_tr):,} / test {len(X_te):,}  {seconds}s  {score}", flush=True)


def self_check():
    t = pd.Timestamp("2025-01-01")
    h = lambda hours: t + pd.Timedelta(hours=hours)
    df = pd.DataFrame({
        "Tail_Number": ["N1", "N1", "N1", "N1"], "Cancelled": [0.0, 0.0, 0.0, 1.0],
        "sched_dep": [h(8), h(12), h(12.75), h(20)], "sched_arr": [h(9), h(12.5), h(14), h(21)],
        "DepDelay": [10.0, 5.0, 0.0, np.nan], "ArrDelay": [5.0, 7.0, 0.0, np.nan]})
    out = add_prev_leg(df)
    assert out.prev_dep_delay[1] == 10 and out.prev_arr_delay[1] == 5, "끝난 앞편은 값이 있어야 함"
    assert out.turnaround[1] == 180, "연결 시간은 예정 값이므로 앞편이 끝나기 전에도 있어야 함"
    assert out.turnaround[2] == MIN_TURNAROUND, "15분 연결은 기체 교체 흔적이므로 잘라야 함"
    assert np.isnan(out.prev_dep_delay[2]) and np.isnan(out.prev_arr_delay[2]), "예측 시점(10시 45분) 뒤에 출발한 앞편이 샘"
    assert np.isnan(out.prev_dep_delay[3]), "결항편은 앞편 특성을 쓰지 않음"
    assert np.isnan(out.prev_dep_delay[0]), "첫 편은 앞편이 없음"


if __name__ == "__main__":
    self_check()
    tasks = sys.argv[1:] or ALL_TASKS
    unknown = set(tasks) - set(ALL_TASKS)
    if unknown:
        raise SystemExit(f"알 수 없는 task: {unknown}. 가능: {ALL_TASKS}")
    (OUT_DIR / "metrics").mkdir(parents=True, exist_ok=True)

    df = load(range(2023, 2027))
    print(f"loaded {len(df):,} rows", flush=True)
    codes = sorted(set(df.Origin) | set(df.Dest))
    df = add_times(df, airport_timezones())
    df = add_aircraft(add_weather(add_congestion(add_prev_leg(df)), load_weather(), stations(codes)))
    new = CONGESTION + PREV_LEG + WEATHER + AIRCRAFT
    print("sched_dep 결측", round(df.sched_dep.isna().mean(), 5))
    print("특성별 값 있는 비율", df.loc[df.Cancelled == 0, new].notna().mean().round(3).to_dict(), flush=True)

    is_train = (df.Year < 2025) | ((df.Year == 2025) & (df.Month < 10))
    train = df[is_train].groupby(["Year", "Month"]).sample(frac=SAMPLE_FRAC, random_state=0)
    val, test = df[(df.Year == 2025) & (df.Month >= 10)], df[df.Year == 2026]
    del df
    print(f"train {len(train):,} / val {len(val):,} / test {len(test):,}", flush=True)
    for task in tasks:
        steps = steps_for(task)
        for step, cols in steps.items():
            run(task, step, cols, train, val, test, save=step == list(steps)[-1])
