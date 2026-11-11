"""BTS 2023~2026 표본에 출발 2시간 전 공항 기상을 붙여 날씨별 지연·결항률을 집계한다.

기상 효과 모델은 ldh/scripts/train_enhanced.py 결과를 읽어 온다. 메모리를 아끼려고 연도별로 나눠 처리한다.
"""
import json
from pathlib import Path

import pandas as pd

from ldh.core.data import REF
from ldh.core.ml import model_row
from ldh.core.airports import airport_timezones, stations
from ldh.scripts.train_enhanced import SKY, WX, add_times, add_weather

SAMPLE_FRAC = 0.1
CONDITIONS = {"ts": "뇌우", "sn": "눈", "fz": "결빙", "fg": "안개·박무", "ra": "비"}
BINS = {"vsby": ("시정", [-1, 1, 3, 5, 100], ["1마일 이하", "1~3마일", "3~5마일", "5마일 초과"]),
        "sknt": ("풍속", [-1, 10, 20, 30, 200], ["10노트 이하", "10~20노트", "20~30노트", "30노트 초과"]),
        "tmpf": ("기온", [-100, 32, 60, 85, 150], ["영하(32°F 이하)", "32~60°F", "60~85°F", "85°F 초과"])}
ADVERSE = ["ts", "sn", "fz", "fg"]
READ = {"station": "string", "valid": "string", "tmpf": "float32", "sknt": "float32", "gust": "float32", "vsby": "float32",
        "p01i": "float32", "skyc1": "string", "skyl1": "float32", "wxcodes": "string"}


def read_year(year):
    wx = pd.read_csv(REF / f"weather_asos/all_{year}.csv", usecols=list(READ), dtype=READ, na_values=["M", "T"])
    djt = REF / f"weather_asos/all_{year}_djt.csv"
    if djt.exists():
        wx = pd.concat([wx, pd.read_csv(djt, usecols=list(READ), dtype=READ, na_values=["M", "T"])])
    wx["valid"] = pd.to_datetime(wx.valid, format="%Y-%m-%d %H:%M").astype("datetime64[ns]")
    wx["sky_cover"] = wx.skyc1.map(SKY).astype("float32")
    for flag, pattern in {"ts": "TS", "sn": "SN", "fz": "FZ", "fg": "FG|BR", "ra": "RA|DZ"}.items():
        wx[flag] = wx.wxcodes.str.contains(pattern, na=False).astype("float32")
    return wx.drop(columns=["skyc1", "wxcodes"]).drop_duplicates(["station", "valid"]).sort_values("valid")


def rates(d, label_col):
    g = d.groupby(label_col, observed=True)
    return pd.DataFrame({"flights": g.size(), "operated": g.ArrDel15.count(), "delayed": g.ArrDel15.sum(), "cancelled": g.Cancelled.sum()}).reset_index()


def finish(parts, keys):
    t = pd.concat(parts).groupby(keys, as_index=False, observed=True).sum()
    t["지연률"] = t.delayed / t.operated.where(t.operated > 0)
    t["결항률"] = t.cancelled / t.flights
    return t


def condition(d, side):
    """관측이 있는 편만 본다. 여러 현상이 겹치면 더 심각한 쪽(뇌우>눈>결빙>안개>비)으로 분류."""
    known = d[d[f"{side}_tmpf"].notna() | d[f"{side}_sknt"].notna()].copy()
    label = pd.Series("현상 없음", index=known.index)
    for flag in reversed(list(CONDITIONS)):
        label[known[f"{side}_{flag}"] == 1] = CONDITIONS[flag]
    return known.assign(날씨=label)


def enhanced_rows():
    labels = {"schedule": "일정만", "congestion": "+혼잡", "prev_leg": "+앞편 지연", "weather": "+기상", "aircraft": "+기체", "full": "전체 특성"}
    rows = []
    enhanced = [json.loads(f.read_text(encoding="utf-8")) for f in Path("ldh/results/legacy_models/enhanced/metrics").glob("*.json")]
    for m in sorted(enhanced, key=lambda m: (m["task"], list(labels).index(m["step"]))):  # 특성을 더해 가는 순서
        if m["kind"] != "binary":
            continue
        rep = m["report"]["1"]
        rows.append(model_row(m["title"], {"kind": "binary", "n_train": m["n_train"], "n_test": m["n_test"], "roc_auc": m["roc_auc"],
                                           "pr_auc": m["pr_auc"], "baseline_pr_auc": m["positive_rate"], "lift": m["pr_auc"] / m["positive_rate"],
                                           "precision": rep["precision"], "recall": rep["recall"]},
                              "2023-01~2025-09", "2026-01~08", labels.get(m["step"], m["step"])))
    return rows


def run(store):
    cond, bins, airport, obs = {"출발 공항": [], "도착 공항": []}, [], [], []
    tz = None
    for year in (2023, 2024, 2025, 2026):
        files = sorted(Path("ldh/datasets/processed/bts_parquet").glob(f"ontime_{year}_*.parquet"))
        cols = ["Year", "Month", "FlightDate", "CRSDepTime", "CRSElapsedTime", "Origin", "Dest", "Cancelled", "ArrDel15"]
        df = pd.concat(pd.read_parquet(f, columns=cols).sample(frac=SAMPLE_FRAC, random_state=0) for f in files)
        df = df.reset_index(drop=True)
        tz = airport_timezones() if tz is None else tz
        ids = stations(sorted(set(df.Origin) | set(df.Dest)))
        wx = read_year(year)
        obs.append(pd.DataFrame({"연도": [year], "관측 수": [len(wx)], "관측소 수": [wx.station.nunique()]}))
        df = add_weather(add_times(df, tz), wx, ids)
        del wx
        for side, name in (("origin", "출발 공항"), ("dest", "도착 공항")):
            cond[name].append(rates(condition(df, side), "날씨"))
        for col, (title, edges, labels) in BINS.items():
            part = df.assign(구간=pd.cut(df[f"origin_{col}"], edges, labels=labels))
            bins.append(rates(part.dropna(subset=["구간"]), "구간").assign(항목=title))
        bad = df[[f"origin_{c}" for c in ADVERSE]].eq(1).any(axis=1)
        airport.append(rates(df.assign(악천후=bad.map({True: "악천후", False: "그 외"})), ["Origin", "악천후"]))
        print(f"  {year}: {len(df):,}편 처리", flush=True)

    tables = {}
    for name, parts in cond.items():
        tables[name] = finish(parts, ["날씨"]).assign(공항=name)
    condition_table = pd.concat(tables.values())
    store.save_table("condition", condition_table)
    bin_table = finish(bins, ["항목", "구간"])
    store.save_table("bins", bin_table)
    store.save_table("observations", pd.concat(obs))

    a = finish(airport, ["Origin", "악천후"]).pivot(index="Origin", columns="악천후", values=["flights", "지연률", "결항률"])
    a.columns = [f"{m}_{k}" for m, k in a.columns]
    a = a.reset_index().sort_values("flights_그 외", ascending=False).head(40)
    a["악천후 비중"] = a["flights_악천후"] / (a["flights_악천후"] + a["flights_그 외"])
    store.save_table("airport", a)

    origin = tables["출발 공항"].set_index("날씨")
    clear = origin.loc["현상 없음"]
    worst = origin.drop("현상 없음").sort_values("결항률").iloc[-1]
    models = enhanced_rows()
    store.save_summary(
        headline=f"출발 공항에 뇌우·눈이 있으면 지연이 크게 늘어납니다. {worst.name} 때 결항률은 {worst.결항률:.1%}로 현상이 없을 때({clear.결항률:.1%})의 {worst.결항률 / clear.결항률:.0f}배입니다.",
        kpis=[(f"{k} 시 지연률", f"{origin.loc[k].지연률:.1%}", f"현상 없을 때 {clear.지연률:.1%}") for k in ["뇌우", "눈"]]
             + [("눈 올 때 결항률", f"{origin.loc['눈'].결항률:.1%}"), ("현상 없을 때 지연률", f"{clear.지연률:.1%}")],
        models=models,
        notes=["출발 2시간 전 최근 관측을 썼습니다. 실제 출발 시각의 날씨와 다를 수 있습니다.",
               f"BTS 2023~2026년의 {SAMPLE_FRAC:.0%} 무작위 표본으로 집계했습니다.",
               "상관관계입니다. 날씨 외 공항·계절 차이도 함께 반영되어 있습니다.",
               "2026년 기상은 9월 1일까지 있습니다."],
        sources=["BTS 정시 운항", "Iowa State IEM ASOS", "OurAirports", "OpenFlights 시간대"],
    )
