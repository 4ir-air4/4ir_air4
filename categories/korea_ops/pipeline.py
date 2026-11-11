"""인천공항 편별 계획·실제 시각, 한국공항공사 지연 데이터의 통계와 지연 예측."""
import json
import zipfile
from pathlib import Path

import pandas as pd

from core.data import REF, START_YEAR, read_csv_any, read_csv_bytes
from core.ml import binary_result, encode, lag_regression, model_row, self_check

KOREA = REF / "korea"
LIVE = Path(__file__).resolve().parents[2] / "data" / "icn_live"  # collect_icn.py 가 쌓는 인천 실시간 자료
CUTOFF = pd.Timedelta(hours=2)
WEATHER_MAX_AGE = pd.Timedelta(hours=3)
SKY = {"CLR": 0, "SKC": 0, "NSC": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4, "VV": 5}
WX = ["tmpf", "sknt", "gust", "vsby", "p01i", "sky_cover", "skyl1", "ts", "sn", "fz", "fg", "ra"]


def delay_minutes(plan, actual):
    """HH:MM 두 시각의 차이(분). 자정을 넘긴 경우를 고려해 -12~+12시간으로 맞춘다."""
    def minutes(s):
        parts = s.astype("string").str.extract(r"^(\d{1,2}):(\d{2})")
        return pd.to_numeric(parts[0], errors="coerce") * 60 + pd.to_numeric(parts[1], errors="coerce")
    return (minutes(actual) - minutes(plan) + 720) % 1440 - 720


def load_flights():
    """반기 파일이 겹치는 구간(2023년 6월 등)은 나중 파일 값을 쓴다. 2023 상반기 도착 파일은 컬럼명이 잘못되어 있어 방향은 파일명으로 판단."""
    parts = []
    for path in sorted(KOREA.glob("인천공항_환승승객현황*.zip")):
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                d = read_csv_bytes(z.read(n))
                assert d.shape[1] == 8, f"{path.name}/{n}: 컬럼 수 {d.shape[1]}"
                d.columns = ["country", "airport", "code", "date", "plan", "actual", "flight", "transfer"]
                parts.append(d.assign(direction="출발" if "출발" in n else "도착", version=path.name))
    d = pd.concat(parts, ignore_index=True)
    d = d.sort_values("version", key=lambda s: s.str.replace(".zip", "_99999999.zip", regex=False))
    d = d.drop_duplicates(["direction", "date", "plan", "code", "flight"], keep="last").drop(columns="version")
    d["delay"] = delay_minutes(d.plan, d.actual)
    d["sched"] = pd.to_datetime(d.date.astype("string") + " " + d.plan.astype("string"), format="%Y-%m-%d %H:%M", errors="coerce")
    d = d.dropna(subset=["delay", "sched"])
    d = d[d.sched.dt.year >= START_YEAR].reset_index(drop=True)
    d["ym"] = d.sched.dt.strftime("%Y-%m")
    d["month"], d["hour"], d["dow"] = d.sched.dt.month, d.sched.dt.hour, d.sched.dt.day_of_week
    d["airline"] = d.flight.astype("string").str[:2]
    d["sched_hour"] = d.groupby(["direction", d.sched.dt.floor("h")]).month.transform("size")
    return d


def load_kac():
    d = read_csv_any(KOREA / "한국공항공사_지연시간_학습데이터.csv")
    d = d[d["연도"] >= START_YEAR].reset_index(drop=True)
    d["t"] = d["연도"] * 12 + d["월"]
    return d


def rksi_weather():
    cols = {"station": "string", "valid": "string", "tmpf": "float32", "sknt": "float32", "gust": "float32",
            "vsby": "float32", "p01i": "float32", "skyc1": "string", "skyl1": "float32", "wxcodes": "string"}
    wx = pd.read_csv(REF / "weather_asos/asos_14airports_2024_2025.csv", usecols=list(cols), dtype=cols, na_values=["M", "T"])
    wx = wx[wx.station == "RKSI"].copy()
    wx["valid"] = pd.to_datetime(wx.valid, format="%Y-%m-%d %H:%M")
    wx["sky_cover"] = wx.skyc1.map(SKY).astype("float32")
    for flag, pattern in {"ts": "TS", "sn": "SN", "fz": "FZ", "fg": "FG|BR", "ra": "RA|DZ"}.items():
        wx[flag] = wx.wxcodes.str.contains(pattern, na=False).astype("float32")
    return wx.drop(columns=["station", "skyc1", "wxcodes"]).drop_duplicates("valid").sort_values("valid")


def with_weather(d):
    """예측 시점(계획 2시간 전) 이전의 가장 최근 관측. 시각은 KST, 기상은 UTC."""
    left = pd.DataFrame({"t": d.sched - pd.Timedelta(hours=9) - CUTOFF}).reset_index(names="row").sort_values("t")
    hit = pd.merge_asof(left, rksi_weather(), left_on="t", right_on="valid", tolerance=WEATHER_MAX_AGE).set_index("row")
    assert (hit.valid.dropna() <= hit.t[hit.valid.notna()]).all(), "예측 시점 이후 기상이 섞임"
    return d.join(hit[WX])


def labels(part, train_mask):
    """출발 실제 시각은 중앙값이 +27분 안팎이라 이륙 시각으로 추정 -> 학습 구간 중앙값(택시 시간)을 뺀 값으로 판정."""
    offset = part.delay[train_mask].median() if part.direction.iloc[0] == "출발" else 0
    return (part.delay - offset >= 15).astype(int)


def flight_models(d):
    rows = []
    cat, num = ["airline", "code", "country"], ["hour", "dow", "month", "sched_hour"]
    for direction, part in d.groupby("direction"):
        tr, val, te = part.ym <= "2024-12", part.ym.between("2025-01", "2025-06"), part.ym.between("2025-07", "2026-06")
        y = labels(part, tr)
        X_tr, X_val, X_te = encode(part[tr], [part[val], part[te]], cat, num)
        rows.append(model_row(f"인천 {direction} 지연 여부", binary_result(X_tr, y[tr], X_val, y[val], X_te, y[te]),
                              f"{START_YEAR}-01~2024-12", "2025-07~2026-06", "일정+혼잡", "2024년 일부 구간 자료 없음"))
    weather = with_weather(d[d.ym.between("2024-11", "2025-12")])
    for direction, part in weather.groupby("direction"):
        tr, val, te = part.ym <= "2025-06", part.ym.between("2025-07", "2025-09"), part.ym >= "2025-10"
        y = labels(part, tr)
        for group, cols in {"일정+혼잡 (기간 맞춤)": num, "일정+혼잡+RKSI 기상": num + WX}.items():
            X_tr, X_val, X_te = encode(part[tr], [part[val], part[te]], cat, cols)
            rows.append(model_row(f"인천 {direction} 지연 여부", binary_result(X_tr, y[tr], X_val, y[val], X_te, y[te]),
                                  "2024-11~2025-06", "2025-10~12", group))
    return rows


def airline_names():
    """IATA 2글자 → 항공사 이름. 인천 실시간 수집 자료의 한글 이름을 먼저 쓰고, 없는 코드는 OpenFlights(2014년) 영문 이름."""
    from categories.us_flights.pipeline import airline_names as openflights
    names = openflights()
    days = sorted(p for p in LIVE.glob("20*") if p.is_dir())
    if days:  # 가장 최근 하루치면 인천 취항 항공사가 거의 다 나온다
        rows = [json.loads(line)["data"] for f in ("arrivals", "departures") for line in open(days[-1] / f"{f}.jsonl", encoding="utf-8")]
        live = pd.DataFrame(rows)[["flightId", "airline"]].dropna()
        names = live.assign(code=live.flightId.str[:2]).groupby("code").airline.agg(lambda s: s.mode().iloc[0]).combine_first(names)
    return names


def airline_monthly(d, names, top=12):
    """2025년 이후 편수 상위 항공사의 월별·방향별 지연률."""
    keep = d[d.ym >= "2025-01"].airline.value_counts().head(top).index
    t = d[d.airline.isin(keep)].groupby(["ym", "airline", "direction"]).agg(편수=("지연", "size"), 지연률=("지연", "mean")).reset_index()
    t.insert(2, "항공사", t.airline.map(names).fillna(t.airline) + " (" + t.airline + ")")
    return t


def run(store):
    self_check()
    assert delay_minutes(pd.Series(["23:50"]), pd.Series(["00:10"]))[0] == 20, "자정 넘긴 지연"
    assert delay_minutes(pd.Series(["00:10"]), pd.Series(["23:50"]))[0] == -20, "자정 전 조기 운항"

    d = load_flights()
    dep_median = d.delay[d.direction == "출발"].median()
    d["지연"] = d.delay - d.direction.map({"출발": dep_median, "도착": 0}) >= 15
    g = d.groupby(["ym", "direction"])
    monthly = pd.DataFrame({"편수": g.size(), "지연률": g.지연.mean(), "중앙값(분)": g.delay.median(), "상위10%(분)": g.delay.quantile(0.9)}).reset_index()
    store.save_table("icn_monthly", monthly)
    recent = d[d.ym >= "2025-01"]
    hour = recent.groupby(["hour", "direction"]).agg(지연률=("지연", "mean"), 편수=("지연", "size")).reset_index()
    store.save_table("icn_hour", hour)
    names = airline_names()
    for name, col in (("icn_airline", "airline"), ("icn_country", "country")):
        t = recent.groupby([col, "direction"]).agg(편수=("지연", "size"), 지연률=("지연", "mean"), 평균지연=("delay", "mean")).reset_index()
        if col == "airline":
            t.insert(1, "항공사", t.airline.map(names))
        keep = recent[col].value_counts().head(30).index
        store.save_table(name, t[t[col].isin(keep)].sort_values("편수", ascending=False))
    store.save_table("icn_airline_monthly", airline_monthly(d, names))

    kac = load_kac()
    kac["ym"] = kac["연도"].astype(str) + "-" + kac["월"].astype(str).str.zfill(2)
    trend = kac.groupby(["ym", "국내_국제"]).agg(지연편=("지연(편)", "sum"), 지연분=("합계지연시간(분)", "sum"), 여객=("여객(명)", "sum")).reset_index()
    trend["여객 천 명당 지연 편수"] = trend.지연편 / (trend.여객 / 1000)
    store.save_table("kac_trend", trend)
    by_hour = kac.groupby(["시간대", "국내_국제"]).agg(지연편=("지연(편)", "sum"), 평균지연=("평균지연시간(분)", "mean")).reset_index()
    store.save_table("kac_hour", by_hour)
    for name, file in (("airport_delay", "한국공항공사_공항별_지연현황"), ("airport_cancel", "한국공항공사_공항별_결항현황")):
        store.save_table(name, read_csv_any(KOREA / f"{file}.csv"))

    models = flight_models(d)
    kac_result = lag_regression(kac, ["국내_국제", "출도착", "시간대"], "지연(편)", (START_YEAR * 12 + 1, 2024 * 12 + 12), (2025 * 12 + 1, 2025 * 12 + 5), [1, 12])
    models.append(model_row("시간대별 지연 편수 (월 단위)", kac_result, f"{START_YEAR}-01~2024-12", "2025-01~05", "전월·전년 값"))

    y2025 = d[d.ym.str.startswith("2025")]
    arr = y2025[y2025.direction == "도착"]
    arr_rate, dep_rate = arr.지연.mean(), y2025[y2025.direction == "출발"].지연.mean()
    best = max((m for m in models if "roc_auc" in m), key=lambda m: m["roc_auc"])
    store.save_summary(
        headline=f"2025년 인천 도착편의 {arr_rate:.1%}가 15분 이상 지연됐습니다. 일정 정보만으로 지연을 가려내는 성능은 ROC-AUC {best['roc_auc']:.3f}이고, RKSI 기상을 더해도 크게 나아지지 않습니다.",
        kpis=[("분석한 인천 운항", f"{len(d):,}편", f"{d.ym.min()}~{d.ym.max()}"),
              ("2025년 도착 지연률", f"{arr_rate:.1%}"), ("2025년 출발 지연률", f"{dep_rate:.1%}", f"출발 중앙값 {dep_median:.0f}분을 뺀 뒤 15분 이상"),
              ("출발 지연 중앙값", f"{dep_median:.0f}분", "게이트 출발이 아니라 이륙 시각으로 추정")],
        models=models,
        notes=["인천 파일은 환승객 통계용이라 전체 운항편이 포함되는지 확인되지 않았습니다.",
               f"출발 '실제 시각'은 분포로 볼 때 이륙 시각으로 추정됩니다. 출발 지연은 중앙값({dep_median:.0f}분)을 뺀 뒤 15분 이상으로 정의했습니다.",
               "2024-01-01~05-15, 2024-07-01~11-05 자료가 없습니다 (원본 공백).",
               "공항별 지연·결항 현황은 원본에 연도 컬럼이 없어 집계 기간을 알 수 없습니다.",
               "RKSI 기상은 2024~2025년만 있어 기상 비교 모델은 학습 기간이 짧습니다.",
               "언론 보도에 따르면 국토교통부 지연 기준이 바뀌어, 연도 간 지연 편수 비교에 주의가 필요합니다."],
        sources=["인천공항 환승승객현황", "한국공항공사 지연시간 학습데이터", "공항별 지연·결항 현황", "IEM ASOS (RKSI)"],
    )
