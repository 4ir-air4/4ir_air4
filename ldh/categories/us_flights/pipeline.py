"""BTS START_YEAR 이후 통계. 모델은 ldh/scripts/train.py, ldh/scripts/train_enhanced.py 가 학습한 결과를 읽어 온다 (여기서 재학습하지 않음)."""
import json
from pathlib import Path

import pandas as pd

from ldh.core.data import REF, START_YEAR
from ldh.core.ml import model_row

DATA = Path("ldh/datasets/processed/bts_parquet")
RECENT_FROM = 2023
CAUSES = {"CarrierDelay": "항공사", "WeatherDelay": "날씨", "NASDelay": "관제(NAS)", "SecurityDelay": "보안", "LateAircraftDelay": "앞편 지연"}
COLS = ["Year", "Month", "DayOfWeek", "CRSDepTime", "Reporting_Airline", "Origin", "Dest", "Cancelled", "Diverted",
        "ArrDel15", "ArrDelay", "CancellationCode", *CAUSES]
DIMENSIONS = {"month": ["Month"], "dow": ["DayOfWeek"], "hour": ["hour"], "airline": ["Reporting_Airline"], "origin": ["Origin"],
              "route": ["Origin", "Dest"]}
STEP_LABELS = {"schedule": "일정", "congestion": "+혼잡", "prev_leg": "+앞편 지연", "weather": "+기상", "aircraft": "+기체", "full": "전체 특성"}


def counts(d, keys):
    """비율은 합칠 수 없으므로 합계만 모은다. 지연은 실제 운항편(ArrDel15 있음)만 기준."""
    g = d.groupby(keys, observed=True)
    return pd.DataFrame({"flights": g.size(), "cancelled": g.Cancelled.sum(), "diverted": g.Diverted.sum(),
                         "operated": g.ArrDel15.count(), "delayed": g.ArrDel15.sum(), "delay_sum": g.ArrDelay.sum(),
                         "delay_n": g.ArrDelay.count()}).reset_index()


def finish(parts, keys):
    t = pd.concat(parts).groupby(keys, as_index=False, observed=True).sum()
    t["delay_rate"] = t.delayed / t.operated.where(t.operated > 0)
    t["cancel_rate"] = t.cancelled / t.flights
    t["avg_delay"] = t.delay_sum / t.delay_n.where(t.delay_n > 0)
    return t.drop(columns=["delay_sum", "delay_n"])


def model_rows():
    rows = []
    for f in sorted(Path("ldh/results/legacy_models/metrics").glob("*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        rows.append(from_train_metrics(m, f"일정, {m['train_years'][0]}~{m['train_years'][1]} 표본"))
    enhanced = [json.loads(f.read_text(encoding="utf-8")) for f in Path("ldh/results/legacy_models/enhanced/metrics").glob("*.json")]
    for m in sorted(enhanced, key=lambda m: (m["task"], list(STEP_LABELS).index(m["step"]))):  # 특성을 더해 가는 순서
        rows.append(from_train_metrics(m, f"2023~2025, {STEP_LABELS.get(m['step'], m['step'])}", "2023-01~2025-09", "2026-01~08"))
    return rows


def from_train_metrics(m, group, train_p=None, test_p=None):
    train_p = train_p or f"{m['train_years'][0]}~{m['train_years'][1]}"
    test_p = test_p or f"{m['test_year']}-01~08"
    base = {"kind": m["kind"], "n_train": m["n_train"], "n_test": m["n_test"]}
    if m["kind"] == "binary":
        rep = m["report"]["1"]
        result = base | {"roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "baseline_pr_auc": m["positive_rate"],
                         "lift": m["pr_auc"] / m["positive_rate"], "precision": rep["precision"], "recall": rep["recall"]}
    elif m["kind"] == "regression":
        s, b = m["model"], m["baseline_mean"]
        result = base | {"mae": s["mae"], "baseline_mae": b["mae"], "improve": 1 - s["mae"] / b["mae"], "r2": s["r2"]}
    else:
        result = base | {"accuracy": m["accuracy"], "majority_rate": m["majority_rate"], "macro_f1": m["macro_f1"]}
    return model_row(m["title"], result, train_p, test_p, group)


def airline_names():
    of = pd.read_csv(REF / "openflights/airlines.dat", header=None, na_values="\\N", usecols=[1, 3, 7], names=["name", "iata", "active"], dtype="string")
    of = of.dropna(subset=["iata"]).sort_values("active", ascending=False).drop_duplicates("iata")
    return of.set_index("iata").name


def run(store):
    yearly, causes, codes = [], [], []
    recent = {k: [] for k in DIMENSIONS}
    files = [f for f in sorted(DATA.glob("ontime_*.parquet")) if int(f.stem.split("_")[1]) >= START_YEAR]
    if len(files) < 12 * (2026 - START_YEAR):
        raise SystemExit(f"ldh/datasets/processed/bts_parquet/ 에 {START_YEAR}년 이후 parquet 이 부족합니다. python -m ldh.scripts.prepare_data 를 먼저 실행하세요.")
    for f in files:
        d = pd.read_parquet(f, columns=COLS)
        yearly.append(counts(d, ["Year"]))
        causes.append(d.groupby("Year")[list(CAUSES)].sum(min_count=1).reset_index())
        cancelled = d[(d.Cancelled == 1) & d.CancellationCode.notna()]
        if len(cancelled):
            codes.append(cancelled.groupby(["Year", "CancellationCode"]).size().rename("n").reset_index())
        if d.Year.iloc[0] >= RECENT_FROM:
            d["hour"] = d.CRSDepTime // 100
            for name, keys in DIMENSIONS.items():
                part = d[d.Year < 2026] if name == "month" else d  # 2026년은 8월까지뿐이라 월별 비교에서 뺌
                recent[name].append(counts(part, keys))

    yearly = finish(yearly, ["Year"])
    store.save_table("yearly", yearly)
    for name, keys in DIMENSIONS.items():
        t = finish(recent[name], keys)
        if name == "airline":
            t["항공사"] = t.Reporting_Airline.map(airline_names()).fillna(t.Reporting_Airline)
        if name in ("route", "origin"):
            t = t.sort_values("flights", ascending=False).head(50)
        store.save_table(name, t)

    cause = pd.concat(causes).groupby("Year", as_index=False).sum(min_count=1).dropna().rename(columns=CAUSES)
    cause = cause.melt("Year", var_name="원인", value_name="분")
    cause["비중"] = cause.분 / cause.groupby("Year").분.transform("sum")
    store.save_table("cause", cause)
    code = pd.concat(codes).groupby(["Year", "CancellationCode"], as_index=False).n.sum()
    code["사유"] = code.CancellationCode.map({"A": "항공사", "B": "날씨", "C": "관제(NAS)", "D": "보안"})
    code["비중"] = code.n / code.groupby("Year").n.transform("sum")
    store.save_table("cancel_code", code)

    last = yearly[yearly.Year == 2025].iloc[0]
    models = model_rows()
    best = max((m for m in models if m["title"].startswith("도착 지연 여부") and "roc_auc" in m), key=lambda m: m["roc_auc"], default=None)
    worst_hour = store.table("hour").query("flights > 50000").sort_values("delay_rate").iloc[[0, -1]]
    store.save_summary(
        headline=f"{START_YEAR}년 이후 {yearly.flights.sum() / 1e6:,.0f}백만 편을 분석했습니다. 2025년 도착 지연률은 {last.delay_rate:.1%}, 결항률은 {last.cancel_rate:.1%}입니다."
                 + (f" 앞편 지연과 기상을 더한 모델이 ROC-AUC {best['roc_auc']:.3f}로 가장 좋습니다." if best else ""),
        kpis=[("분석한 운항 편수", f"{yearly.flights.sum() / 1e6:,.0f}백만", f"{START_YEAR}-01~2026-08"),
              ("2025년 도착 지연률", f"{last.delay_rate:.1%}", "15분 이상 지연, 실제 운항편 기준"),
              ("2025년 결항률", f"{last.cancel_rate:.1%}"),
              ("2025년 평균 도착 지연", f"{last.avg_delay:.1f}분", "조기 도착은 음수로 합산")],
        models=models,
        notes=[f"지연률이 가장 낮은 출발 시간대는 {int(worst_hour.hour.iloc[0])}시({worst_hour.delay_rate.iloc[0]:.1%}), 가장 높은 시간대는 {int(worst_hour.hour.iloc[1])}시({worst_hour.delay_rate.iloc[1]:.1%})입니다.",
               f"{START_YEAR}년 이전 자료는 쓰지 않았습니다. 2020~2021년은 코로나19로 운항과 지연 패턴이 평소와 다릅니다.",
               "2026년은 8월까지만 있습니다.",
               "지연 시간 회귀는 큰 지연을 거의 맞히지 못합니다. 실제 판단에는 지연 여부 모델이 더 적합합니다."],
        sources=["BTS 정시 운항", "OpenFlights 항공사명"],
    )
