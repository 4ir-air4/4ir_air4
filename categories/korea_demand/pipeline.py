"""인천 운송실적, 한국공항공사 기종별 실적, 제주 일별 여객, 국토교통부 연간 실적의 통계와 수요 예측."""
import re

import pandas as pd

from core.data import REF, START_YEAR, read_csv_any, to_number
from core.ml import lag_regression, model_row, self_check

KOREA = REF / "korea"


def pick(d, prefix):
    hits = [c for c in d.columns if c.strip().startswith(prefix)]
    assert hits, f"{prefix} 로 시작하는 컬럼 없음: {list(d.columns)}"
    return d[hits[0]]


def version_key(path):
    """파일명 끝 날짜. 날짜가 없으면 최신판."""
    m = re.search(r"_(\d{8})\.csv$", path.name)
    return m.group(1) if m else "99999999"


def load_incheon():
    """같은 달이 여러 판에 있으면(상반기판과 연간판) 가장 최신 판 한 벌만 쓴다. 판을 합치면 상반기가 두 번 집계된다."""
    parts = []
    for path in sorted(KOREA.glob("인천공항_항공사별_노선별_운송실적*.csv"), key=version_key):
        d = read_csv_any(path)
        if "년월" in d:
            ym = d["년월"].astype(str).str.extract(r"(\d{4})\D*(\d{1,2})")
        else:
            ym = pd.concat([d["년도"].astype(str).str.extract(r"(\d{4})")[0], d["월"].astype(str).str.extract(r"(\d{1,2})")[0]], axis=1, keys=[0, 1])
        ok = ym.notna().all(axis=1)  # 2023년 판에 완전히 빈 행이 있음
        d, ym = d[ok], ym[ok]
        new = "공항" in d  # 2025년 판부터 국가·운항(편)이 생기고 경유지공항이 공항으로 바뀜
        parts.append(pd.DataFrame({
            "year": ym[0].astype(int), "month": ym[1].astype(int), "airline": pick(d, "항공사(ICAO)"), "name": pick(d, "항공사명"),
            "country": d["국가"] if new else pd.NA, "airport": d["공항"] if new else d["경유지공항"],
            "direction": pick(d, "도착_출발"), "charter": pick(d, "정기_부정기"),
            "flights": to_number(d["운항(편)"]) if new else float("nan"),
            "passengers": to_number(pick(d, "유임승객")), "version": version_key(path)}))
    d = pd.concat(parts, ignore_index=True)
    latest = d.groupby(["year", "month"]).version.transform("max")
    return d[(d.version == latest) & (d.year >= START_YEAR)].drop(columns="version").assign(t=lambda x: x.year * 12 + x.month, ym=lambda x: x.year.astype(str) + "-" + x.month.astype(str).str.zfill(2))


def load_types():
    """최신판(2025년)만 쓴다. 2021~2023년 판은 공항 4곳뿐이고 코드 체계가 다르며, 2024년 판은 월이 모두 1(연간 합계)이라 합칠 수 없다."""
    d = read_csv_any(KOREA / "한국공항공사_기종별_운송실적.csv").rename(columns={"운항(편)": "flights", "여객(명)": "passengers"})
    assert d["월"].nunique() == 12, "최신판이 월별 자료가 아님"
    d["passengers"], d["flights"] = to_number(d.passengers), to_number(d.flights)
    return d.assign(t=d["운항년도"] * 12 + d["월"], ym=d["운항년도"].astype(str) + "-" + d["월"].astype(str).str.zfill(2))


def run(store):
    self_check()
    models = []
    period = lambda t: f"{(t - 1) // 12}-{(t - 1) % 12 + 1:02d}"

    icn = load_incheon()
    assert icn.groupby(["year", "month"]).size().min() > 100, "한 달 행 수가 비정상적으로 적음"
    monthly = icn.groupby("ym", as_index=False)[["flights", "passengers"]].sum()
    store.save_table("icn_monthly", monthly)
    last12 = icn[icn.t > icn.t.max() - 12]
    for name, keys in (("icn_country", ["country"]), ("icn_airline", ["name"])):
        store.save_table(name, last12.groupby(keys, as_index=False)[["flights", "passengers"]].sum().sort_values("passengers", ascending=False).head(25))
    keys = ["airline", "airport", "direction", "charter"]
    panel = icn[icn.passengers > 0].groupby(keys + ["t"], as_index=False).passengers.sum()
    tmax = int(panel.t.max())
    models.append(model_row("인천 항공사·공항별 월 여객", lag_regression(panel, keys, "passengers", (int(panel.t.min()) + 12, tmax - 12), (tmax - 11, tmax), [1, 12]),
                            f"{period(int(panel.t.min()) + 12)}~{period(tmax - 12)}", f"{period(tmax - 11)}~{period(tmax)}", "전월·전년 값"))

    types = load_types()
    store.save_table("type_top", types[types.t > types.t.max() - 12].groupby("기종", as_index=False)[["flights", "passengers"]].sum().sort_values("passengers", ascending=False).head(20))
    store.save_table("type_airport", types[types.t > types.t.max() - 12].groupby("공항", as_index=False)[["flights", "passengers"]].sum().sort_values("passengers", ascending=False))
    store.save_table("type_monthly", types.groupby("ym", as_index=False)[["flights", "passengers"]].sum())
    keys = ["공항", "항공사", "기종", "구분"]
    panel = types.groupby(keys + ["t"], as_index=False).passengers.sum()
    tmax = int(panel.t.max())
    models.append(model_row("공항·항공사·기종별 월 여객", lag_regression(panel, keys, "passengers", (int(panel.t.min()) + 1, tmax - 3), (tmax - 2, tmax), [1]),
                            f"{period(int(panel.t.min()) + 1)}~{period(tmax - 3)}", f"{period(tmax - 2)}~{period(tmax)}", "전월 값", "월별 자료가 2025년뿐이라 학습 기간이 짧습니다"))

    jeju = read_csv_any(KOREA / "한국공항공사_제주공항_여객_학습데이터.csv")
    jeju.columns = ["date", "arr", "dep", "y"]
    date = pd.to_datetime(jeju.date)
    jeju = jeju.assign(all="제주", t=date.map(pd.Timestamp.toordinal), x_dow=date.dt.day_of_week, x_month=date.dt.month)
    store.save_table("jeju_daily", jeju.assign(날짜=date.dt.strftime("%Y-%m-%d"))[["날짜", "y"]])
    start = pd.Timestamp("2025-01-01").toordinal()
    models.append(model_row("제주공항 일별 여객", lag_regression(jeju, ["all"], "y", (int(jeju.t.min()), start - 1), (start, int(jeju.t.max())), [7, 1, 364], ["x_dow", "x_month"]),
                            "2023-01~2024-12", "2025-01~10", "요일·전주·전일·전년 값"))

    nat = read_csv_any(KOREA / "국토교통부_연도별_국내국제_항공수송실적.csv")
    nat = pd.DataFrame({"연도": nat["연도"], "구분": nat["국내외구분"], "여객": to_number(pick(nat, "여객(명)")), "화물(톤)": to_number(pick(nat, "화물(톤)")), "운항": to_number(pick(nat, "운항(편)"))})
    nat = nat[nat["연도"] >= START_YEAR]
    store.save_table("national", nat)
    jy = read_csv_any(KOREA / "제주도_제주공항_항공수송실적.csv")
    jy = jy[jy["연도별"] >= START_YEAR]
    store.save_table("jeju_yearly", jy.groupby("연도별", as_index=False)[["국내선 여객 (명)", "국제선 여객 (명)"]].sum())
    ap = read_csv_any(KOREA / "한국공항공사_공항별_여객실적.csv")
    ap["여객"] = ap[["여객성인도착", "여객성인출발", "여객유아도착", "여객유아출발"]].apply(to_number).sum(axis=1)
    ap["공항"] = ap["공항"].str.split("_").str[0]
    store.save_table("airport_yearly", ap.groupby(["연도", "공항", "노선"], as_index=False).여객.sum().sort_values("여객", ascending=False))

    n25 = nat[nat["연도"] == nat["연도"].max()].set_index("구분")
    y = int(nat["연도"].max())
    peak = monthly.sort_values("passengers").iloc[-1]
    store.save_summary(
        headline=f"{y}년 국제선 여객은 {n25.loc['국제', '여객'] / 1e6:,.1f}백만 명, 국내선은 {n25.loc['국내', '여객'] / 1e6:,.1f}백만 명입니다. 월별 수요는 전월·전년 값이 이미 강한 예측값이라 모델 효과는 제한적입니다.",
        kpis=[(f"{y}년 국제선 여객", f"{n25.loc['국제', '여객'] / 1e6:,.1f}백만"), (f"{y}년 국내선 여객", f"{n25.loc['국내', '여객'] / 1e6:,.1f}백만"),
              ("인천 최대 월 여객", f"{peak.passengers / 1e6:,.2f}백만", f"{peak.ym}, 유임승객 기준"), ("인천 자료 기간", f"{icn.ym.min()}~{icn.ym.max()}")],
        models=models,
        notes=["인천 운송실적은 판이 겹치는 달을 최신 판 한 벌로만 집계했습니다.",
               "기종별 실적은 2025년 판만 사용했습니다. 2021~2023년 판은 공항 4곳뿐이고, 2024년 판은 월 값이 모두 1이라(연간 합계로 보임) 월별로 합칠 수 없습니다.",
               "2020~2022년은 코로나19로 수요가 크게 줄어 모델 학습에 영향을 줍니다."],
        sources=["인천 노선별 운송실적", "한국공항공사 기종별 실적·공항별 여객·제주 일별 여객", "국토교통부 수송실적", "제주도 수송실적"],
    )
