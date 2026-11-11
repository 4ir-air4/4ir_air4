"""KOSIS 국내항공료 물가지수의 계절성 통계와 월별 예측. 수집 명령은 DATA_SOURCES.md 참고."""
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.data import COVID, REF, START_YEAR, read_csv_any
from core.ml import model_row, regression_scores

KOSIS = REF / "kosis"
START = pd.Period(f"{START_YEAR}-01", "M")
SHOCK = pd.Period("2026-03", "M")  # 제트유 급등이 시작된 달. 평가 오차를 이 전후로 나눠 본다
TEST_MONTHS = 24
HORIZONS = (1, 3)
SEASONS = {12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄", 6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"}
DRIVERS = {"pax": "국내선 여객", "jet": "제트유", "crude": "원유(달러)", "fx": "원/달러 환율"}


def series(name, **filters):
    d = read_csv_any(KOSIS / f"{name}.csv", dtype={"PRD_DE": str})
    for col, value in filters.items():
        d = d[d[col] == value]
    s = pd.Series(pd.to_numeric(d["DT"], errors="coerce").to_numpy(), index=pd.PeriodIndex(d["PRD_DE"], freq="M")).sort_index()
    assert s.index.is_unique and s.notna().all(), f"{name}: 중복되거나 숫자가 아닌 값이 있음"
    return s


def load_monthly():
    return pd.DataFrame({
        "price": series("국내항공료_지수"), "ppi": series("국내항공여객_생산자물가"), "cpi": series("소비자물가_총지수"),
        "pax": series("국내선_노선별_여객", C1_NM="합계"), "jet": series("제트유_생산자물가"),
        "crude": series("원유_수입물가_달러"), "fx": series("원달러_환율"),
    }).asfreq("M").loc[str(START):]


def load_routes():
    """노선 × 월의 여객과 운항. 노선별 값은 2020-02부터 있고, 출발·도착 공항이 같은 노선(최근 값 없음)과 합계는 뺀다."""
    out = []
    for name in ("국내선_노선별_여객", "국내선_노선별_운항"):
        d = read_csv_any(KOSIS / f"{name}.csv", dtype={"PRD_DE": str})
        ends = d.C1_NM.str.split("-", n=1)
        d = d[(d.C1_NM != "합계") & (ends.str[0] != ends.str[1])]
        table = d.pivot_table(index="PRD_DE", columns="C1_NM", values="DT", aggfunc="sum")
        table.index = pd.PeriodIndex(table.index, freq="M")
        out.append(table.asfreq("M"))
    return out


def route_demand(price_index):
    """노선별 여객 계절 지수와 최근 12개월 규모. price_index: 월(1~12) → 전국 항공료 계절 지수."""
    pax, flights = load_routes()
    last12 = pax.index[-12:]
    recent = pax.loc[last12].sum()
    rows, heat = [], []
    daily = pax.div(pax.index.days_in_month, axis=0)  # 2월이 짧아 비수기로 보이지 않게 하루 평균으로
    for route in recent.sort_values(ascending=False).index:
        ratio = usable(seasonal_ratio(daily[route]))
        if ratio.index.month.nunique() < 12:  # 달마다 한 번 이상 있어야 계절 지수를 만든다
            continue
        idx = month_index(ratio)
        rows.append({"노선": route, "최근 12개월 여객": int(recent[route]), "여객 비중(%)": recent[route] / recent.sum() * 100,
                     "편당 여객": recent[route] / flights.loc[last12, route].sum(), "성수기": f"{idx.idxmax()}월", "비수기": f"{idx.idxmin()}월",
                     "성수기/비수기 배율": idx.max() / idx.min(), "항공료 계절 지수와 상관": idx.corr(price_index), "계절 지수 자료(개월)": len(ratio)})
        heat += [{"노선": route, "월": f"{m:02d}월", "여객 계절 지수": v} for m, v in idx.items()]
    summary = pd.DataFrame(rows)
    assert abs(pd.DataFrame(heat).groupby("노선")["여객 계절 지수"].mean() - 100).max() < 1e-6, "노선별 계절 지수 평균이 100이 아님"
    return summary, pd.DataFrame(heat), f"{last12[0]}~{last12[-1]}"


# 설·추석 당일(손으로 넣은 값). 연휴가 두 달에 걸쳐도 당일이 있는 달로 본다
SEOL = ["2020-01-25", "2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10", "2025-01-29", "2026-02-17"]
CHUSEOK = ["2020-10-01", "2021-09-21", "2022-09-10", "2023-09-29", "2024-09-17", "2025-10-06", "2026-09-25"]
FARES = ["국내항공료", "국제항공료", "시외버스료", "여객선료", "택시료"]  # 교통요금_지수.csv 의 소비자물가 품목 (2020=100)


def year_month(price):
    """연도 × 월 항공료: 그해 평균 대비(%)와 전년 같은 달 대비(%)."""
    s = price[price.index >= START]
    t = pd.DataFrame({"연도": s.index.year.astype(str), "월": [f"{m:02d}월" for m in s.index.month], "지수": s.to_numpy()})
    t["그해 평균 대비(%)"] = t.지수 / t.groupby("연도").지수.transform("mean") * 100 - 100
    t["전년 같은 달 대비(%)"] = (s / price.shift(12).reindex(s.index) * 100 - 100).to_numpy()
    return t


def holiday_effect(ratio):
    """설(1·2월)과 추석(9·10월) 달의 계절 비율(추세 대비 %)을 그 달에 명절이 든 해와 아닌 해로 나눈다."""
    holiday = {pd.Period(day, "M"): name for name, days in (("설", SEOL), ("추석", CHUSEOK)) for day in days}
    rows = []
    for name, months in (("설", (1, 2)), ("추석", (9, 10))):
        for m in months:
            r = ratio[ratio.index.month == m]
            has = np.array([holiday.get(p) == name for p in r.index])
            rows.append({"명절": name, "월": f"{m}월", "명절 든 해(%)": (r[has].mean() - 1) * 100, "명절 없는 해(%)": (r[~has].mean() - 1) * 100,
                         "명절 든 해": ", ".join(str(p.year) for p in r.index[has]), "명절 없는 해": ", ".join(str(p.year) for p in r.index[~has])})
    return pd.DataFrame(rows)


def fare_tables():
    """국내항공료와 다른 교통 요금의 월별 지수(2020=100)와 최근 달 요약."""
    d = read_csv_any(KOSIS / "교통요금_지수.csv", dtype={"PRD_DE": str})
    t = d.pivot_table(index="PRD_DE", columns="C2_NM", values="DT")[FARES]
    t.index = pd.PeriodIndex(t.index, freq="M")
    last = t.index[-1]
    swing = [month_index(usable(seasonal_ratio(t[c]))) for c in FARES]
    summary = pd.DataFrame({"항목": FARES, "최근 지수": t.loc[last].to_numpy(), "2020년 대비(%)": (t.loc[last] - 100).to_numpy(),
                            "전년 같은 달 대비(%)": (t.loc[last] / t.loc[last - 12] * 100 - 100).to_numpy(),
                            "계절 변동 폭(포인트)": [s.max() - s.min() for s in swing], "가장 비싼 달": [f"{s.idxmax()}월" for s in swing]})
    shown = t.loc[str(START):]
    monthly = shown.set_axis(shown.index.astype(str)).rename_axis("월").reset_index().melt(id_vars="월", var_name="항목", value_name="지수")
    return summary, monthly, str(last)


FULL_SERVICE = {"KAL", "AAR"}  # 대한항공, 아시아나항공. 나머지는 저비용 항공사로 묶는다


def load_route_airlines():
    """노선 × 항공사 × 월의 여객과 운항(긴 표). 항공사는 코드로 묶는다(이름은 KOSIS가 최신 이름으로 준다)."""
    parts = []
    for name, col in (("국내선_노선별_항공사별_여객", "여객"), ("국내선_노선별_항공사별_운항", "운항")):
        d = read_csv_any(KOSIS / f"{name}.csv", dtype={"PRD_DE": str})
        ends = d.C1_NM.str.split("-", n=1)
        d = d[(d.C1_NM != "합계") & (d.C3_NM != "계") & (ends.str[0] != ends.str[1])]
        parts.append(d.assign(코드=d.C3_NM.str.split().str[0]).groupby(["C1_NM", "코드", "PRD_DE"]).DT.sum().rename(col))
    d = pd.concat(parts, axis=1).fillna(0).reset_index().rename(columns={"C1_NM": "노선", "PRD_DE": "월"})
    names = read_csv_any(KOSIS / "국내선_노선별_항공사별_여객.csv").C3_NM.drop_duplicates()
    names = dict(zip(names.str.split().str[0], names.str.split(n=1).str[1]))
    return d.assign(항공사=d.코드.map(names), 구분=np.where(d.코드.isin(FULL_SERVICE), "대형", "저비용"))


def airline_tables():
    """항공사별 최근 12개월 요약, 월별 저비용 비중, 노선별 항공사 점유율."""
    d = load_route_airlines()
    months = sorted(d.월.unique())
    recent = d[d.월.isin(months[-12:])]
    a = recent.groupby(["코드", "항공사", "구분"], as_index=False)[["여객", "운항"]].sum()
    a = a[a.여객 > 0].sort_values("여객", ascending=False)
    a["노선 수"] = a.코드.map(recent[recent.여객 > 0].groupby("코드").노선.nunique())
    a["여객 비중(%)"] = a.여객 / a.여객.sum() * 100
    a["편당 여객"] = a.여객 / a.운항
    monthly = d.pivot_table(index="월", columns="구분", values="여객", aggfunc="sum")
    monthly = pd.DataFrame({"월": [f"{m[:4]}-{m[4:]}" for m in monthly.index], "저비용 비중(%)": (monthly.저비용 / monthly.sum(axis=1) * 100).to_numpy()})
    share = recent.groupby(["노선", "항공사"], as_index=False).여객.sum()
    share = share[share.여객 > 0]
    share["점유율(%)"] = share.여객 / share.groupby("노선").여객.transform("sum") * 100
    assert abs(share.groupby("노선")["점유율(%)"].sum() - 100).max() < 1e-6, "노선별 점유율 합이 100이 아님"
    return a, monthly, share, f"{months[-12][:4]}-{months[-12][4:]}~{months[-1][:4]}-{months[-1][4:]}"


def in_covid(index, margin=0):
    return (index >= COVID[0] - margin) & (index <= COVID[1] + margin)


def seasonal_ratio(s):
    """2x12 중심 이동평균 대비 비율. 추세를 빼고 월별 변동만 남긴다."""
    return s / s.rolling(12).mean().rolling(2).mean().shift(-6)


def month_index(ratio):
    """월별 중앙값을 평균 100으로 맞춘 계절 지수."""
    median = ratio.groupby(ratio.index.month).median()
    return median / median.mean() * 100


def usable(ratio):
    # 이동평균이 앞뒤 6개월을 쓰므로 코로나 구간 앞뒤 6개월까지 뺀다
    return ratio[(ratio.index >= START) & ~in_covid(ratio.index, 6)].dropna()


def features(df, h):
    """h개월 뒤를 예측할 때 쓸 값. 자기 값은 h개월 전까지, 외부 지표는 공표가 늦어 h+1개월 전까지만 쓴다."""
    log = np.log(df)
    lag = h + 1

    def change(col, shift, span):
        return log[col].shift(shift) - log[col].shift(shift + span)

    own = pd.DataFrame({"전년 대비 추세": change("price", h, 12), "작년 같은 기간 변화": change("price", 12, h),
                        "코로나 구간": in_covid(df.index).astype(float)}, index=df.index)
    for month in range(1, 13):
        own[f"{month}월"] = (df.index.month == month).astype(float)
    demand = pd.DataFrame({"여객 전년 대비": change("pax", lag, 12)})
    cost = pd.DataFrame({f"{DRIVERS[c]} {name}": change(c, lag, span) for c in ("jet", "crude", "fx") for name, span in (("3개월 변화", 3), ("전년 대비", 12))})
    return {"계절·과거값": own, "+ 여객": own.join(demand), "+ 유가·환율": own.join(cost), "전체": own.join(demand).join(cost)}


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12), cv=TimeSeriesSplit(5)))


def boosting():
    # 데이터가 적어 얕은 트리와 낮은 학습률로 고정한다
    return GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.03, subsample=0.8, min_samples_leaf=5, random_state=0)


def walk_forward(X, y, price, h, test_index, make_model):
    """평가 달마다, 그 달을 예측하는 시점에 이미 확정된 목표값까지만으로 다시 학습한다."""
    out = {}
    for t in test_index:
        known = (X.index >= START) & (X.index <= t - h) & X.notna().all(axis=1).to_numpy() & y.notna().to_numpy()
        model = make_model().fit(X[known], y[known])
        out[t] = price[t - h] * np.exp(model.predict(X.loc[[t]])[0])
    return pd.Series(out)


def baselines(price, h, test_index):
    log = np.log(price)
    return {"직전 값 그대로": price.shift(h).reindex(test_index),
            "작년 같은 달 값": price.shift(12).reindex(test_index),
            "직전 값 + 작년 변화율": np.exp(log.shift(h) + log.shift(12) - log.shift(12 + h)).reindex(test_index)}


def lag_correlation(df):
    """전년 대비 변화율끼리의 시차 상관. 코로나 구간은 뺀다. 자기상관이 커서 유의성 판단에는 쓰지 않는다."""
    log = np.log(df)
    target = (log.price - log.price.shift(12))
    rows = []
    for col, name in DRIVERS.items():
        driver = log[col] - log[col].shift(12)
        for lag in range(0, 7):
            pair = pd.concat([target, driver.shift(lag)], axis=1).dropna()
            pair = pair[(pair.index >= START) & ~in_covid(pair.index, 12)]
            rows.append({"지표": name, "시차(개월)": lag, "상관계수": float(pair.iloc[:, 0].corr(pair.iloc[:, 1])), "행 수": len(pair)})
    return pd.DataFrame(rows)


def self_check():
    index = pd.period_range("2010-01", periods=120, freq="M")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(np.exp(rng.normal(size=(120, 7)) / 10) * 100, index=index, columns=["price", "ppi", "cpi", "pax", "jet", "crude", "fx"])
    for h in HORIZONS:
        t = index[100]
        future = df.copy()
        future.loc[t - h + 1:, "price"] *= 2  # 예측 시점 이후의 항공료
        future.loc[t - h:, ["pax", "jet", "crude", "fx"]] *= 3  # 예측 시점에 아직 공표되지 않은 외부 지표
        before, after = features(df, h)["전체"].loc[t], features(future, h)["전체"].loc[t]
        changed = before.index[(before - after).abs() > 1e-9].tolist()
        # 작년 같은 기간 값은 h < 12 이므로 예측 시점 이전이다
        assert not changed, f"예측 시점 이후 값이 특성에 들어감: {changed}"

    assert all(pd.Period(d, "M").month in (1, 2) for d in SEOL) and all(pd.Period(d, "M").month in (9, 10) for d in CHUSEOK), "명절 날짜의 달"
    pattern = 100 + 10 * np.sin(np.arange(120) * 2 * np.pi / 12)
    got = month_index(seasonal_ratio(pd.Series(pattern, index=index)))
    assert abs(got.mean() - 100) < 1e-9 and got.idxmax() == index[3].month, "계절 지수의 최고 달이 입력과 다름"


def run(store):
    self_check()
    df = load_monthly()
    assert df.price.notna().sum() > 60, "항공료 지수 행이 비정상적으로 적음"
    price = df.price.dropna()

    # 계절성 통계
    ratio, pax_ratio = usable(seasonal_ratio(price)), usable(seasonal_ratio(df.pax.dropna()))
    seasonal = pd.DataFrame({"월": range(1, 13)})
    seasonal["계절"] = seasonal.월.map(SEASONS)
    for name, values in (("항공료 계절 지수", ratio), ("여객 계절 지수", pax_ratio)):
        seasonal[name] = month_index(values).reindex(seasonal.월).to_numpy()
    spread = ratio.groupby(ratio.index.month).agg(lambda v: (v.quantile(.75) - v.quantile(.25)) * 100)
    seasonal["연도별 흔들림(IQR)"] = spread.reindex(seasonal.월).to_numpy()
    seasonal["자료 수"] = ratio.groupby(ratio.index.month).size().reindex(seasonal.월).to_numpy()
    store.save_table("seasonal_month", seasonal)

    by_season = seasonal.groupby("계절", as_index=False)[["항공료 계절 지수", "여객 계절 지수"]].mean()
    by_season = by_season.set_index("계절").loc[["봄", "여름", "가을", "겨울"]].reset_index()
    store.save_table("seasonal_season", by_season)

    groups = [v.to_numpy() for _, v in ratio.groupby(ratio.index.month)]
    p_value = float(kruskal(*groups).pvalue)
    pax_link = float(seasonal["항공료 계절 지수"].corr(seasonal["여객 계절 지수"]))

    shown = df.loc[str(START):].copy()
    shown["실질 항공료"] = shown.price / shown.cpi * 100
    monthly = pd.DataFrame({"월": shown.index.astype(str), "항공료 지수": shown.price, "실질 항공료": shown["실질 항공료"], "국내항공여객 생산자물가": shown.ppi,
                            "여객(명)": shown.pax, "제트유": shown.jet, "원유(달러)": shown.crude, "원/달러 환율": shown.fx}).reset_index(drop=True)
    store.save_table("monthly", monthly)
    both = shown[["price", "ppi"]].dropna()
    ppi_link = float(both.price.corr(both.ppi))

    corr = lag_correlation(df)
    store.save_table("lag_corr", corr)

    store.save_table("price_year_month", year_month(price))
    holidays = holiday_effect(ratio)
    store.save_table("holiday_effect", holidays)
    fare_summary, fare_monthly, fare_last = fare_tables()
    store.save_table("fare_compare", fare_summary)
    store.save_table("fare_monthly", fare_monthly)
    domestic = fare_summary.set_index("항목").loc["국내항공료"]

    routes, route_month, recent_period = route_demand(seasonal.set_index("월")["항공료 계절 지수"])
    store.save_table("route_summary", routes)
    store.save_table("route_month", route_month)
    jeju = routes[routes.노선.str.contains("제주")]
    airlines, lcc_monthly, route_airline, airline_period = airline_tables()
    store.save_table("airline_recent", airlines)
    store.save_table("airline_lcc_monthly", lcc_monthly)
    store.save_table("route_airline", route_airline)
    store.save_table("route_airline_monthly", load_route_airlines().drop(columns=["코드", "구분"]))
    lcc = airlines[airlines.구분 == "저비용"]["여객 비중(%)"].sum()

    # 예측: 최근 24개월을 달마다 다시 학습하며 평가한다
    test_index = price.index[-TEST_MONTHS:]
    period = f"{test_index[0]}~{test_index[-1]}"
    log = np.log(df)
    models, compare, predictions = [], [], pd.DataFrame({"월": test_index.astype(str), "실제": price.reindex(test_index).to_numpy()})
    for h in HORIZONS:
        y = (log.price - log.price.shift(h))
        base = baselines(price, h, test_index)
        mean_mae = float(np.abs(price[test_index] - price[(price.index >= START) & (price.index < test_index[0])].mean()).mean())
        baseline_mae = min(mean_mae, *(float(np.abs(price[test_index] - p).mean()) for p in base.values()))
        sets = features(df, h)
        candidates = {name: (X, ridge) for name, X in sets.items()}
        candidates["전체 (얕은 트리)"] = (sets["전체"], boosting)
        results = dict(base)
        for name, (X, maker) in candidates.items():
            X = X.reindex(df.index)
            results[name] = walk_forward(X, y, price, h, test_index, maker)
            n_train = int(((X.index >= START) & (X.index <= test_index[-1] - h) & X.notna().all(axis=1).to_numpy() & y.notna().to_numpy()).sum())
            scores = regression_scores(price[test_index], results[name])
            models.append(model_row(f"{h}개월 뒤 항공료 지수", {"kind": "regression", "n_train": n_train, "n_test": len(test_index), **scores, "mean_mae": mean_mae,
                                                         "baseline_mae": baseline_mae, "improve": 1 - scores["mae"] / baseline_mae},
                                    f"{START}~ (달마다 재학습)", period, name, "기준선은 학습 평균, 직전 값, 작년 같은 달 값, 직전 값 + 작년 변화율 중 오차가 가장 작은 것입니다."))
        for name, pred in results.items():
            error = (price[test_index] - pred).abs()
            compare.append({"예측 시점": f"{h}개월 전", "모델": name, "종류": "기준선" if name in base else "학습 모델", "전체 MAE": error.mean(),
                            "제트유 급등 전 MAE": error[error.index < SHOCK].mean(), "제트유 급등 후 MAE": error[error.index >= SHOCK].mean()})
            predictions[f"{h}개월 전: {name}"] = pred.reindex(test_index).to_numpy()
    compare = pd.DataFrame(compare)
    store.save_table("model_compare", compare)
    store.save_table("predictions", predictions)

    # 마지막 시점까지 학습한 1개월 뒤 모델의 표준화 계수
    X = features(df, 1)["전체"].reindex(df.index)
    y = log.price - log.price.shift(1)
    known = (X.index >= START) & X.notna().all(axis=1).to_numpy() & y.notna().to_numpy()
    final = ridge().fit(X[known], y[known])
    coef = pd.DataFrame({"특성": X.columns, "표준화 계수": final[-1].coef_}).sort_values("표준화 계수", key=np.abs, ascending=False)
    store.save_table("coefficients", coef)

    # 요약
    top, low = seasonal.loc[seasonal["항공료 계절 지수"].idxmax()], seasonal.loc[seasonal["항공료 계절 지수"].idxmin()]
    one = compare[compare["예측 시점"] == "1개월 전"]
    best_base = one[one.종류 == "기준선"].sort_values("전체 MAE").iloc[0]
    best = one[one.종류 == "학습 모델"].sort_values("전체 MAE").iloc[0]
    gain = 1 - best["전체 MAE"] / best_base["전체 MAE"]
    calm_months = int((test_index < SHOCK).sum())
    calm_base = one[one.종류 == "기준선"]["제트유 급등 전 MAE"].min()
    calm_gain = 1 - best["제트유 급등 전 MAE"] / calm_base
    verdict = (f"1개월 뒤 예측은 '{best.모델}' 모델이 기준선 '{best_base.모델}'보다 오차가 {gain:.0%} 작지만, 제트유 급등 전 {calm_months}개월만 보면 최선의 기준선과 차이가 {calm_gain:.0%}입니다." if gain > 0
               else f"1개월 뒤 예측은 어떤 학습 모델도 기준선 '{best_base.모델}'보다 낫지 않습니다.")
    actual = predictions["실제"]
    wins = int(((actual - predictions[f"1개월 전: {best.모델}"]).abs() < (actual - predictions[f"1개월 전: {best_base.모델}"]).abs()).sum())
    jet = corr[corr.지표 == "제트유"].sort_values("상관계수", ascending=False).iloc[0]
    calm_corr = lag_correlation(df[df.index < SHOCK])
    calm_jet = calm_corr[calm_corr.지표 == "제트유"].sort_values("상관계수", ascending=False).iloc[0]
    store.save_summary(
        headline=f"국내항공료 지수는 {int(top.월)}월이 가장 높고({top['항공료 계절 지수']:.1f}) {int(low.월)}월이 가장 낮습니다({low['항공료 계절 지수']:.1f}). {verdict}",
        kpis=[("가장 비싼 달", f"{int(top.월)}월", f"{START_YEAR}년 이후, 코로나 구간을 뺀 계절 지수. 연평균이 100입니다."),
              ("가장 싼 달", f"{int(low.월)}월", f"{START_YEAR}년 이후, 코로나 구간을 뺀 계절 지수. 연평균이 100입니다."),
              ("1개월 뒤 최선 모델 MAE", f"{best['전체 MAE']:.2f}", f"{best.모델}, 지수 포인트, {period}"),
              ("기준선 MAE", f"{best_base['전체 MAE']:.2f}", best_base.모델),
              ("제트유 상관 최고 시차", f"{int(jet['시차(개월)'])}개월", f"전년 대비 변화율 상관계수 {jet['상관계수']:.2f}")],
        models=models,
        notes=["값은 원화 운임이 아니라 전국 국내항공료 소비자물가지수(2020=100)입니다. 노선, 예약 시점별 가격은 알 수 없습니다.",
               f"월별 차이 Kruskal-Wallis p값은 {p_value:.3g}입니다. 같은 달 값이 연도 간 독립이라는 가정이 정확하지 않아 참고용입니다.",
               f"{START_YEAR}년 이후에서 코로나 구간을 빼면 계절 지수에 쓰는 자료가 {len(ratio)}개월뿐이라 불확실성이 큽니다. 코로나 전후 비교는 {START_YEAR}년 이전 자료가 없어 뺐습니다.",
               f"항공료와 여객 계절 지수의 상관계수는 {pax_link:.2f}입니다. 12개 값으로 계산했습니다.",
               f"국내항공여객 생산자물가와 소비자물가 항공료 지수의 상관계수는 {ppi_link:.2f}입니다.",
               "2020-02~2022-04는 코로나 구간으로 보고 계절 지수와 상관 계산에서 뺐고, 학습에서는 구분 변수로 넣었습니다.",
               f"제트유 상관은 급등 전({SHOCK} 이전)만 보면 시차 {int(calm_jet['시차(개월)'])}개월에서 {calm_jet['상관계수']:.2f}입니다. 시차 상관은 원인을 뜻하지 않으며, 유류할증료 자료를 직접 쓰지 않았습니다.",
               f"평가 {len(test_index)}개월 중 최선 모델이 최선 기준선보다 오차가 작은 달은 {wins}개월입니다. 평가 구간이 짧아 개선 폭의 불확실성이 큽니다.",
               "평가 구간의 제트유 급등은 학습 구간에 비슷한 사례가 드물어 오차를 키웁니다. 급등 전후 오차를 나눠 표시했습니다.",
               "표준화 계수는 서로 상관된 지표끼리 나뉘어 부호를 원인으로 해석하면 안 됩니다.",
               "월별 여객은 달의 일수를 보정하지 않은 합계입니다.",
               f"노선별 수요: 운임은 노선별로 공개되지 않아 노선별 여객의 계절 지수로 보완했습니다. {len(routes)}개 노선이며, 노선별 자료가 2020-02부터라 코로나 구간을 빼면 계절 지수에 쓰는 달이 노선마다 {routes['계절 지수 자료(개월)'].min()}~{routes['계절 지수 자료(개월)'].max()}개월입니다.",
               f"{recent_period} 여객의 {jeju['여객 비중(%)'].sum():.0f}%가 제주 노선입니다. 노선별 계절 지수는 하루 평균 여객으로 만든 수요 지표이고 가격이 아니며, 항공료 계절 지수와의 상관은 12개 값으로 계산했습니다.",
               "명절 효과: 설·추석이 든 달과 같은 달의 다른 해를 비교했습니다. 코로나 구간을 빼면 달마다 3~4개 해뿐이라 크기는 불확실합니다. 명절 날짜는 코드에 직접 넣었습니다.",
               f"다른 교통 요금(소비자물가, 2020=100): {fare_last} 국내항공료는 2020년보다 {domestic['2020년 대비(%)']:.1f}%, 전년 같은 달보다 {domestic['전년 같은 달 대비(%)']:.1f}% 높습니다. 지수끼리 비교이며 금액 비교가 아닙니다.",
               f"항공사별: {airline_period} 국내선 여객의 {lcc:.0f}%가 저비용 항공사입니다(대한항공·아시아나항공 외 전부). 운임은 항공사별로도 공개되지 않아 여객과 운항만 봅니다. 좌석 수가 없어 탑승률 대신 편당 여객을 씁니다.",
               "원유는 달러 기준 수입물가지수, 제트유는 생산자물가지수입니다. 실제 유가(달러/배럴)가 아닙니다."],
        sources=["KOSIS 소비자물가조사", "KOSIS 국내선 노선별 통계", "KOSIS 생산자물가지수·수입물가지수(한국은행)", "KOSIS 국내 통화 환율(IMF)"],
    )
