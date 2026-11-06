"""KOSIS 국내항공료 물가지수의 계절성 통계와 월별 예측. 수집 명령은 DATA_SOURCES.md 참고."""
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.data import REF, START_YEAR, read_csv_any
from core.ml import model_row, regression_scores

KOSIS = REF / "kosis"
START = pd.Period(f"{START_YEAR}-01", "M")
COVID = (pd.Period("2020-02", "M"), pd.Period("2022-04", "M"))
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
               "원유는 달러 기준 수입물가지수, 제트유는 생산자물가지수입니다. 실제 유가(달러/배럴)가 아닙니다."],
        sources=["KOSIS 소비자물가조사", "KOSIS 국내선 노선별 통계", "KOSIS 생산자물가지수·수입물가지수(한국은행)", "KOSIS 국내 통화 환율(IMF)"],
    )
