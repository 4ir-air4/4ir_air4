"""EUROCONTROL 공항 교통량, 지연, 추가 시간 START_YEAR 이후의 통계와 다음 기간 예측.

엑셀이 커서 처음 한 번은 수십 분 걸린다. DATA 시트를 data/cache/ 에 parquet 으로 저장해 재사용한다.
EUROCONTROL 자료는 비상업 이용 조건이 있어 결과 공개 전 확인이 필요하다.
"""
from concurrent.futures import ProcessPoolExecutor

import openpyxl
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ldh.core.data import CACHE, REF, START_YEAR
from ldh.core.ml import add_lags, binary_result, encode, lag_regression, model_row, self_check

DAILY = ["Airport_Traffic", "Airport_Arrival_ATFM_Delay", "All_Pre-Departure_Delay", "ATC_Pre-Departure_Delay", "En-Route_ATFM_Delay_AUA"]
MONTHLY = ["Taxi-Out_Additional_Time", "Taxi-In_Additional_Time", "ASMA_Additional_Time"]
CAUSE_NAMES = {"A": "사고·장애", "C": "관제 용량", "D": "제빙", "E": "장비(비관제)", "G": "공항 용량", "I": "관제 파업", "M": "공역 관리", "N": "파업(비관제)",
               "O": "기타", "P": "특별 행사", "R": "경로", "S": "관제 인력", "T": "관제 장비", "V": "환경", "W": "날씨", "NA": "미분류"}


def read_sheet(name):
    path = CACHE / f"euro_{name}.parquet"
    if path.exists():
        return str(path)
    wb = openpyxl.load_workbook(REF / f"eurocontrol/{name}.xlsx", read_only=True, data_only=True)
    rows = wb["DATA"].iter_rows(values_only=True)
    head = next(rows)
    d = pd.DataFrame((r for r in rows if r[0] is not None), columns=head)
    wb.close()
    d = d.loc[:, d.columns.notna()]  # 일부 시트는 헤더 뒤에 빈 열이 있음
    d = d.drop(columns=[c for c in d.columns if "label" in c.lower() or c in ("COMMENT", "MONTH_MON", "APT_NAME", "STATE_NAME", "ENTITY_TYPE")])
    d["YEAR"], d["MONTH_NUM"] = d.YEAR.astype(int), d.MONTH_NUM.astype(int)
    for c in d.columns.difference(["FLT_DATE", "APT_ICAO", "ENTITY_NAME"]):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    d.to_parquet(tmp, index=False)
    tmp.replace(path)
    return str(path)


def read_table(name):
    d = pd.read_parquet(CACHE / f"euro_{name}.parquet")
    return d[d.YEAR >= START_YEAR].reset_index(drop=True)


def windows(dates, train_months=24):
    """마지막 6개월을 평가, 그 앞 train_months 개월을 학습 구간으로 쓴다."""
    last = dates.max()
    test_start = (last - pd.DateOffset(months=5)).replace(day=1)
    train_start = test_start - pd.DateOffset(months=train_months)
    return train_start, test_start, last


def daily_model(d, key, target, title, lags=(1, 7)):
    d = d.dropna(subset=[target]).copy()
    d["t"] = d.date.map(pd.Timestamp.toordinal)
    d["x_dow"] = d.date.dt.day_of_week
    train_start, test_start, last = windows(d.date)
    train_range = (train_start.toordinal(), test_start.toordinal() - 1)
    test_range = (test_start.toordinal(), last.toordinal())
    result = lag_regression(d, [key], target, train_range, test_range, list(lags), ["x_dow"])
    return model_row(title, result, f"{train_start:%Y-%m}~{test_start - pd.Timedelta(days=1):%Y-%m}", f"{test_start:%Y-%m}~{last:%Y-%m}", "과거 값, 요일")


def atfm_models(d):
    d = d.copy()
    d["t"], d["x_dow"], d["delay"] = d.date.map(pd.Timestamp.toordinal), d.date.dt.day_of_week, d.DLY_APT_ARR_1.fillna(0)
    train_start, test_start, last = windows(d.date)
    val_start = test_start - pd.DateOffset(months=3)
    d = d[d.date >= train_start - pd.Timedelta(days=7)]
    lagged = add_lags(d[["APT_ICAO", "t", "delay", "x_dow"]], ["APT_ICAO"], "delay", [1, 7])
    lagged["y"] = (lagged.delay > 0).astype(int)
    to_t = pd.Timestamp.toordinal
    tr = lagged[lagged.t.between(to_t(train_start), to_t(val_start) - 1)]
    val = lagged[lagged.t.between(to_t(val_start), to_t(test_start) - 1)]
    te = lagged[lagged.t >= to_t(test_start)]
    X_tr, X_val, X_te = encode(tr, [val, te], ["APT_ICAO"], ["lag1", "lag7", "x_dow"])
    train_p, test_p = f"{train_start:%Y-%m}~{val_start - pd.Timedelta(days=1):%Y-%m}", f"{test_start:%Y-%m}~{last:%Y-%m}"
    title = "공항별 도착 ATFM 지연 발생 여부 (일 단위)"
    rows = [model_row(title, binary_result(X_tr, tr.y, X_val, val.y, X_te, te.y), train_p, test_p, "공항, 전일·전주, 요일")]
    # 단순 기준선: 학습 구간의 공항별 발생 비율만 사용. 공항 구분만으로 얼마나 맞는지 보여 준다.
    score = te.APT_ICAO.map(tr.groupby("APT_ICAO").y.mean()).fillna(0)
    rows.append(model_row(title, {"kind": "binary", "n_train": len(tr), "n_test": len(te), "roc_auc": float(roc_auc_score(te.y, score)),
                                  "pr_auc": float(average_precision_score(te.y, score)), "baseline_pr_auc": float(te.y.mean()),
                                  "lift": float(average_precision_score(te.y, score) / te.y.mean())}, train_p, test_p, "기준선: 공항별 과거 비율"))
    return rows


def monthly_additional(name, title):
    d = read_table(name)
    d["y"] = d.TOTAL_ADD_TIME_MIN / d.TOTAL_REF_NB_FL.where(d.TOTAL_REF_NB_FL > 0)
    d["t"] = d.YEAR * 12 + d.MONTH_NUM
    d = d.dropna(subset=["y"])
    tmax = int(d.t.max())
    label = lambda t: f"{(t - 1) // 12}-{(t - 1) % 12 + 1:02d}"
    model = model_row(title, lag_regression(d, ["APT_ICAO"], "y", (tmax - 29, tmax - 6), (tmax - 5, tmax), [1, 12]), f"{label(tmax - 29)}~{label(tmax - 6)}", f"{label(tmax - 5)}~{label(tmax)}", "전월·전년 값")
    recent = d[d.t > tmax - 12].groupby("APT_ICAO", as_index=False)[["TOTAL_ADD_TIME_MIN", "TOTAL_REF_NB_FL"]].sum()
    recent["추가 시간(분/편)"] = recent.TOTAL_ADD_TIME_MIN / recent.TOTAL_REF_NB_FL
    trend = d.groupby("YEAR", as_index=False)[["TOTAL_ADD_TIME_MIN", "TOTAL_REF_NB_FL"]].sum()
    trend["추가 시간(분/편)"] = trend.TOTAL_ADD_TIME_MIN / trend.TOTAL_REF_NB_FL
    return model, recent[recent.TOTAL_REF_NB_FL > 5000].sort_values("추가 시간(분/편)", ascending=False).head(15)[["APT_ICAO", "추가 시간(분/편)"]], trend[["YEAR", "추가 시간(분/편)"]]


def run(store):
    self_check()
    with ProcessPoolExecutor(max_workers=2) as ex:
        for p in ex.map(read_sheet, DAILY + MONTHLY):
            print("  읽음", p, flush=True)

    tables = {n: read_table(n) for n in DAILY}
    for t in tables.values():
        t["date"] = pd.to_datetime(t.FLT_DATE)
    traffic, atfm, allpre, atc, route = (tables[n] for n in DAILY)
    models = []

    store.save_table("traffic_month", traffic.groupby(traffic.date.dt.to_period("M").astype(str)).FLT_TOT_1.sum().rename("편수").rename_axis("월").reset_index())
    last = traffic.date.max()
    recent = traffic[traffic.date > last - pd.DateOffset(years=1)]
    store.save_table("traffic_top", recent.groupby("APT_ICAO", as_index=False).FLT_TOT_1.sum().sort_values("FLT_TOT_1", ascending=False).head(20).rename(columns={"FLT_TOT_1": "편수"}))
    models.append(daily_model(traffic, "APT_ICAO", "FLT_TOT_1", "공항별 일 운항 편수", lags=(7, 1)))

    g = atfm.groupby(atfm.date.dt.to_period("M").astype(str))
    store.save_table("atfm_month", pd.DataFrame({"도착편당 지연(분)": g.DLY_APT_ARR_1.sum() / g.FLT_ARR_1.sum()}).rename_axis("월").reset_index())
    cause_cols = {c: CAUSE_NAMES.get(c.split("_")[3], c) for c in atfm.columns if c.startswith("DLY_APT_ARR_") and c != "DLY_APT_ARR_1"}
    cause = atfm.groupby("YEAR")[list(cause_cols)].sum().rename(columns=cause_cols).T.groupby(level=0).sum().T.reset_index().melt("YEAR", var_name="원인", value_name="분")
    top = cause.groupby("원인").분.sum().nlargest(6).index
    cause["원인"] = cause.원인.where(cause.원인.isin(top), "그 외")  # 범주 수를 줄여 색 구분이 가능하게 함
    cause = cause.groupby(["YEAR", "원인"], as_index=False).분.sum()
    cause["비중"] = cause.분 / cause.groupby("YEAR").분.transform("sum")
    store.save_table("atfm_cause", cause)
    recent = atfm[atfm.date > last - pd.DateOffset(years=1)].groupby("APT_ICAO", as_index=False)[["DLY_APT_ARR_1", "FLT_ARR_1"]].sum()
    recent["도착편당 지연(분)"] = recent.DLY_APT_ARR_1 / recent.FLT_ARR_1
    store.save_table("atfm_top", recent[recent.FLT_ARR_1 > 5000].sort_values("도착편당 지연(분)", ascending=False).head(15)[["APT_ICAO", "도착편당 지연(분)"]])
    models += atfm_models(atfm)

    allpre["y"] = allpre.DLY_ALL_PRE_2 / allpre.FLT_DEP_IFR_2.where(allpre.FLT_DEP_IFR_2 > 0)
    atc["y"] = atc.DLY_ATC_PRE_2 / atc.FLT_DEP_IFR_2.where(atc.FLT_DEP_IFR_2 > 0)
    route["y"] = route.DLY_ERT_1.fillna(0) / route.FLT_ERT_1.where(route.FLT_ERT_1 > 0)
    frames = []
    for label, t, n, dep in (("전체 사전 지연", allpre, "DLY_ALL_PRE_2", "FLT_DEP_IFR_2"), ("ATC 사전 지연", atc, "DLY_ATC_PRE_2", "FLT_DEP_IFR_2")):
        g = t.dropna(subset=[n, dep]).groupby("YEAR")
        frames.append(pd.DataFrame({"출발편당 지연(분)": g[n].sum() / g[dep].sum(), "지표": label}).reset_index())
    store.save_table("pre_year", pd.concat(frames))
    models.append(daily_model(allpre, "APT_ICAO", "y", "공항별 출발편당 전체 사전 지연(분)"))
    models.append(daily_model(atc, "APT_ICAO", "y", "공항별 출발편당 ATC 사전 지연(분)"))
    models.append(daily_model(route, "ENTITY_NAME", "y", "관제 구역별 항공편당 항로 ATFM 지연(분)"))
    g = route.groupby("YEAR")
    store.save_table("route_year", pd.DataFrame({"항공편당 지연(분)": g.DLY_ERT_1.sum() / g.FLT_ERT_1.sum()}).reset_index())
    rec = route[route.date > last - pd.DateOffset(years=1)].groupby("ENTITY_NAME", as_index=False)[["DLY_ERT_1", "FLT_ERT_1"]].sum()
    rec["항공편당 지연(분)"] = rec.DLY_ERT_1 / rec.FLT_ERT_1
    store.save_table("route_top", rec.sort_values("항공편당 지연(분)", ascending=False).head(15)[["ENTITY_NAME", "항공편당 지연(분)"]])

    trends, tops = [], {}
    for name, title in zip(MONTHLY, ["택시아웃", "택시인", "접근 구역(ASMA)"]):
        model, top, trend = monthly_additional(name, f"공항별 월 {title} 추가 시간(분/편)")
        models.append(model)
        store.save_table(f"add_top_{MONTHLY.index(name)}", top)
        trends.append(trend.assign(지표=title))
    store.save_table("add_year", pd.concat(trends))

    y = last.year - 1
    flights = traffic[traffic.YEAR == y].FLT_TOT_1.sum()
    busiest = store.table("traffic_top").iloc[0]
    atfm_rate = atfm[atfm.YEAR == y].DLY_APT_ARR_1.sum() / atfm[atfm.YEAR == y].FLT_ARR_1.sum()
    store.save_summary(
        headline=f"{y}년 유럽 공항 운항은 {flights / 1e6:,.1f}백만 편(공항별 합계)입니다. 일 단위 지연은 전일 값보다 조금 더 잘 맞히지만, 월 단위 추가 시간은 전월 값과 큰 차이가 없습니다.",
        kpis=[(f"{y}년 운항 합계", f"{flights / 1e6:,.1f}백만", "공항별 출발+도착 합계라 같은 편이 두 번 집계됩니다"),
              ("최다 운항 공항", busiest.APT_ICAO, f"최근 12개월 {busiest.편수:,.0f}편"),
              (f"{y}년 도착 ATFM 지연", f"{atfm_rate:.2f}분/편"), ("자료 끝", f"{last:%Y-%m-%d}")],
        models=models,
        notes=["EUROCONTROL 자료는 출처 표시가 필요하고 상업적 이용이 금지됩니다. 가공물의 공개 가능 여부는 확인이 필요합니다.",
               "운항 합계는 공항별 집계를 더한 값이라 같은 항공편이 출발과 도착 공항에서 두 번 집계됩니다.",
               "도착 ATFM 지연 분류의 높은 ROC-AUC는 대부분 '지연이 잦은 공항'을 구분한 결과입니다. 기준선 행과 비교해서 보세요.",
               "원인 코드의 한글 이름은 EUROCONTROL 코드 체계를 옮긴 것이고, 데이터의 META 시트와 대조하지는 않았습니다."],
        sources=["EUROCONTROL Performance Review Unit"],
    )
