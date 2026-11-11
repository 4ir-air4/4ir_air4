"""유럽 공항 4개 모델: START_YEAR 이후 일별·월별 교통량과 지연."""
import numpy as np
import pandas as pd

from ldh.categories.europe.pipeline import read_table
from ldh.core.panel import day_label, month_label, panel_data


def _daily(name, key, value_fn, lags, y_fn=None):
    d = read_table(name)
    date = pd.to_datetime(d.FLT_DATE)
    d = d.assign(t=date.map(pd.Timestamp.toordinal), x_dow=date.dt.day_of_week, y=value_fn(d))
    last = date[d.y.notna()].max()
    test_start = (last - pd.DateOffset(months=5)).replace(day=1)
    val_start = test_start - pd.DateOffset(months=6)
    to = pd.Timestamp.toordinal
    calendar = {"x_dow": lambda t: pd.Timestamp.fromordinal(t).day_of_week}
    return panel_data(d, [key], "y", (int(d.t.min()), to(val_start) - 1), (to(val_start), to(test_start) - 1), (to(test_start), to(last)), lags, ["x_dow"], calendar, label=day_label, seq_len=14,
                      y_fn=y_fn, group="과거 값, 요일", train_period=f"{date.min():%Y-%m}~{val_start - pd.Timedelta(days=1):%Y-%m}", test_period=f"{test_start:%Y-%m}~{last:%Y-%m}")


def _monthly(name):
    d = read_table(name)
    d["y"] = d.TOTAL_ADD_TIME_MIN / d.TOTAL_REF_NB_FL.where(d.TOTAL_REF_NB_FL > 0)
    d["t"] = d.YEAR * 12 + d.MONTH_NUM
    # 원본 Taxi-Out 표에 LZIB 2023년 12개월이 같은 값으로 두 번 들어 있다. 겹치면 과거 값 결합에서 행이 불어난다
    d = d.dropna(subset=["y"]).drop_duplicates(["APT_ICAO", "t"], keep="last")
    t_min, t_max = int(d.t.min()), int(d.t.max())
    # 변화량 학습 + 2·3·13달 전 값: trial_weak_tasks 검증 비교에서 세 과제 모두 직전 달 값보다 좋았다 (WORK_LOG 2026-11-11)
    return panel_data(d, ["APT_ICAO"], "y", (t_min, t_max - 12), (t_max - 11, t_max - 6), (t_max - 5, t_max), [1, 2, 3, 12, 13], label=month_label, seq_len=12, group="과거 값 (직전 달 대비 변화 학습)", delta=True,
                      train_period=f"{month_label(t_min)}~{month_label(t_max - 12)}", test_period=f"{month_label(t_max - 5)}~{month_label(t_max)}")


def tasks():
    ratio = lambda a, b: (lambda d: d[a] / d[b].where(d[b] > 0))
    t = lambda task, title, kind, rows, load: {"task": task, "title": title, "kind": kind, "rows": rows, "load": load}
    return [t("euro_traffic", "공항별 일 운항 편수", "regression", 450_000, lambda: _daily("Airport_Traffic", "APT_ICAO", lambda d: d.FLT_TOT_1, [7, 1])),
            t("euro_atfm", "공항별 도착 ATFM 지연 발생 여부 (일 단위)", "binary", 450_001, lambda: _daily("Airport_Arrival_ATFM_Delay", "APT_ICAO", lambda d: d.DLY_APT_ARR_1.fillna(0), [1, 7], y_fn=lambda v: np.where(np.isfinite(v), (v > 0).astype(np.float32), np.nan))),
            t("euro_all_pre", "공항별 출발편당 전체 사전 지연(분)", "regression", 120_000, lambda: _daily("All_Pre-Departure_Delay", "APT_ICAO", ratio("DLY_ALL_PRE_2", "FLT_DEP_IFR_2"), [1, 7])),
            t("euro_atc_pre", "공항별 출발편당 ATC 사전 지연(분)", "regression", 120_001, lambda: _daily("ATC_Pre-Departure_Delay", "APT_ICAO", ratio("DLY_ATC_PRE_2", "FLT_DEP_IFR_2"), [1, 7])),
            t("euro_enroute", "관제 구역별 항공편당 항로 ATFM 지연(분)", "regression", 100_000, lambda: _daily("En-Route_ATFM_Delay_AUA", "ENTITY_NAME", lambda d: d.DLY_ERT_1.fillna(0) / d.FLT_ERT_1.where(d.FLT_ERT_1 > 0), [1, 7])),
            t("euro_taxi_out", "공항별 월 택시아웃 추가 시간(분/편)", "regression", 3_000, lambda: _monthly("Taxi-Out_Additional_Time")),
            t("euro_taxi_in", "공항별 월 택시인 추가 시간(분/편)", "regression", 3_001, lambda: _monthly("Taxi-In_Additional_Time")),
            t("euro_asma", "공항별 월 접근 구역 추가 시간(분/편)", "regression", 3_002, lambda: _monthly("ASMA_Additional_Time"))]
