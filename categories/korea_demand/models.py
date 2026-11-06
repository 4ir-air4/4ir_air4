"""한국 수요 4개 모델: 인천 항공사·공항별 월 여객, 기종별 월 여객, 제주 일별 여객."""
import pandas as pd

from categories.korea_demand.pipeline import KOREA, load_incheon, load_types
from core.data import read_csv_any
from core.panel import day_label, month_label, month_ranges, panel_data


def incheon():
    d = load_incheon()
    keys = ["airline", "airport", "direction", "charter"]
    panel = d[d.passengers > 0].groupby(keys + ["t"], as_index=False).passengers.sum()
    tr, val, te = month_ranges(int(panel.t.min()), int(panel.t.max()))
    return panel_data(panel, keys, "passengers", tr, val, te, [1, 12], label=month_label, seq_len=12, group="과거 12개월 값",
                      train_period=f"{month_label(tr[0])}~{month_label(tr[1])}", test_period=f"{month_label(te[0])}~{month_label(te[1])}")


def types():
    d = load_types()
    keys = ["공항", "항공사", "기종", "구분"]
    panel = d.groupby(keys + ["t"], as_index=False).passengers.sum()
    tr, val, te = month_ranges(int(panel.t.min()), int(panel.t.max()), test_months=3, val_months=2)
    return panel_data(panel, keys, "passengers", tr, val, te, [1], label=month_label, seq_len=6, group="과거 값", note="월별 자료가 2025년뿐이라 학습 기간이 짧습니다",
                      train_period=f"{month_label(tr[0])}~{month_label(tr[1])}", test_period=f"{month_label(te[0])}~{month_label(te[1])}")


def jeju():
    d = read_csv_any(KOREA / "한국공항공사_제주공항_여객_학습데이터.csv")
    d.columns = ["date", "arr", "dep", "y"]
    date = pd.to_datetime(d.date)
    d = d.assign(all="제주", t=date.map(pd.Timestamp.toordinal), x_dow=date.dt.day_of_week, x_month=date.dt.month)
    start = pd.Timestamp("2025-01-01").toordinal()
    val_start = pd.Timestamp("2024-10-01").toordinal()
    calendar = {"x_dow": lambda t: pd.Timestamp.fromordinal(t).day_of_week, "x_month": lambda t: pd.Timestamp.fromordinal(t).month}
    return panel_data(d, ["all"], "y", (int(d.t.min()), val_start - 1), (val_start, start - 1), (start, int(d.t.max())), [7, 1, 364], ["x_dow", "x_month"], calendar,
                      label=day_label, seq_len=14, group="과거 값, 달력", train_period="2023-01~2024-09", test_period="2025-01~10")


def tasks():
    return [{"task": "icn_passengers", "title": "인천 항공사·공항별 월 여객", "kind": "regression", "rows": 60_000, "load": incheon},
            {"task": "type_passengers", "title": "공항·항공사·기종별 월 여객", "kind": "regression", "rows": 5_000, "load": types},
            {"task": "jeju_passengers", "title": "제주공항 일별 여객", "kind": "regression", "rows": 1_100, "load": jeju}]
