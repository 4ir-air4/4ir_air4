"""네트워크와 원본 없이 도는 forecast_icn_monthly.py 점검. 실행: python test_forecast_icn.py"""
import json
import tempfile
from pathlib import Path

import pandas as pd

import forecast_icn_monthly as f


def rates():
    months = pd.period_range("2024-11", "2026-06", freq="M").astype(str)
    return pd.Series([0.2 if m < f.SHIFT else 0.1 for m in months], index=months)


def check_rates():
    days = pd.date_range("2026-01-01", "2026-01-30").strftime("%Y-%m-%d")
    full = pd.DataFrame({"ym": "2026-01", "date": days, "delay": [15.0, 14.0] * 15})
    short = pd.DataFrame({"ym": "2026-02", "date": ["2026-02-01"], "delay": [60.0]})
    rate = f.monthly_rates(pd.concat([full, short]))
    assert rate.to_dict() == {"2026-01": 0.5}, "15분 이상을 지연으로 세고, 자료가 빠진 달은 뺀다"


def check_estimate():
    rate = rates()
    ratio = f.level_ratio(rate, f.LEVEL_MONTHS)
    assert abs(ratio - 0.5) < 1e-9, "2026년 수준이 전년의 절반"
    assert abs(f.estimate(rate, "2026-07", ratio) - 0.1) < 1e-9, "낮아지기 전의 달에는 비율을 곱한다"
    rate["2025-11"] = 0.13
    assert f.estimate(rate, "2026-11", ratio) == 0.13, "이미 낮아진 달은 그대로 쓴다"
    result = f.backtest(rates())
    assert abs(result["추정 평균 오차"]) < 1e-9 and abs(result["작년 값 평균 오차"] - 0.1) < 1e-9, "시험 오차"


def check_frozen(path):
    rate = rates()
    first = f.update(path, rate, "t1")
    assert all(row["실제"] is None for row in first["달"]), "처음에는 실제 값이 없다"
    changed = rate * 3  # 나중 원본 값이 달라져도 고정한 추정은 바뀌지 않아야 한다
    changed["2026-07"] = 0.25
    second = f.update(path, changed, "t2")
    assert second["만든 시각"] == "t1" and [r["추정"] for r in second["달"]] == [r["추정"] for r in first["달"]], "추정값 고정"
    assert second["달"][0]["실제"] == 0.25 and second["달"][1]["실제"] is None, "들어온 달만 실제 값을 채운다"
    assert json.loads(path.read_text(encoding="utf-8"))["달"][0]["실제"] == 0.25, "파일에 저장"


if __name__ == "__main__":
    check_rates()
    check_estimate()
    with tempfile.TemporaryDirectory() as tmp:
        check_frozen(Path(tmp) / "forecast.json")
    print("forecast_icn_monthly 점검 통과")
