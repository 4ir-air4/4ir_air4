"""인천 도착편의 월별 지연률 추정.  사용법: python forecast_icn_monthly.py

작년 같은 달 값에 최근 수준 변화를 곱한다. 편별 모델의 결과가 아닌 단순 계산이다.
정답을 모르는 상태의 예측을 채점하려고 처음 낸 값을 outputs/korea_ops/icn_monthly_forecast.json 에 고정한다.
다시 실행하면 추정값은 그대로 두고, 그사이 원본에 들어온 달의 실제 값만 채운다.
"""
import json
from datetime import datetime

import pandas as pd

from core.store import Store

TARGETS = ["2026-07", "2026-08", "2026-09", "2026-10", "2026-11"]
SHIFT = "2025-11"  # 이 달부터 지연률 수준이 낮아졌다. 이후 달은 이미 낮아진 값이라 비율을 곱하지 않는다
LEVEL_MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
TEST_LEVEL_MONTHS = ["2025-11", "2025-12", "2026-01", "2026-02"]
TEST_MONTHS = ["2026-03", "2026-04", "2026-05", "2026-06"]
MIN_DAYS = 25  # 원본이 빠진 달은 지연률이 왜곡되므로 쓰지 않는다


def monthly_rates(arrivals):
    months = arrivals.assign(late=arrivals.delay >= 15).groupby("ym").agg(rate=("late", "mean"), days=("date", "nunique"))
    return months.rate[months.days >= MIN_DAYS]


def year_before(month):
    return f"{int(month[:4]) - 1}{month[4:]}"


def level_ratio(rate, months):
    return rate[months].mean() / rate[[year_before(m) for m in months]].mean()


def estimate(rate, month, ratio):
    last = year_before(month)
    return float(rate[last]) if last >= SHIFT else float(rate[last] * ratio)


def backtest(rate):
    """앞쪽 달의 비율로 뒤쪽 달을 맞춰 본다."""
    ratio = level_ratio(rate, TEST_LEVEL_MONTHS)
    rows = [{"달": m, "실제": float(rate[m]), "추정": estimate(rate, m, ratio), "작년 같은 달": float(rate[year_before(m)])} for m in TEST_MONTHS]
    return {"비율": float(ratio), "추정 평균 오차": sum(abs(r["실제"] - r["추정"]) for r in rows) / len(rows),
            "작년 값 평균 오차": sum(abs(r["실제"] - r["작년 같은 달"]) for r in rows) / len(rows), "달": rows}


def update(path, rate, now):
    """고정된 추정값을 유지하고 실제 값만 채운다."""
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
    else:
        ratio = level_ratio(rate, LEVEL_MONTHS)
        saved = {"만든 시각": now, "비율": float(ratio), "비율 기준": f"{LEVEL_MONTHS[0]}~{LEVEL_MONTHS[-1]}", "시험": backtest(rate),
                 "달": [{"달": m, "작년 같은 달": float(rate[year_before(m)]), "추정": estimate(rate, m, ratio), "실제": None} for m in TARGETS]}
    for row in saved["달"]:
        if row["달"] in rate.index:
            row["실제"] = float(rate[row["달"]])
    path.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
    return saved


if __name__ == "__main__":
    from categories.korea_ops.pipeline import load_flights

    flights = load_flights()
    result = update(Store("korea_ops").dir / "icn_monthly_forecast.json", monthly_rates(flights[flights.direction == "도착"]), datetime.now().isoformat(timespec="minutes"))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    print(pd.DataFrame(result["달"]).round(3).to_string(index=False))
