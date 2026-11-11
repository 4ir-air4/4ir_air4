"""공항 이용객 데이터·모델 점검표 (읽기만 하고 아무것도 바꾸지 않습니다).

    python ojh/tools/check_airport_data.py

수집(fetch_airports.py) 뒤에는 '데이터' 부분이, 학습(models/train2.py) 뒤에는 '모델' 부분까지 나옵니다.
[주의] 표시가 붙은 항목을 위에서부터 확인하세요.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]  # ojh/tools -> ojh
DATA_DIR = BASE / "data" / "airport_demand"
SAVE_DIR = BASE / "models" / "saved" / "airport"
warnings = []
MIN_HISTORY_MONTHS = 36   # models/train2.py 와 같은 값


def note(message):
    """코드가 이미 처리하는 내용이라 [주의]로 세지 않는 참고 사항"""
    print(f"  (참고) {message}")


def title(text):
    print(f"\n{'=' * 8} {text} {'=' * 8}")


def flag(message):
    warnings.append(message)
    print(f"  [주의] {message}")


passengers_path = DATA_DIR / "airport_demand_passengers_monthly.csv"
gdp_path = DATA_DIR / "airport_demand_gdp_growth.csv"
if not passengers_path.exists():
    raise SystemExit(f"{passengers_path} 이 없습니다. 먼저 python ojh/collect/fetch_airports.py 를 실행하세요.")

df = pd.read_csv(passengers_path, parse_dates=["date"])
title("1. 데이터 개요")
print(f"  공항 {df['airport'].nunique():,}개 | 행 {len(df):,}개 | 기간 {df['date'].min():%Y-%m} ~ {df['date'].max():%Y-%m}")
print(df.groupby(["region", "source"])["airport"].nunique().rename("공항 수").to_string())

title("2. 출처별 월 합계 (마지막 15개월, 전년 같은 달 대비)")
for (region, source), g in df.groupby(["region", "source"]):
    total = g.groupby("date")["passengers"].sum().sort_index()
    yoy = (total / total.shift(12)).tail(15)
    print(f"  < {region} / {source} >")
    for d, v in yoy.items():
        print(f"    {d:%Y-%m}  합계 {total[d]:>14,.0f}   전년비 {v:5.2f}" if np.isfinite(v) else f"    {d:%Y-%m}  합계 {total[d]:>14,.0f}")
    recent = yoy.dropna()
    if len(recent) >= 6 and recent.iloc[-3:].min() < 0.8 * recent.iloc[:-3].median():
        note(f"{region}/{source}: 마지막 몇 달의 합계가 낮습니다. 나라마다 통계가 올라오는 시점이 달라서 생기는 정상 현상이고, "
             "train2.py 가 학습에서는 미완성 달을 잘라내고 예측은 공항별 마지막 실측 달부터 이어 붙입니다.")

title("3. 달별 공항 수 (마지막 8개월)")
counts = df.groupby("date")["airport"].nunique().sort_index()
print(counts.tail(8).to_string())
if counts.iloc[-1] < 0.8 * counts.tail(24).median():
    note("마지막 달들의 공항 수가 적습니다. 위와 같은 이유이며 train2.py 가 자동으로 처리합니다.")

title("4. 규모 상위 10개 공항 (최근 12개월 평균 월 이용객)")
recent_df = df[df["date"] > df["date"].max() - pd.DateOffset(months=12)]
top = recent_df.groupby(["airport", "airport_name", "region"])["passengers"].mean().sort_values(ascending=False).head(10)
print(top.round(0).to_string())
print("  -> 상식과 맞나요? (예: 런던 히스로·애틀랜타·파리 샤를드골이 위쪽에 있어야 합니다)")
if df["region"].nunique() > 1:
    note("미국(BTS)·유럽(Eurostat)·영국(CAA)·한국(에어포탈)은 집계 기준이 조금씩 달라요. "
         "특히 한국은 국내선을 출발만 세서 김포·제주가 작게 잡힙니다. 지역 간 절대 순위 비교는 참고용으로만 보세요.")

title("5. 이력이 짧거나 구멍이 많은 공항")
span = df.groupby("airport")["date"].agg(["min", "max", "count"])
span["months"] = (span["max"].dt.year - span["min"].dt.year) * 12 + span["max"].dt.month - span["min"].dt.month + 1
span["missing"] = span["months"] - span["count"]
short = span[span["count"] < MIN_HISTORY_MONTHS]
# 빠진 달은 학습에 쓰일 만한 공항(월평균 2만 명 이상, train2.py 와 같은 기준)만 본다. 작은 공항은 어차피 학습에서 빠진다.
big = df.groupby("airport")["passengers"].mean() >= 20_000
span_big = span[big.reindex(span.index, fill_value=False)]
gappy = span_big[span_big["missing"] > 12]
print(f"  {MIN_HISTORY_MONTHS}개월 미만 이력: {len(short)}개 (학습에서 제외됩니다)")
print(f"  월평균 2만 명 이상 공항 {len(span_big)}개 중 빠진 달이 12개월 초과: {len(gappy)}개")
if len(gappy):
    print("    예: " + ", ".join(f"{a}({int(m)}개월)" for a, m in gappy["missing"].sort_values(ascending=False).head(8).items()))
if len(gappy) > 0.1 * max(len(span_big), 1):
    flag("규모 있는 공항 중 빠진 달이 많은 곳이 10%를 넘습니다. 수집 파일(특히 연도별 Eurostat)이 모두 받아졌는지 확인하세요.")

title("6. GDP 성장률")
if gdp_path.exists():
    gdp = pd.read_csv(gdp_path)
    missing = sorted(set(df["country"].dropna().unique()) - set(gdp["country"].unique()))
    print(f"  {gdp['country'].nunique()}개 국가, {gdp['year'].min()}~{gdp['year'].max()}")
    if missing:
        flag(f"GDP가 없는 국가: {missing} (평균값으로 대체됩니다)")
else:
    flag("GDP 파일이 없습니다. GDP 피처가 평균값으로 대체됩니다 (python ojh/collect/fetch_airports.py --gdp-only).")

comparison_path = SAVE_DIR / "model_comparison.csv"
if comparison_path.exists():
    title("7. 모델 비교")
    comp = pd.read_csv(comparison_path)
    kind = "WAPE" if "Val WAPE (%)" in comp.columns else "MAPE"   # 선정 기준 (WAPE: 전체 절대오차 합 / 전체 실제값 합)
    cols = [c for c in ["Model", f"Val {kind} (%)", f"Val 12개월 재귀 {kind} (%)", f"Test {kind} (%)", f"12개월 재귀 {kind} (%)"] if c in comp.columns]
    key = f"Val 12개월 재귀 {kind} (%)" if f"Val 12개월 재귀 {kind} (%)" in comp.columns else f"Val {kind} (%)"
    print(comp[cols].round(2).to_string(index=False))
    base = comp[comp["Model"].str.startswith("Baseline")]
    models = comp[~comp["Model"].str.startswith("Baseline")]
    if len(base) and len(models):
        b, best = base.iloc[0], models.sort_values(key).iloc[0]
        print(f"  선정 모델: {best['Model']} | {key} {best[key]:.2f}% vs 기준선 {b[key]:.2f}%")
        if best[key] >= b[key]:
            flag("선정 모델이 '작년 같은 달' 기준선보다 낫지 않습니다. 모델이 학습한 가치가 없다는 뜻이라 데이터/피처를 점검해야 합니다.")
        if best[f"Test {kind} (%)"] > 2.5 * best[f"Val {kind} (%)"] and best[f"Test {kind} (%)"] > 5:
            flag(f"Test {kind}가 Validation보다 훨씬 큽니다. 마지막 달들의 미완성 데이터나 구간 변화가 원인일 수 있어요.")
        if best[f"Test {kind} (%)"] > 15:
            flag(f"Test {kind}가 15%를 넘습니다. 예측 숫자를 그대로 믿기 어렵습니다.")
    for name in ("XGBoost", "LightGBM"):
        if name not in set(comp["Model"]):
            flag(f"{name} 결과가 없습니다 (라이브러리가 설치되지 않아 건너뛴 것일 수 있어요: pip install {name.lower()}).")
    per_path = SAVE_DIR / "airport_test_mape.csv"
    if per_path.exists():
        per = pd.read_csv(per_path)
        print("\n  지역별 Test MAPE (선정 모델)")
        print(per.groupby("region")["Test MAPE (%)"].agg(["count", "mean", "median"]).round(2).to_string())
        print("\n  Test MAPE가 큰 공항 5개")
        print(per.sort_values("Test MAPE (%)", ascending=False).head(5)[["airport", "airport_name", "Test MAPE (%)"]].round(2).to_string(index=False))
else:
    print("\n(모델 점검은 python ojh/models/train2.py 로 학습한 뒤 다시 실행하면 나옵니다.)")

title("요약")
print(f"  [주의] 항목 {len(warnings)}개" if warnings else "  눈에 띄는 문제가 없습니다.")