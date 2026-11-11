"""app.py의 세 번째 탭: 🛫 전 세계 공항 이용객 수 수요 예측.

models/train2.py 가 만든 파일(models/saved/airport/)을 읽어서
  1) 파이프라인 개요  2) 모델 성능 비교  3) 이용객 예측 시뮬레이터 를 보여줍니다.
"""

import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SAVE_DIR = ROOT_DIR / "models" / "saved" / "airport"
TRAIN2_SCRIPT = ROOT_DIR / "models" / "train2.py"
CONTEXT_FILE = SAVE_DIR / "airport_context.pkl"
COMPARISON_FILE = SAVE_DIR / "model_comparison.csv"
TEST_MAPE_FILE = SAVE_DIR / "airport_test_mape.csv"

MODEL_FILES = {
    "Linear Regression": "model_linear_regression.pkl",
    "Random Forest": "model_random_forest.pkl",
    "XGBoost": "model_xgboost.pkl",
    "LightGBM": "model_lightgbm.pkl",
}
REGION_LABELS = {"US": "미국", "Europe": "유럽", "Korea": "한국", "Other": "기타"}
BASELINE_PREFIX = "Baseline"


# ---------------------------------------------------------
# 불러오기 (파일이 바뀌면 캐시가 자동으로 갱신됨)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False, max_entries=1)
def _import_trainer(mtime):
    spec = importlib.util.spec_from_file_location("airport_trainer", TRAIN2_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_trainer():
    return _import_trainer(TRAIN2_SCRIPT.stat().st_mtime)


@st.cache_resource(show_spinner=False, max_entries=8)
def _load_pickle(path, mtime):
    return joblib.load(path)


def load_pickle(path):
    path = Path(path)
    if not path.exists():
        return None
    return _load_pickle(str(path), path.stat().st_mtime)


def airport_label(meta, code):
    row = meta.loc[code]
    return f"{row['airport_name']} ({code})"


def metric(metrics, name="WAPE (%)"):
    """성능 지표 꺼내기. 예전 .pkl(WAPE 없음)이면 MAPE로 대신한다."""
    metrics = metrics or {}
    value = metrics.get(name, metrics.get("MAPE (%)", np.nan))
    return float(value) if value is not None else float("nan")


def full_history(ctx):
    """예측용 이력 (공항별 마지막 실측 달까지). 예전 .pkl 이면 학습용 표"""
    return ctx.get("history_full", ctx["history"])


KOREA_NOTE = (
    "🇰🇷 한국(에어포탈)은 **국내선은 출발만, 국제선은 출발+도착**을 세는 기준입니다. "
    "국내선 비중이 큰 김포·제주 등은 다른 나라(출발+도착) 기준보다 작게 잡혀서, 다른 나라 공항과 순위를 비교할 때 불리하게 보입니다. "
    "같은 공항 안에서의 변화(증감·계절성)를 보는 데는 문제가 없습니다."
)


def fmt_people(value):
    return "-" if value is None or not np.isfinite(value) else f"{value:,.0f}명"


# ---------------------------------------------------------
# 1. 파이프라인 개요
# ---------------------------------------------------------
PASSENGERS_CSV = ROOT_DIR / "data" / "airport_demand" / "airport_demand_passengers_monthly.csv"


@st.cache_data(show_spinner=False)
def _airport_funnel_cached(csv_mtime, script_mtime):
    trainer = load_trainer()
    data, _ = trainer.load_inputs()
    return trainer.airport_funnel(data)


def _airport_funnel_from_csv():
    """예전 .pkl 처럼 모델 파일에 공항 수 집계가 없을 때 데이터 파일에서 바로 계산"""
    try:
        if not PASSENGERS_CSV.exists() or not hasattr(load_trainer(), "airport_funnel"):
            return None
        return _airport_funnel_cached(PASSENGERS_CSV.stat().st_mtime, TRAIN2_SCRIPT.stat().st_mtime)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _region_summary_cached(ctx_mtime, script_mtime):
    return load_trainer().region_summary(load_pickle(CONTEXT_FILE))


def _region_summary(ctx):
    """지역별 공항 수 · 기간 · 행 수. 예전 .pkl 이면 바로 계산 (1~2초, 한 번만)"""
    if ctx.get("region_summary"):
        return ctx["region_summary"]
    try:
        if not hasattr(load_trainer(), "region_summary"):
            return None
        return _region_summary_cached(CONTEXT_FILE.stat().st_mtime, TRAIN2_SCRIPT.stat().st_mtime)
    except Exception:
        return None


def _by_region(summary, fmt):
    return " · ".join(f"{REGION_LABELS.get(r, r)} {fmt(v)}" for r, v in summary.items())


def render_overview(ctx, comparison):
    meta = ctx["meta"]
    counts = meta["region"].map(lambda r: REGION_LABELS.get(r, r)).value_counts()
    split = ctx["split_rows"]
    summary = _region_summary(ctx) or {}

    c1, c2, c3, c4 = st.columns(4)

    # --- 분석 공항 수 ---
    c1.metric("분석 공항 수", f"{len(meta):,}개")
    funnel = ctx.get("airport_funnel") or _airport_funnel_from_csv()
    if funnel:
        closed = sum(v["closed"] for v in summary.values())
        per_region = _by_region(summary, lambda v: f"{v['airports']}개") if summary else ""
        c1.caption(
            f"전체 {funnel['total']:,}개 중 조건 통과 {funnel['eligible']:,}개"
            + (f" → 규모 상위 {funnel['used']:,}개 사용" if funnel["used"] < funnel["eligible"] else " (전부 사용)")
            + (f"  \n{per_region}" if per_region else ""),
            help=(
                f"수집된 공항 {funnel['total']:,}개\n\n"
                f"→ 이력 {funnel['min_history_months']}개월 이상: {funnel['history_ok']:,}개\n\n"
                f"→ 월평균 승객 {funnel['min_avg_passengers']:,}명 이상: {funnel['eligible']:,}개\n\n"
                + (f"→ 학습 속도를 위해 규모 상위 최대 {funnel['max_airports']:,}개: {funnel['used']:,}개"
                   if funnel.get("max_airports") else "→ 조건을 통과한 공항을 전부 분석에 사용")
                + (f"\n\n※ 이 중 {closed}개는 폐쇄 등으로 데이터가 1년 넘게 끊겨서 학습에만 쓰고, 시뮬레이터·순위 레이스에서는 뺍니다 (예: 베를린 테겔)." if closed else "")
            ),
        )

    # --- 데이터 기간 ---
    c2.metric("데이터 기간", f"{ctx['first_date']:%Y-%m} ~ {ctx['last_date']:%Y-%m}")
    if summary:
        c2.caption(
            # '~' 두 개가 한 줄에 있으면 마크다운 취소선이 되므로 '–' 를 쓴다
            _by_region(summary, lambda v: f"{v['first']:%Y-%m}–{v['last']:%Y-%m}"),
            help=(
                "지역마다 출처와 기간이 다릅니다.\n\n"
                + "\n\n".join(f"- **{REGION_LABELS.get(r, r)}** ({v['source']}): {v['first']:%Y-%m} – {v['last']:%Y-%m}" for r, v in summary.items())
                + f"\n\n나라마다 통계가 올라오는 시점이 달라서, **학습은 모든 공항을 {ctx['last_date']:%Y-%m}에서 맞춰 자릅니다** "
                "(위의 큰 숫자). 예측은 공항마다 자기 마지막 실측 달 다음부터 시작합니다."
            ),
        )

    # --- 학습용 행 수 ---
    c3.metric("학습용 행 수", f"{ctx['n_rows']:,}")
    obs = sum(v["obs"] for v in summary.values())
    c3.caption(
        f"Train {split['train']:,} · Val {split['val']:,} · Test {split['test']:,}",
        help=(
            "1행 = 공항 1곳의 1개월 기록입니다.\n\n"
            + (f"학습 기간 안의 실제 공항-월 기록 {obs:,}개 중, 직전 12개월 가운데 9개월 이상 기록이 있어야 "
               "'과거 승객 수' 피처를 만들 수 있어서 각 공항의 첫 1년치와 공백 직후 달은 빠집니다.\n\n" if obs else "")
            + (("지역별: " + _by_region(summary, lambda v: f"{v['rows']:,}행") + "\n\n") if summary else "")
            + f"- Train ~{(ctx['val_start'] - pd.DateOffset(months=1)):%Y-%m}: {split['train']:,}행 (학습)\n"
            f"- Validation {ctx['val_start']:%Y-%m}~: {split['val']:,}행 (튜닝·모델 선정)\n"
            f"- Test {ctx['test_start']:%Y-%m}~: {split['test']:,}행 (최종 평가, 선정에 안 씀)"
        ),
    )

    # --- 최종 선정 모델 ---
    c4.metric("최종 선정 모델", ctx["best_model"])
    key = ctx.get("selection_metric", "")
    if key in comparison.columns:
        best_row = comparison[comparison["Model"] == ctx["best_model"]]
        base_row = comparison[comparison["Model"].astype(str).str.startswith(BASELINE_PREFIX)]
        if len(best_row) and len(base_row):
            b, base = float(best_row[key].iloc[0]), float(base_row[key].iloc[0])
            test_key = "Test WAPE (%)" if "Test WAPE (%)" in comparison.columns else "Test MAPE (%)"
            c4.caption(
                f"12개월 예측 오차 {b:.2f}% (기준선 {base:.2f}%)",
                help=(
                    f"선정 기준: **{key}**\n\n"
                    "Validation 구간에서 시작점을 0·6·12개월로 바꿔 가며 12개월 앞까지 이어서 예측했을 때의 오차입니다 "
                    "(시뮬레이터의 연간 곡선과 같은 방식).\n\n"
                    f"- {ctx['best_model']}: {b:.2f}%\n- 기준선(작년 같은 달 값 그대로): {base:.2f}%\n"
                    f"- 참고: Test 한 달 앞 오차 {float(best_row[test_key].iloc[0]):.2f}%\n\n"
                    "WAPE = 전체 절대오차 합 ÷ 전체 실제 승객 합"
                ),
            )

    st.markdown(
        f"""
**6단계 파이프라인**

1. **피처 엔지니어링** — 월(sin/cos), 코로나 구간, 국가 GDP 성장률, 지역,
   과거 승객 수(1·2·3·6·12개월 전), 최근 3·6개월 평균/변동성, 전년 대비 증감, 공항 규모.
   타깃은 *직전 12개월 평균 대비 변화율(로그)* 이라 공항 규모와 상관없이 학습됩니다.
2. **시점 기준 분할** — Train ~{(ctx['val_start'] - pd.DateOffset(months=1)):%Y-%m} ({split['train']:,}행) |
   Validation {ctx['val_start']:%Y-%m} ~ ({split['val']:,}행) | Test {ctx['test_start']:%Y-%m} ~ ({split['test']:,}행). 랜덤 셔플은 쓰지 않았습니다.
3. **모델 학습·튜닝** — Linear Regression, Random Forest, XGBoost, LightGBM (Optuna로 Validation 기준 튜닝)
4. **평가** — MAE · RMSE · MAPE · **WAPE** · R², 그리고 앱의 연간 곡선과 같은 방식인 *12개월 재귀 예측* 정확도
5. **모델 선정** — Validation 구간에서 시작점을 바꿔 가며(0·6·12개월) 12개월 재귀 예측을 평가한 **WAPE**가 가장 낮은 모델 (Test는 선정에 쓰지 않음)
   - WAPE = 전체 절대오차 합 ÷ 전체 실제 승객 합. "전체 승객 중 몇 %를 틀렸나"입니다.
   - MAPE(공항·월별 오차율의 평균)는 겨울에 승객이 수십 명뿐인 휴양지 공항 몇 곳이 수천 %를 만들어 크게 부풀려지기 때문에 선정에 쓰지 않습니다.
6. **저장** — `.pkl` (모델 + 피처 목록 + 성능 지표)
"""
    )

    st.markdown("**지역별 공항 수**")
    st.dataframe(counts.rename("공항 수").to_frame(), use_container_width=True)

    st.caption(
        "⚠️ 미국은 BTS T-100(출발+도착 승객), 유럽은 Eurostat(Passengers carried), 영국은 CAA(Terminal passengers) 기준이라 지표 정의가 조금 다릅니다. "
        "공항 간 절대 순위 비교보다 '같은 공항의 변화'를 보는 용도에 더 적합합니다. "
        "한국 외 아시아·중동 등 다른 지역은 포함되어 있지 않습니다."
    )
    if (meta["region"] == "Korea").any():
        st.caption(KOREA_NOTE)


# ---------------------------------------------------------
# 2. 모델 성능 비교
# ---------------------------------------------------------
def render_comparison(ctx, comparison):
    st.markdown("#### 모델 성능 비교")
    show = comparison.copy()
    for col in show.columns[1:]:
        show[col] = show[col].astype(float).round(2)
    st.dataframe(show, use_container_width=True, hide_index=True)

    models_only = comparison[~comparison["Model"].astype(str).str.startswith(BASELINE_PREFIX)]
    fig = go.Figure()
    kind = "WAPE" if "Val WAPE (%)" in comparison.columns else "MAPE"
    for col, name in [(f"Val {kind} (%)", "Validation(한 달 앞)"), (f"Val 12개월 재귀 {kind} (%)", "Validation 12개월 재귀(선정 기준)"), (f"Test {kind} (%)", "Test(한 달 앞)"), (f"12개월 재귀 {kind} (%)", "Test 12개월 재귀")]:
        if col in comparison.columns:
            fig.add_bar(x=comparison["Model"], y=comparison[col], name=name)
    fig.update_layout(barmode="group", title=f"모델별 {kind} (%) — 낮을수록 좋음", height=380, yaxis_title=f"{kind} (%)")
    st.plotly_chart(fig, use_container_width=True)

    best = ctx["best_model"]
    st.success(f"선정 모델: **{best}** ({ctx.get('selection_metric', 'Validation 12개월 재귀')} 기준). 'Baseline'은 '작년 같은 달 값을 그대로 쓰는' 단순 기준선입니다.")
    if kind == "WAPE":
        st.caption("WAPE = 전체 절대오차 합 ÷ 전체 실제 승객 합. 표의 MAPE 칸은 참고용이며, 승객이 거의 없는 달이 있는 공항 때문에 크게 나올 수 있습니다.")

    st.markdown("#### 공항별 예측 vs 실제")
    available = [m for m, f in MODEL_FILES.items() if (SAVE_DIR / f).exists()]
    if not available:
        st.info("모델 파일이 없습니다.")
        return
    meta = ctx["meta"]
    left, right = st.columns(2)
    model_name = left.selectbox("모델", available, index=available.index(best) if best in available else 0, key="airport_cmp_model")
    order = list(meta.index)
    airport = right.selectbox("공항", order, format_func=lambda c: airport_label(meta, c), key="airport_cmp_airport")
    bundle = load_pickle(SAVE_DIR / MODEL_FILES[model_name])

    history = ctx["history"].loc[airport].dropna()
    history = history[history.index >= ctx["test_start"] - pd.DateOffset(months=36)]
    rec = bundle["recursive_predictions"]
    rec = rec[rec["airport"] == airport].sort_values("date")
    one = bundle["test_predictions"]
    one = one[one["airport"] == airport].sort_values("date")

    fig = go.Figure()
    fig.add_scatter(x=history.index, y=history.values, name="실제", mode="lines+markers")
    fig.add_scatter(x=one["date"], y=one["predicted"], name="한 달 앞 예측(Test)", mode="lines+markers")
    fig.add_scatter(x=rec["date"], y=rec["predicted"], name="12개월 재귀 예측", mode="lines", line=dict(dash="dash"))
    fig.add_scatter(x=one["date"], y=one["baseline"], name="Baseline(작년 같은 달)", mode="lines", line=dict(dash="dot"))
    fig.update_layout(height=420, yaxis_title="월 이용객 수(명)", title=f"{airport_label(meta, airport)} — Test 구간 예측 비교")
    st.plotly_chart(fig, use_container_width=True)

    importance = bundle.get("feature_importance") or {}
    if importance:
        top = list(importance.items())[:10][::-1]
        fig = go.Figure(go.Bar(x=[v for _, v in top], y=[k for k, _ in top], orientation="h"))
        fig.update_layout(height=360, title=f"{model_name} 피처 중요도 (상위 10)", xaxis_title="상대 중요도")
        st.plotly_chart(fig, use_container_width=True)

    if TEST_MAPE_FILE.exists():
        with st.expander("공항별 Test MAPE (선정 모델)"):
            st.dataframe(pd.read_csv(TEST_MAPE_FILE).round(2), use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# 3. 예측 시뮬레이터
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def _top10_table(best_model, mtime):
    ctx = load_pickle(CONTEXT_FILE)
    trainer = load_trainer()
    bundle = load_pickle(SAVE_DIR / MODEL_FILES[best_model])
    meta = ctx["meta"]
    filled, last_obs = trainer.forecast_all_filled(bundle, ctx, n_months=12)
    # 공항마다 자기 마지막 실측 달 다음 12개월의 합
    totals = {c: float(filled.loc[c, (filled.columns > d) & (filled.columns <= d + pd.DateOffset(months=12))].sum()) for c, d in last_obs.items()}
    codes = list(totals)
    out = pd.DataFrame({"공항": [airport_label(meta, c) for c in codes], "지역": [REGION_LABELS.get(meta.loc[c, "region"], meta.loc[c, "region"]) for c in codes],
                        "예측 시작": [f"{last_obs[c] + pd.DateOffset(months=1):%Y-%m}" for c in codes],
                        "향후 12개월 예상 이용객(명)": [round(totals[c]) for c in codes]})
    return out.sort_values("향후 12개월 예상 이용객(명)", ascending=False).head(10).reset_index(drop=True)


GDP_GRID = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
FLAT_THRESHOLD_PCT = 0.5  # 성장률을 -2% -> +5%로 바꿔도 예측이 이보다 덜 변하면 '반응 없음'으로 안내


def _target_date(series, month):
    return next(d for d in series.index if d.month == month)


def gdp_sensitivity(trainer, bundle, ctx, airport, month):
    """성장률을 -2%~5%로 바꿔 가며 선택한 월의 예측이 실제로 변하는지 확인한다."""
    rows = []
    for g in GDP_GRID:
        s = trainer.forecast_for_airport(bundle, ctx, airport, gdp_growth=g)
        rows.append({"gdp": g, "month_pred": float(s[_target_date(s, month)]), "year_total": float(s.sum())})
    return pd.DataFrame(rows)


def compare_airports(trainer, bundle, ctx, codes, gdp):
    """여러 공항에 같은 조건을 적용. 공항 규모 차이를 없애려고 '최근 12개월 평균 = 100'인 지수로도 돌려준다."""
    history = full_history(ctx)
    curves, rows = {}, []
    for code in codes:
        s = trainer.forecast_for_airport(bundle, ctx, code, gdp_growth=gdp)
        recent = history.loc[code].dropna().iloc[-12:]
        level = float(recent.mean()) if len(recent) else float("nan")
        curves[code] = s / level * 100 if level and np.isfinite(level) else s * np.nan
        recent_total = float(recent.sum()) if len(recent) == 12 else float("nan")
        total = float(s.sum())
        rows.append({
            "공항": airport_label(ctx["meta"], code),
            "향후 12개월 예상(명)": round(total),
            "최근 12개월 대비(%)": round((total / recent_total - 1) * 100, 1) if np.isfinite(recent_total) and recent_total else np.nan,
        })
    return curves, pd.DataFrame(rows)


def render_result(trainer, ctx, result, airports):
    meta, history = ctx["meta"], full_history(ctx)
    code = result["airport"]
    scenario, baseline = result["scenario"], result["baseline"]
    target = _target_date(scenario, result["month"])
    pred, base_pred, normal = float(scenario[target]), float(baseline[target]), result["normal"]
    year_total, base_total = float(scenario.sum()), float(baseline.sum())
    recent = history.loc[code].dropna().iloc[-12:]
    recent_total = float(recent.sum()) if len(recent) == 12 else float("nan")

    st.markdown("#### ④ 예측 결과")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        f"{target:%Y년 %m월} 예상 이용객", fmt_people(pred),
        delta=None if not np.isfinite(normal) else f"{pred - normal:+,.0f}명 (평년 대비)",
    )
    m2.metric(
        "조건 변경 효과 (기본 전망 대비)", f"{(pred / base_pred - 1) * 100:+.2f}%" if base_pred else "-",
        delta=f"{pred - base_pred:+,.0f}명", delta_color="off",
    )
    m3.metric(
        "향후 12개월 합계", fmt_people(year_total),
        delta=f"{(year_total / recent_total - 1) * 100:+.1f}% (최근 12개월 대비)" if np.isfinite(recent_total) and recent_total else None,
    )
    m4.metric("검증 / 테스트 WAPE", f"{result['val_mape']:.2f}% / {result['test_mape']:.2f}%")
    st.caption(
        f"{result['model']} · {airport_label(meta, code)} · 성장률 {result['gdp']:.1f}% · "
        f"12개월 재귀 예측 WAPE {result['rec_mape']:.2f}% (연간 곡선의 정확도 참고)"
    )

    tab_curve, tab_gdp, tab_cmp = st.tabs(["📈 연간 곡선", "🎚️ 성장률 민감도", "🆚 공항 비교"])

    # --- 연간 곡선 (+ 평균 오차 범위) ---
    with tab_curve:
        band = result["rec_mape"] if np.isfinite(result["rec_mape"]) else result["test_mape"]
        past = history.loc[code].dropna().iloc[-24:]
        fig = go.Figure()
        if np.isfinite(band):
            upper = (scenario.values * (1 + band / 100)).tolist()
            lower = (scenario.values * (1 - band / 100)).tolist()
            fig.add_scatter(
                x=list(scenario.index) + list(scenario.index[::-1]), y=upper + lower[::-1], fill="toself",
                fillcolor="rgba(31,119,180,0.12)", line=dict(width=0), name=f"평균 오차 범위 (±{band:.1f}%)", hoverinfo="skip",
            )
        fig.add_scatter(x=past.index, y=past.values, name="실제(최근 24개월)", mode="lines+markers")
        fig.add_scatter(x=baseline.index, y=baseline.values, name="기본 전망", mode="lines", line=dict(dash="dot"))
        fig.add_scatter(x=scenario.index, y=scenario.values, name="내가 설정한 조건", mode="lines+markers")
        fig.add_scatter(x=[target], y=[pred], name="선택한 월", mode="markers", marker=dict(size=14, symbol="star"))
        fig.update_layout(height=430, yaxis_title="월 이용객 수(명)", title=f"{airport_label(meta, code)} 연간 월별 예상 이용객 곡선")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"음영은 이 모델의 12개월 재귀 예측 평균 오차(WAPE ±{band:.1f}%)를 곡선에 그대로 얹은 참고용 범위이며, 통계적 신뢰구간은 아닙니다.")
        table = pd.DataFrame({"월": [f"{d:%Y-%m}" for d in scenario.index], "내가 설정한 조건(명)": scenario.round(0).to_numpy(), "기본 전망(명)": baseline.round(0).to_numpy()})
        st.download_button(
            "📥 예측 결과 CSV 내려받기", table.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"airport_forecast_{code}.csv", mime="text/csv", key="airport_sim_download",
        )

    # --- 성장률 민감도 ---
    with tab_gdp:
        sens = result.get("sens")
        if sens is None or sens.empty:
            st.info("민감도 계산 결과가 없습니다. 예측을 다시 실행해주세요.")
        else:
            fig = go.Figure()
            fig.add_scatter(x=sens["gdp"], y=sens["month_pred"], mode="lines+markers", name=f"{target:%m}월 예상 이용객")
            fig.add_scatter(x=[result["gdp"]], y=[pred], mode="markers", name="내가 고른 값", marker=dict(size=14, symbol="star"))
            fig.update_layout(height=360, xaxis_title="국가 경제 성장률 (%)", yaxis_title="월 이용객 수(명)", title=f"성장률에 따른 {target:%m}월 예측 변화")
            st.plotly_chart(fig, use_container_width=True)
            low, high = float(sens["month_pred"].iloc[0]), float(sens["month_pred"].iloc[-1])
            change = (high / low - 1) * 100 if low else float("nan")
            if not np.isfinite(change):
                st.info("민감도를 계산하지 못했습니다.")
            elif abs(change) < FLAT_THRESHOLD_PCT:
                st.info(
                    f"성장률을 -2%에서 +5%로 바꿔도 예측은 {change:+.2f}%만 변합니다. 이 모델은 성장률을 거의 쓰지 않는다는 뜻이고, "
                    "국가·연 단위 지표라 공항·월 단위 패턴에 묻혔을 가능성이 큽니다. 슬라이더 효과가 작아 보이는 것은 오류가 아니라 모델이 배운 결과입니다."
                )
            elif change > 0:
                st.success(f"성장률이 -2%에서 +5%로 오르면 {target:%m}월 예측이 {change:+.1f}% 변합니다. (이 모델이 과거 패턴에서 배운 관계이며 인과를 보장하지는 않습니다)")
            else:
                st.warning(
                    f"성장률이 오를수록 예측이 {change:+.1f}% 줄어드는 방향입니다. 코로나 시기의 성장률·이용객 변화가 섞여 배운 결과일 수 있어, "
                    "'성장률↑ = 수요↑'로 해석하면 안 됩니다."
                )

    # --- 공항 비교 ---
    with tab_cmp:
        pool = [a for a in airports if a != code]
        picked = st.multiselect(
            "함께 비교할 공항 (최대 5개, 같은 조건 적용)", pool, default=pool[:2], max_selections=5,
            format_func=lambda c: airport_label(meta, c), key=f"airport_sim_compare_{code}",
        )
        if not picked:
            st.info("비교할 공항을 선택해주세요.")
        else:
            bundle = load_pickle(SAVE_DIR / MODEL_FILES[result["model"]])
            with st.spinner("공항별 예측을 계산하는 중입니다..."):
                curves, table = compare_airports(trainer, bundle, ctx, [code] + list(picked), result["gdp"])
            fig = go.Figure()
            for c, curve in curves.items():
                fig.add_scatter(x=curve.index, y=curve.values, mode="lines+markers", name=airport_label(meta, c), line=dict(width=4 if c == code else 2))
            fig.add_hline(y=100, line_dash="dot", annotation_text="최근 12개월 평균 = 100")
            fig.update_layout(height=400, yaxis_title="지수 (최근 12개월 평균 = 100)", title="공항별 향후 12개월 수요 지수 (규모 차이 제거)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.caption("성장률 슬라이더 값을 모든 공항에 똑같이 적용한 비교입니다. 공항마다 실제 국가 성장률이 다르다는 점은 반영하지 않습니다.")

    st.caption(
        "※ 경제 성장률은 국가·연 단위 지표라서 모델이 배운 영향이 작을 수 있고, 코로나 시기의 급락과 섞여 있을 수 있습니다. "
        "수치는 '이 모델이 과거 패턴으로 추정한 값'이며 실제 이용객을 보장하지 않습니다."
    )


def render_simulator(ctx, comparison):
    trainer = load_trainer()
    meta = ctx["meta"]
    history = ctx["history"]
    last_date = ctx["last_date"]

    st.markdown("#### ① 분석 대상 공항 및 모델 선택")
    size = history.iloc[:, -12:].mean(axis=1).sort_values(ascending=False)
    # 폐쇄 등으로 데이터가 1년 넘게 끊긴 공항은 예측할 '현재'가 없어서 목록에서 뺀다
    active = set(trainer.active_airports(ctx)) if hasattr(trainer, "active_airports") else set(size.index)
    airports = [a for a in size.index if a in active]
    available = [m for m, f in MODEL_FILES.items() if (SAVE_DIR / f).exists()]
    best = ctx["best_model"]

    c1, c2 = st.columns(2)
    airport = c1.selectbox("공항 선택", airports, format_func=lambda c: airport_label(meta, c), key="airport_sim_airport")
    model_name = c2.selectbox("모델 선택", available, index=available.index(best) if best in available else 0, key="airport_sim_model")

    st.markdown("#### ② 예측 조건 설정")
    gdp_default = float(np.clip(round(trainer.latest_gdp(ctx, airport), 1), -2.0, 5.0))
    c3, c5 = st.columns(2)
    month = c3.selectbox("예측 월", list(range(1, 13)), format_func=lambda m: f"{m}월", key="airport_sim_month")
    gdp = c5.slider("국가 경제 성장률 전망 (%)", -2.0, 5.0, gdp_default, 0.1, key=f"airport_sim_gdp_{airport}")
    own_last = trainer.last_observed(ctx, airport) if hasattr(trainer, "last_observed") else last_date
    st.caption(
        f"이 공항의 마지막 실측 달은 {own_last:%Y-%m}입니다. 선택한 월은 그 다음에 처음 돌아오는 해당 월을 예측합니다. "
        f"(성장률 기본값 {gdp_default:.1f}%는 해당 국가의 가장 최근 값)"
    )
    if meta.loc[airport, "region"] == "Korea":
        st.caption(KOREA_NOTE)

    st.markdown("#### ③ 머신러닝 예측 연산")
    if st.button("🚀 승객 수요 예측 실행", type="primary", key="airport_sim_run"):
        bundle = load_pickle(SAVE_DIR / MODEL_FILES[model_name])
        with st.spinner("시나리오를 계산하는 중입니다..."):
            scenario = trainer.forecast_for_airport(bundle, ctx, airport, gdp_growth=gdp)
            baseline = trainer.forecast_for_airport(bundle, ctx, airport)
            sens = gdp_sensitivity(trainer, bundle, ctx, airport, month)
        st.session_state["airport_sim_result"] = {
            "airport": airport, "model": model_name, "month": month, "gdp": gdp,
            "scenario": scenario, "baseline": baseline, "sens": sens,
            "normal": trainer.normal_level(ctx, airport, month),
            "val_mape": metric(bundle.get("val_metrics")), "test_mape": metric(bundle.get("test_metrics")),
            "rec_mape": metric(bundle.get("recursive_metrics")),
        }

    result = st.session_state.get("airport_sim_result")
    if not result:
        st.info("조건을 고른 뒤 '승객 수요 예측 실행'을 눌러주세요.")
    elif result["airport"] not in history.index:
        st.info("모델을 다시 학습해서 공항 목록이 바뀌었습니다. 예측을 다시 실행해주세요.")
    else:
        render_result(trainer, ctx, result, airports)

    with st.expander("🌍 향후 12개월 예상 이용객 TOP 10 공항 (선정 모델)"):
        try:
            st.dataframe(_top10_table(best, CONTEXT_FILE.stat().st_mtime), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.warning(f"TOP 10 계산 실패: {exc}")


# ---------------------------------------------------------
# 4. 공항 순위 레이스
# ---------------------------------------------------------
REGION_COLORS = {"US": "#4C78A8", "Europe": "#F58518", "Korea": "#54A24B", "Other": "#9D9D9D"}


@st.cache_data(show_spinner=False)
def _race_table(model_name, ctx_mtime, model_mtime):
    """공항별 '최근 12개월 누적 이용객'(연간 규모)의 월별 표 + 예측 여부 표"""
    ctx = load_pickle(CONTEXT_FILE)
    bundle = load_pickle(SAVE_DIR / MODEL_FILES[model_name])
    filled, last_obs = load_trainer().forecast_all_filled(bundle, ctx, n_months=12)
    annual = filled.T.rolling(12, min_periods=12).sum().T          # 각 달 = 그 달까지 12개월 합계
    # 그 12개월 안에 예측값이 하나라도 섞였으면 '예측' 표시
    is_forecast = pd.DataFrame(
        np.asarray(annual.columns)[None, :] > last_obs.to_numpy()[:, None], index=annual.index, columns=annual.columns,
    )
    return annual, is_forecast


def _race_frames(annual, is_forecast, meta, dates, top_n, highlight):
    frames, ranks = [], {}
    for d in dates:
        col = annual[d].dropna().sort_values(ascending=False)
        ranks[d] = pd.Series(np.arange(1, len(col) + 1), index=col.index)
        top = col.head(top_n)
        names = [meta.loc[c, "airport_name"] for c in top.index]
        fc = is_forecast.loc[top.index, d].to_numpy()
        frames.append({
            "date": d,
            "x": top.to_numpy() / 1e6,
            "y": np.arange(1, len(top) + 1),
            "text": [f"{'⭐ ' if c == highlight else ''}{n} ({c})  {v / 1e6:,.1f}M{' *' if f else ''}" for c, n, v, f in zip(top.index, names, top.to_numpy(), fc)],
            "color": [REGION_COLORS.get(meta.loc[c, "region"], "#9D9D9D") for c in top.index],
            "opacity": [0.55 if f else 1.0 for f in fc],
            "line": [3 if c == highlight else 0 for c in top.index],
            "forecast": bool(fc.any()),
        })
    return frames, ranks


def render_race(ctx):
    meta = ctx["meta"]
    last_date = ctx["last_date"]
    available = [m for m, f in MODEL_FILES.items() if (SAVE_DIR / f).exists()]
    if not available:
        st.info("모델 파일이 없습니다.")
        return

    st.markdown("#### 🏁 공항 순위 레이스")
    st.caption(
        "각 달의 막대 = 그 달까지 **최근 12개월 이용객 합계**(연간 규모). "
        "공항마다 마지막 실측 달 다음부터 12개월 넘게 모델 예측을 이어 붙였습니다 (나라마다 통계가 올라오는 시점이 달라요). "
        "흐린 막대와 `*` 표시는 예측값이 섞인 값이에요."
    )

    best = ctx["best_model"]
    c1, c2, c3, c4 = st.columns(4)
    model_name = c1.selectbox("예측 모델", available, index=available.index(best) if best in available else 0, key="race_model")
    region = c2.selectbox("지역", ["전체", "미국", "유럽", "한국"], key="race_region")
    top_n = c3.select_slider("표시할 공항 수", [5, 10, 15, 20], value=10, key="race_top")
    step = c4.selectbox("프레임 간격", ["분기", "월", "연"], key="race_step")

    with st.spinner("모든 공항의 미래 이용객을 예측하는 중입니다... (처음 한 번만 오래 걸려요)"):
        try:
            annual, is_forecast = _race_table(
                model_name, CONTEXT_FILE.stat().st_mtime, (SAVE_DIR / MODEL_FILES[model_name]).stat().st_mtime,
            )
        except Exception as exc:
            st.warning(f"레이스 데이터를 만들지 못했습니다: {exc}")
            return

    region_code = {"미국": "US", "유럽": "Europe", "한국": "Korea"}.get(region)
    if region_code:
        keep = [c for c in annual.index if meta.loc[c, "region"] == region_code]
        if not keep:
            st.info(f"{region} 공항 데이터가 없습니다.")
            return
        annual, is_forecast = annual.loc[keep], is_forecast.loc[keep]

    valid_dates = [d for d in annual.columns if annual[d].notna().sum() >= min(top_n, len(annual))]
    if not valid_dates:
        st.info("레이스를 그릴 데이터가 부족합니다.")
        return
    years = sorted({d.year for d in valid_dates})
    c5, c6 = st.columns([3, 1])
    start_year = c5.slider("시작 연도", years[0], years[-1], max(years[0], years[-1] - 10), key="race_start")
    order = list(annual[valid_dates[-1]].sort_values(ascending=False).index)
    highlight = c6.selectbox("응원할 공항 ⭐", ["(없음)"] + order, format_func=lambda c: c if c == "(없음)" else airport_label(meta, c), key="race_pick")

    months_step = {"월": 1, "분기": 3, "연": 12}[step]
    dates = [d for d in valid_dates if d.year >= start_year]
    last_frame = dates[-1]
    dates = [d for d in dates if ((last_frame.year - d.year) * 12 + last_frame.month - d.month) % months_step == 0]
    # 실제 마지막 달은 프레임에 꼭 넣어서 '지금 순위'를 볼 수 있게 한다
    if last_date in valid_dates and last_date not in dates and last_date.year >= start_year:
        dates = sorted(dates + [last_date])

    frames, ranks = _race_frames(annual, is_forecast, meta, dates, top_n, highlight)
    x_max = max(float(np.nanmax(f["x"])) for f in frames) * 1.45

    def bar(f):
        return go.Bar(
            x=f["x"], y=f["y"], orientation="h", text=f["text"], textposition="outside", cliponaxis=False,
            marker=dict(color=f["color"], opacity=f["opacity"], line=dict(width=f["line"], color="#E45756")),
            hovertemplate="%{text}<extra></extra>",
        )

    def title(f):
        tag = "📈 예측 포함" if f["forecast"] else "📊 실제"
        return f"{f['date']:%Y년 %m월} 기준 최근 12개월 이용객 TOP {top_n}  ·  {tag}"

    fig = go.Figure(
        data=[bar(frames[0])],
        frames=[go.Frame(data=[bar(f)], name=f"{f['date']:%Y-%m}", layout=go.Layout(title_text=title(f))) for f in frames],
    )
    fig.update_layout(
        title=title(frames[0]),
        height=120 + 34 * top_n,
        xaxis=dict(range=[0, x_max], title="최근 12개월 이용객 (백만 명)"),
        yaxis=dict(range=[top_n + 0.5, 0.5], showticklabels=False, title=""),
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False,
        updatemenus=[dict(
            type="buttons", direction="left", x=0, y=-0.08, xanchor="left", yanchor="top",
            buttons=[
                dict(label="▶ 재생", method="animate", args=[None, {"frame": {"duration": 350, "redraw": True}, "transition": {"duration": 250}, "fromcurrent": True}]),
                dict(label="⏸ 정지", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.18, y=-0.06, len=0.82, currentvalue={"prefix": "", "visible": False},
            steps=[dict(label=f"{f['date']:%Y-%m}", method="animate", args=[[f"{f['date']:%Y-%m}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]) for f in frames],
        )],
    )
    st.plotly_chart(fig, use_container_width=True)
    legend = " · ".join(f"<span style='color:{REGION_COLORS[r]}'>■</span> {REGION_LABELS[r]}" for r in ("US", "Europe", "Korea"))
    st.markdown(legend, unsafe_allow_html=True)

    # --- 예측 기간 순위 변동 ---
    if last_date in ranks and frames[-1]["date"] > last_date:
        end_date = frames[-1]["date"]
        now, future = ranks[last_date], ranks[end_date]
        col_now, col_future = f"{last_date:%Y-%m} 실제 순위", f"{end_date:%Y-%m} 예상 순위"
        rows = []
        for c in list(future.index[: max(top_n, 10)]):
            a, b = (int(now[c]) if c in now.index else None), int(future[c])
            if a is None:
                move = "—"
            elif a > b:
                move = f"▲ {a - b}"
            elif a < b:
                move = f"▼ {b - a}"
            else:
                move = "="
            rows.append({"공항": airport_label(meta, c), col_now: a, col_future: b, "변동": move})
        table = pd.DataFrame(rows)
        st.markdown("#### 🔮 앞으로 12개월, 순위는 어떻게 바뀔까?")
        movers = table[table["변동"].str.startswith(("▲", "▼"))]
        if movers.empty:
            st.info("예측 기간 동안 상위권 순위 변동은 없을 것으로 보입니다.")
        else:
            for _, r in movers.iterrows():
                show = st.success if r["변동"].startswith("▲") else st.warning
                show(f"{r['공항']}: {r[col_now]}위 → {r[col_future]}위 ({r['변동']})")
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.caption(
        "⚠️ 미국(BTS)·영국(CAA)·유럽(Eurostat)·한국은 집계 기준이 조금씩 달라서 지역 간 순위는 참고용입니다. "
        "폐쇄 등으로 가장 최신 데이터보다 12개월 넘게 끊긴 공항은 레이스에서 빠집니다."
    )
    if (meta.loc[annual.index, "region"] == "Korea").any():
        st.caption(KOREA_NOTE)


# ---------------------------------------------------------
# 탭 진입점
# ---------------------------------------------------------
def show_airport_demand():
    st.subheader("🛫 전 세계 공항 이용객 수 수요 예측")
    st.caption("미국(BTS T-100) · 유럽(Eurostat) · 영국(CAA) · 한국 공항의 월별 승객 수와 국가별 GDP 성장률로 학습한 머신러닝 수요 예측입니다.")

    if not (CONTEXT_FILE.exists() and COMPARISON_FILE.exists() and TRAIN2_SCRIPT.exists()):
        st.warning(
            "학습된 공항 모델이 없습니다. 터미널에서 순서대로 실행해주세요.\n\n"
            "1. `python ojh/collect/fetch_airports.py`  (공항별 월 승객 수 + GDP 수집)\n"
            "2. `python ojh/models/train2.py`  (학습, 평가, .pkl 저장)"
        )
        return

    ctx = load_pickle(CONTEXT_FILE)
    comparison = pd.read_csv(COMPARISON_FILE)

    tab_overview, tab_compare, tab_sim, tab_race = st.tabs(["🔍 파이프라인 개요", "🏆 모델 성능 비교", "🔮 이용객 예측 시뮬레이터", "🏁 공항 순위 레이스"])
    with tab_overview:
        render_overview(ctx, comparison)
    with tab_compare:
        render_comparison(ctx, comparison)
    with tab_sim:
        render_simulator(ctx, comparison)
    with tab_race:
        render_race(ctx)