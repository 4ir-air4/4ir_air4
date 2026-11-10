"""학습한 모델 비교. 모델을 골라 지표와 예측값을 보고, 같은 과제의 다른 모델과의 차이를 본다."""
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from core.model_explain import explain_section
from core.models import MODEL_NAMES, collect_rows
from core.registry import discover
from core.store import OUTPUT_DIR, Store
from core.ui import COMMON, PREVIEW_ROWS, SECTIONS, _preview, load

SOURCES = {"전체 학습": OUTPUT_DIR, "2026년 학습": OUTPUT_DIR.parent / "outputs_2026"}
METRICS = {"binary": ["pr_auc", "roc_auc", "precision", "recall"], "regression": ["mae", "r2", "improve"], "multiclass": ["macro_f1", "accuracy"]}  # 첫 값이 기준 지표
REFERENCE = {"binary": "baseline_pr_auc", "regression": "baseline_mae", "multiclass": "majority_rate"}
KEYS = {v: k for k, v in MODEL_NAMES.items()}


def show(metric, value):
    return "-" if pd.isna(value) else f"{value:.1%}" if metric == "improve" else f"{value:,.3f}"


def label(kind, metric):
    return SECTIONS[kind][1][metric]


def below_baseline(g):
    """한 과제의 모든 모델이 기준선보다 나쁜가. 다중 분류는 같은 지표의 기준값이 없어 판단하지 않는다."""
    k = g.kind.iloc[0]
    if k == "multiclass":
        return False
    m, ref = METRICS[k][0], REFERENCE[k]
    return g[m].min() > g[ref].min() if k == "regression" else g[m].max() <= g[ref].max()


def results_of(root):
    """결과가 있는 카테고리: {slug: (이름, Store, 결과 행)}."""
    found = {}
    for c in discover()[0]:
        store = Store(c["slug"])
        store.dir = root / c["slug"]
        rows = collect_rows(store)
        if rows:
            found[c["slug"]] = (c["title"], store, rows)
    return found


available = {name: results for name, root in SOURCES.items() if (results := results_of(root))}
if not available:
    st.info("아직 학습 결과가 없습니다. 드라이브의 `outputs` 를 이 프로젝트로 복사하거나 `run_models.py` 를 실행하세요.", icon=":material/hourglass_empty:")
    st.stop()
source = st.segmented_control("결과", list(available), default=next(iter(available)), required=True)
results = available[source]
left, right = st.columns(2)
slug = left.selectbox("카테고리", list(results), format_func=lambda s: results[s][0])
_, store, rows = results[slug]
frame = pd.DataFrame(rows)
titles = dict(zip(frame.task, frame.title))
task = right.selectbox("예측 대상", list(titles), format_func=titles.get)
part = frame[frame.task == task].copy()
kind = part.kind.iloc[0]
main = METRICS[kind][0]
lower_better = main == "mae"
part = part.sort_values(main, ascending=lower_better)
done = [m for m in MODEL_NAMES.values() if m in set(part.model)]
model = st.segmented_control("모델", done, default=part.model.iloc[0], required=True, key=f"model_{slug}_{task}")
row = part[part.model == model].iloc[0]
others = part[part.model != model]

st.subheader(f"{model} 결과")
with st.container(horizontal=True):
    for metric in METRICS[kind]:
        value = row.get(metric)
        rivals = others[metric].dropna() if metric in others else pd.Series(dtype=float)
        low = metric == "mae"
        best = None if rivals.empty else (rivals.min() if low else rivals.max())
        delta = None if best is None or pd.isna(value) else float(value - best)
        st.metric(label(kind, metric), show(metric, value), delta=None if delta is None else f"{delta:+.1%}" if metric == "improve" else round(delta, 3),
                  delta_color="inverse" if low else "normal", border=True, help="화살표는 다른 모델 중 가장 좋은 값과의 차이입니다." if delta is not None else None)
    st.metric(label(kind, REFERENCE[kind]), show(REFERENCE[kind], row.get(REFERENCE[kind])), border=True,
              help="가장 흔한 값만 고를 때의 정확도입니다. 정확도와 비교하며 macro F1과는 직접 비교할 수 없습니다." if kind == "multiclass" else "모델 없이 얻을 수 있는 기준값입니다.")
st.caption(f"학습 {row.train_period} ({row.n_train:,}건) · 평가 {row.test_period} ({row.n_test:,}건) · 구성 {row.group} · 걸린 시간 {row.seconds:,}초")
if below_baseline(part):
    st.warning("이 과제는 모든 모델이 기준선보다 나쁩니다. 예측값으로는 기준선을 쓰세요.", icon=":material/warning:")

st.subheader("모델별 차이")
if others.empty:
    st.info("이 과제는 아직 이 모델만 학습했습니다. 다른 모델이 끝나면 여기에서 비교할 수 있습니다.", icon=":material/hourglass_empty:")
else:
    left, right = st.columns(2)
    left.markdown(f"**{label(kind, main)}** ({'낮을수록' if lower_better else '높을수록'} 좋음)")
    left.bar_chart(part, x="model", y=main, sort=False, horizontal=True, height=220, x_label="모델", y_label=label(kind, main), alt=f"모델별 {label(kind, main)}")
    right.markdown("**걸린 시간**")
    right.bar_chart(part, x="model", y="seconds", sort=False, horizontal=True, height=220, x_label="모델", y_label="초", alt="모델별 걸린 시간")
columns = {"model": "모델"} | {m: label(kind, m) for m in METRICS[kind] if m in part} | {"seconds": "걸린 시간(초)", "n_train": COMMON["n_train"]}
table = part.reindex(columns=list(columns)).rename(columns=columns)
table.insert(2, f"{model} 대비 차이", part[main].to_numpy() - row[main])
config = {"기준선 대비 개선율": st.column_config.NumberColumn(format="percent"), COMMON["n_train"]: st.column_config.NumberColumn(format="localized")}
st.dataframe(table.round(3), hide_index=True, column_config=config, alt=f"{titles[task]}의 모델별 지표와 선택 모델 대비 차이")
st.caption("LSTM은 실행마다 결과가 조금씩 달라지므로, 작은 차이를 모델 간 우열로 읽지 마세요.")

st.subheader("예측값")
name = f"{task}__{KEYS[model]}"
agg_tables = {m: load(store, f"predagg_{task}__{KEYS[m]}") for m in done if store.table_path(f"predagg_{task}__{KEYS[m]}").exists()}
if kind == "multiclass" and model in agg_tables:
    agg = agg_tables[model]
    pick = st.selectbox("클래스", sorted(agg.클래스.unique()), key=f"{name}_class")
    st.markdown(f"**기간별 실제 대 {model} 예측 비율**")
    st.line_chart(agg[agg.클래스 == pick], x="기간", y="비율", color="구분", x_label="기간", y_label="비율", alt=f"{pick} 실제 비율과 예측 비율")
elif agg_tables:
    first = next(iter(agg_tables.values()))
    long = pd.concat([first[["기간", "실제"]].rename(columns={"실제": "값"}).assign(구분="실제")] + [t[["기간", "예측"]].rename(columns={"예측": "값"}).assign(구분=m) for m, t in agg_tables.items()])
    st.markdown("**기간별 실제와 모델별 예측** (평가 구간 평균)")
    st.line_chart(long, x="기간", y="값", color="구분", x_label="기간", y_label="평균", alt="기간별 실제 평균과 모델별 예측 평균")
    if kind == "binary":
        st.caption("이진 분류는 실제 양성 비율과 평균 예측 점수입니다. 가중치 학습 모델의 점수는 실제 확률이 아닙니다.")

forecast_path = store.table_path(f"forecast_{name}")
if forecast_path.exists():
    forecast = load(store, f"forecast_{name}").drop(columns="실제", errors="ignore")
    st.markdown(f"**{model}의 다음 기간 예측값 ({forecast['기간'].iloc[0]})**")
    st.dataframe(forecast.sort_values("예측", ascending=False).round(2), hide_index=True, alt=f"{model}의 다음 기간 예측값")

pred_path = store.table_path(f"pred_{name}")
if pred_path.exists():
    n = pq.ParquetFile(pred_path).metadata.num_rows
    st.markdown(f"**{model}의 건별 예측값** (전체 {n:,}건 중 앞 {min(n, PREVIEW_ROWS):,}건)")
    st.dataframe(_preview(str(pred_path), pred_path.stat().st_mtime, PREVIEW_ROWS).round(3), hide_index=True, alt=f"{model}의 평가 구간 건별 실제값과 예측값")
    st.caption(f"전체 파일: `{store.dir.parent.name}/{slug}/pred_{name}.parquet`")

explain_section(store, name, kind, model, KEYS[model])

st.subheader("이 카테고리의 과제별 비교")
summary = []
for t, g in frame.groupby("task", sort=False):
    k = g.kind.iloc[0]
    metric = METRICS[k][0]
    low = metric == "mae"
    values = g.set_index("model")[metric]
    summary.append({"예측 대상": g.title.iloc[0], "지표": label(k, metric), **values.to_dict(), "최고 모델": values.idxmin() if low else values.idxmax(),
                    "기준": None if k == "multiclass" else g[REFERENCE[k]].iloc[0], "판단": "기준선 사용 권장" if below_baseline(g) else ""})
overview = pd.DataFrame(summary).reindex(columns=["예측 대상", "지표", *[m for m in MODEL_NAMES.values() if m in set(frame.model)], "최고 모델", "기준", "판단"])
st.dataframe(overview.round(3), hide_index=True, alt="과제별 모델 기준 지표, 최고 모델, 기준선 대비 판단")
st.caption("빈칸은 아직 학습하지 않은 모델입니다. MAE는 낮을수록, 나머지 지표는 높을수록 좋습니다. 기준은 회귀는 기준선 MAE, 이진 분류는 무작위 PR-AUC이고, 다중 분류는 같은 지표의 기준이 없어 비웠습니다.")
