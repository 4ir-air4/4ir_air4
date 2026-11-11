"""카테고리 화면 공통 요소. page.py 는 이 함수들과 Streamlit 기본 차트만으로 작성한다."""
from pathlib import Path

import altair
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from core.data import COVID
from core.models import MODEL_NAMES, collect_rows
from core.store import Store

COVID_NOTE = f"코로나 시기 {COVID[0]}~{COVID[1]}"

BINARY = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "baseline_pr_auc": "무작위 PR-AUC", "lift": "무작위 대비 배수",
          "precision": "정밀도", "recall": "재현율"}
REGRESSION = {"mae": "MAE", "baseline_mae": "기준선 MAE", "improve": "기준선 대비 개선율", "r2": "R²"}
MULTICLASS = {"accuracy": "정확도", "majority_rate": "최빈값 정확도", "macro_f1": "Macro F1"}
COMMON = {"title": "예측 대상", "model": "모델", "group": "구성", "train_period": "학습", "test_period": "평가", "n_train": "학습 건수", "n_test": "평가 건수"}
PREVIEW_ROWS = 2000
SECTIONS = {"binary": ("이진 분류", BINARY), "regression": ("회귀", REGRESSION), "multiclass": ("다중 분류", MULTICLASS)}


@st.cache_data(max_entries=200)
def _read_table(path, mtime):
    return pd.read_parquet(path)


def store_for(page_file):
    """page.py 위치의 폴더 이름이 카테고리 이름이다."""
    return Store(Path(page_file).resolve().parent.name)


def load(store, name):
    path = store.table_path(name)
    return _read_table(str(path), path.stat().st_mtime)


def header(store):
    """요약 문장과 KPI, 주의 사항을 그린다. 결과가 없으면 실행 방법을 안내하고 멈춘다."""
    summary = store.summary()
    if summary is None:
        st.info(f"아직 결과가 없습니다. `python run_all.py {store.slug}` 를 실행하세요.", icon=":material/hourglass_empty:")
        st.stop()
    st.write(summary["headline"])
    if summary["kpis"]:
        with st.container(horizontal=True):
            for k in summary["kpis"]:
                st.metric(k["label"], k["value"], help=k["help"], border=True)
    st.caption(f"결과 갱신 {summary['updated']} · 데이터: {', '.join(summary['sources'])}")
    return summary


def notes(summary):
    if summary["notes"]:
        with st.expander("이 카테고리 결과를 읽을 때 주의할 점", icon=":material/warning:"):
            for n in summary["notes"]:
                st.markdown(f"- {n}")


def model_tables(models, show_category=False):
    """결과 행을 종류별 표로 그린다. 홈 화면과 카테고리 화면이 같이 쓴다."""
    frame = pd.DataFrame(models)
    if frame.empty:
        st.caption("이 카테고리에는 학습한 모델이 없습니다.")
        return
    for kind, (label, cols) in SECTIONS.items():
        part = frame[frame.kind == kind]
        if part.empty:
            continue
        st.markdown(f"**{label}**")
        columns = ({"category": "카테고리"} if show_category else {}) | COMMON | cols
        shown = part.reindex(columns=list(columns)).rename(columns=columns)
        config = {"기준선 대비 개선율": st.column_config.NumberColumn(format="percent")}
        st.dataframe(shown.round(3), hide_index=True, column_config=config, alt=f"{label} 모델 성능 표")
    if (frame.kind == "regression").any():
        st.caption("회귀 기준선은 학습 평균과 과거 값 그대로 쓰기 중 오차가 작은 쪽입니다. 개선율이 음수면 기준선이 더 낫습니다.")
    for row in frame[frame.note.fillna("") != ""].drop_duplicates("note").itertuples() if "note" in frame else []:
        st.caption(f"{row.title}: {row.note}")


@st.cache_data(max_entries=50)
def _preview(path, mtime, rows):
    """평가 구간이 수백만 건이어도 앞부분만 읽어 화면에 보여 준다. 전체는 parquet 파일에 있다."""
    batch = next(pq.ParquetFile(path).iter_batches(batch_size=rows))
    return batch.to_pandas()


def prediction_section(store):
    """모델 비교, 기간별 실제 대 예측, 건별 예측값, 다음 기간 예측값."""
    rows = collect_rows(store)
    st.subheader("예측값")
    if not rows:
        st.info(f"아직 4개 모델 결과가 없습니다. `python run_models.py {store.slug}` 를 실행하세요.", icon=":material/hourglass_empty:")
        return
    frame = pd.DataFrame(rows)
    titles = dict(zip(frame.task, frame.title))
    task = st.selectbox("예측 대상", list(titles), format_func=titles.get, key=f"{store.slug}_task")
    part = frame[frame.task == task]
    kind = part.kind.iloc[0]
    st.markdown("**모델 비교**")
    key_metric = {"binary": "pr_auc", "regression": "mae", "multiclass": "macro_f1"}[kind]
    best = part.sort_values(key_metric, ascending=kind == "regression").iloc[0]
    st.caption(f"평가 구간 기준 최고 모델: {best.model} ({SECTIONS[kind][1][key_metric]} {best[key_metric]:.3f})")
    model_tables(part.to_dict("records"))
    names = {v: k for k, v in MODEL_NAMES.items()}
    done = part.model.tolist()
    model = st.segmented_control("모델", done, default=best.model, key=f"{store.slug}_{task}_model") or best.model
    name = f"{task}__{names[model]}"

    agg_path = store.table_path(f"predagg_{name}")
    if agg_path.exists():
        agg = load(store, f"predagg_{name}")
        st.markdown("**기간별 실제 대 예측** (평가 구간)")
        if kind == "multiclass":
            top = agg[agg.구분 == "예측"].groupby("클래스").비율.mean().nlargest(1).index[0]
            pick = st.selectbox("클래스", sorted(agg.클래스.unique()), index=sorted(agg.클래스.unique()).index(top), key=f"{name}_class")
            time_chart(st, agg[agg.클래스 == pick], x="기간", y="비율", color="구분", y_label="비율", alt=f"{pick} 실제 비율과 예측 비율")
        else:
            long = agg.melt("기간", ["실제", "예측"], var_name="구분", value_name="값")
            time_chart(st, long, x="기간", y="값", color="구분", y_label="평균", alt="기간별 실제 평균과 예측 평균")
            st.caption("이진 분류는 실제 양성 비율과 평균 예측 점수입니다." + (" 가중치 학습 모델의 점수는 실제 확률이 아닙니다." if kind == "binary" else ""))

    forecast_path = store.table_path(f"forecast_{name}")
    if forecast_path.exists():
        forecast = load(store, f"forecast_{name}").drop(columns="실제", errors="ignore")
        st.markdown(f"**다음 기간 예측값 ({forecast['기간'].iloc[0]})**")
        if kind == "regression":
            st.metric("예측 합계", f"{forecast['예측'].sum():,.1f}", border=True)
        st.dataframe(forecast.sort_values("예측", ascending=False).round(2), hide_index=True, alt="다음 기간 예측값")

    pred_path = store.table_path(f"pred_{name}")
    if pred_path.exists():
        n = pq.ParquetFile(pred_path).metadata.num_rows
        st.markdown(f"**건별 예측값** (전체 {n:,}건 중 앞 {min(n, PREVIEW_ROWS):,}건)")
        st.dataframe(_preview(str(pred_path), pred_path.stat().st_mtime, PREVIEW_ROWS).round(3), hide_index=True, alt="평가 구간의 건별 실제값과 예측값")
        st.caption(f"전체 파일: `{store.dir.parent.name}/{store.slug}/pred_{name}.parquet`")


def rate(n, d):
    return (n / d.where(d > 0)).astype("float64")


def in_covid(values):
    """연도(정수), 'YYYY-MM', 'YYYYMM', 'YYYY-MM-DD' 값이 코로나 시기에 걸치는지. 연도는 일부만 걸쳐도 넣는다(2020~2022)."""
    v = pd.Series(values)
    if pd.api.types.is_integer_dtype(v):
        return v.between(COVID[0].year, COVID[1].year)
    t = pd.to_datetime(v.astype(str).str.replace(r"^(\d{4})(\d{2})$", r"\1-\2", regex=True), errors="coerce")
    return t.between(COVID[0].start_time, COVID[1].end_time)


def time_chart(where, data, x, y, color=None, kind="line", x_label=None, y_label=None, stack=None, height=None, alt=None):
    """st.line_chart·area_chart·bar_chart 대신 쓴다. 코로나 시기가 들어 있으면 그 구간에 회색 띠를 깔고, 없으면 기본 차트 그대로다."""
    covid = in_covid(data[x])
    if not covid.any():
        kw = {k: v for k, v in {"height": height, "stack": stack if kind != "line" else None}.items() if v is not None}  # None을 넘기면 오류
        return getattr(where, f"{kind}_chart")(data, x=x, y=y, color=color, x_label=x_label, y_label=y_label, alt=alt, **kw)
    covid_x = sorted(data.loc[covid.values, x].unique())
    if isinstance(y, list):
        data, y, color = data.melt(x, y, var_name="구분", value_name="값"), "값", "구분"
    title = lambda t: altair.Undefined if t is None else t  # noqa: E731
    scale = altair.Scale(paddingInner=0, paddingOuter=0)  # 띠가 달마다 끊기지 않게
    xenc = altair.X(f"{x}:O", title=title(x_label), scale=scale, axis=altair.Axis(labelOverlap=True))
    yenc = altair.Y(f"{y}:Q", title=title(y_label), stack=altair.Undefined if stack is None else {True: "zero", False: None}.get(stack, stack),
                    axis=altair.Axis(format="%") if stack == "normalize" else altair.Undefined)
    base = altair.Chart(data).encode(x=xenc, y=yenc, color=altair.Color(f"{color}:N") if color else altair.Undefined,
                                     tooltip=[x, y] + ([color] if color else []))
    marks = {"line": base.mark_line(), "area": base.mark_area(), "bar": base.mark_bar(width=altair.RelativeBandSize(0.8))}
    band = altair.Chart(pd.DataFrame({x: covid_x})).mark_rect(color="gray", opacity=0.15).encode(
        x=altair.X(f"{x}:O", scale=scale), tooltip=altair.value(COVID_NOTE))
    label = altair.Chart(pd.DataFrame({x: covid_x[:1], "글": ["코로나 시기"]})).mark_text(
        align="left", baseline="bottom", y=altair.ExprRef("height - 2"), color="gray", fontSize=11  # 선은 대개 위쪽에 있어 아래에 쓴다
    ).encode(x=altair.X(f"{x}:O", scale=scale), text="글:N")
    chart = altair.layer(band, marks[kind], label)
    where.altair_chart(chart.properties(height=height) if height else chart, width="stretch", alt=alt)
