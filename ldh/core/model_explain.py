"""모델 설명 차트. 분류는 ROC·PR 곡선과 혼동 행렬, 회귀는 잔차, 트리 모델은 변수 중요도와 SHAP(ldh/scripts/explain_models.py 결과)."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

TOP = 12  # 중요도·SHAP 에 보여 줄 특성 수
UNIT = {"binary": "로그 오즈", "regression": "예측값 단위", "multiclass": "예측 클래스의 로그 오즈"}


@st.cache_data(show_spinner=False, max_entries=8)
def _pred(path, mtime, columns):
    return pd.read_parquet(path, columns=list(columns))


def _thin(df, n=400):
    return df.iloc[np.unique(np.linspace(0, len(df) - 1, min(n, len(df))).astype(int))]


@st.cache_data(show_spinner=False, max_entries=8)
def _curves(path, mtime):
    d = _pred(path, mtime, ("실제", "예측 점수"))
    y, s = d["실제"].to_numpy(), d["예측 점수"].to_numpy()
    fpr, tpr, _ = roc_curve(y, s)
    p, r, _ = precision_recall_curve(y, s)
    return (_thin(pd.DataFrame({"거짓 양성 비율": fpr, "참 양성 비율": tpr})), _thin(pd.DataFrame({"재현율": r, "정밀도": p})),
            roc_auc_score(y, s), average_precision_score(y, s), float(y.mean()))


@st.cache_data(show_spinner=False, max_entries=8)
def _confusion(path, mtime):
    d = _pred(path, mtime, ("실제", "예측")).astype(str)
    count = pd.crosstab(d["실제"], d["예측"])
    share = count.div(count.sum(axis=1), axis=0)
    return pd.DataFrame({"실제": np.repeat(count.index, count.shape[1]), "예측": np.tile(count.columns, count.shape[0]),
                         "건수": count.to_numpy().ravel(), "비율": share.to_numpy().ravel()})


@st.cache_data(show_spinner=False, max_entries=8)
def _residuals(path, mtime):
    d = _pred(path, mtime, ("실제", "예측")).dropna()
    err = d["예측"] - d["실제"]
    low, high = err.quantile([0.01, 0.99])
    return err[err.between(low, high)].to_frame("오차"), d.sample(min(3000, len(d)), random_state=0)


def _curve_chart(df, x, y, diagonal):
    line = alt.Chart(df).mark_line().encode(x=alt.X(f"{x}:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y(f"{y}:Q", scale=alt.Scale(domain=[0, 1])))
    guide = alt.Chart(pd.DataFrame({x: [0, 1], y: diagonal})).mark_line(strokeDash=[4, 4], color="gray").encode(x=f"{x}:Q", y=f"{y}:Q")
    return (guide + line).properties(height=280)


def _classification(path, mtime, kind, model):
    if kind == "binary":
        roc, pr, auc, ap, rate = _curves(path, mtime)
        left, right = st.columns(2)
        left.markdown(f"**ROC 곡선** (AUC {auc:.3f})")
        left.altair_chart(_curve_chart(roc, "거짓 양성 비율", "참 양성 비율", [0, 1]), alt=f"{model}의 ROC 곡선, AUC {auc:.3f}")
        right.markdown(f"**PR 곡선** (PR-AUC {ap:.3f})")
        right.altair_chart(_curve_chart(pr, "재현율", "정밀도", [rate, rate]), alt=f"{model}의 정밀도-재현율 곡선, PR-AUC {ap:.3f}")
        st.caption(f"점선: ROC는 무작위 예측, PR은 실제 양성 비율({rate:.1%})입니다. 양성이 드문 과제는 PR 곡선이 더 정확한 판단 기준입니다.")
    cm = _confusion(path, mtime)
    st.markdown("**혼동 행렬** (행: 실제, 칸 숫자: 실제 클래스 안의 비율)")
    base = alt.Chart(cm).encode(x=alt.X("예측:N", title="예측"), y=alt.Y("실제:N", title="실제"))
    heat = base.mark_rect().encode(color=alt.Color("비율:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]), legend=None),
                                   tooltip=["실제:N", "예측:N", alt.Tooltip("건수:Q", format=","), alt.Tooltip("비율:Q", format=".1%")])
    text = base.mark_text().encode(text=alt.Text("비율:Q", format=".0%"), color=alt.condition("datum.비율 > 0.5", alt.value("white"), alt.value("black")))
    st.altair_chart((heat + text).properties(height=max(160, 45 * cm["실제"].nunique())), alt=f"{model}의 혼동 행렬")
    if kind == "binary":
        st.caption("이진 분류는 저장된 기준값으로 0과 1을 나눈 결과입니다. 1이 양성입니다.")


def _regression(path, mtime, model):
    err, sample = _residuals(path, mtime)
    left, right = st.columns(2)
    left.markdown("**오차 분포** (예측 − 실제, 양끝 1% 제외)")
    left.altair_chart(alt.Chart(err).mark_bar().encode(x=alt.X("오차:Q", bin=alt.Bin(maxbins=50), title="오차"), y=alt.Y("count():Q", title="건수")).properties(height=280),
                      alt=f"{model}의 오차 분포")
    right.markdown(f"**실제 대 예측** (표본 {len(sample):,}건)")
    lo, hi = float(sample[["실제", "예측"]].min().min()), float(sample[["실제", "예측"]].max().max())
    dots = alt.Chart(sample).mark_circle(size=12, opacity=0.4).encode(x=alt.X("실제:Q", scale=alt.Scale(domain=[lo, hi])), y=alt.Y("예측:Q", scale=alt.Scale(domain=[lo, hi])))
    guide = alt.Chart(pd.DataFrame({"실제": [lo, hi], "예측": [lo, hi]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="실제:Q", y="예측:Q")
    right.altair_chart((guide + dots).properties(height=280), alt=f"{model}의 실제값 대 예측값 산점도")
    st.caption("오차가 0을 중심으로 좌우가 비슷하면 한쪽으로 치우치지 않은 것입니다. 점선 위의 점은 실제보다 크게, 아래는 작게 예측한 경우입니다.")


def _trees(store, name, kind, model):
    imp_path, shap_path = store.table_path(f"importance_{name}"), store.table_path(f"shap_{name}")
    if not shap_path.exists():
        st.info(f"변수 중요도와 SHAP이 아직 없습니다. `python -m ldh.scripts.explain_models {store.slug}`를 실행하세요.", icon=":material/hourglass_empty:")
        return
    shap = pd.read_parquet(shap_path)
    mean_abs = shap.assign(크기=shap.SHAP.abs()).groupby("특성", sort=False)["크기"].mean().sort_values(ascending=False)
    order = list(mean_abs.index[:TOP])
    imp = pd.read_parquet(imp_path).sort_values("중요도", ascending=False).head(TOP)
    left, right = st.columns(2)
    left.markdown("**SHAP 평균 영향** (|SHAP| 평균)")
    left.altair_chart(alt.Chart(mean_abs.head(TOP).rename("평균 영향").reset_index()).mark_bar().encode(
        x=alt.X("평균 영향:Q", title=f"평균 영향 ({UNIT[kind]})"), y=alt.Y("특성:N", sort=order, title=None)).properties(height=24 * len(order) + 40), alt=f"{model}의 특성별 SHAP 평균 영향")
    right.markdown("**변수 중요도** (분할 이득 비율)")
    right.altair_chart(alt.Chart(imp).mark_bar().encode(x=alt.X("중요도:Q", axis=alt.Axis(format="%"), title="분할 이득 비율"), y=alt.Y("특성:N", sort=list(imp.특성), title=None))
                       .properties(height=24 * len(imp) + 40), alt=f"{model}의 특성별 분할 이득 중요도")
    top = shap[shap.특성.isin(order)].sample(frac=1, random_state=0).groupby("특성").head(800)  # 특성마다 최대 800점
    top["흔들기"] = np.random.default_rng(0).uniform(-0.4, 0.4, len(top))
    n = shap.groupby("특성").size().iloc[0]
    st.markdown(f"**SHAP 분포** (평가 구간 표본 {n:,}건, 점 하나가 한 건)")
    st.altair_chart(alt.Chart(top).mark_circle(size=14, opacity=0.6).encode(
        x=alt.X("SHAP:Q", title=f"예측에 준 영향 ({UNIT[kind]})"), y=alt.Y("특성:N", sort=order, title=None), yOffset=alt.YOffset("흔들기:Q", scale=alt.Scale(domain=[-0.5, 0.5])),
        color=alt.Color("순위:Q", scale=alt.Scale(scheme="redblue", reverse=True, domain=[0, 1]), title="값 백분위"), tooltip=["특성:N", alt.Tooltip("값:Q", format=",.3f"), alt.Tooltip("SHAP:Q", format=",.3f")])
        .properties(height=34 * len(order) + 40), alt=f"{model}의 특성별 SHAP 값 분포")
    st.caption("오른쪽은 예측을 키운 경우, 왼쪽은 줄인 경우입니다. 빨강은 그 특성 값이 큰 건, 파랑은 작은 건입니다. 범주형 특성은 값이 코드라 색의 크고 작음에 의미가 없습니다. "
               "SHAP은 상관관계를 설명할 뿐 원인을 뜻하지 않습니다. 분할 이득 중요도는 특성이 함께 움직이면 한쪽으로 몰릴 수 있습니다.")


def explain_section(store, name, kind, model, key):
    """name: <과제>__<모델 키>. pred_<name>.parquet 과 ldh/scripts/explain_models.py 결과를 읽는다."""
    st.subheader("모델 설명")
    pred_path = store.table_path(f"pred_{name}")
    if pred_path.exists():
        mtime = pred_path.stat().st_mtime
        if kind == "regression":
            _regression(str(pred_path), mtime, model)
        else:
            _classification(str(pred_path), mtime, kind, model)
    else:
        st.info("이 모델의 건별 예측 파일이 없어 곡선과 오차 차트를 그릴 수 없습니다.", icon=":material/hourglass_empty:")
    if key == "ensemble":
        st.info("앙상블은 여러 모델 예측의 평균이라 SHAP을 따로 계산하지 않았습니다. 구성 모델 중 트리 모델을 고르면 볼 수 있습니다.", icon=":material/info:")
    elif key == "lstm":
        st.info("LSTM은 변수 중요도와 SHAP을 계산하지 않았습니다. 같은 과제의 트리 모델을 고르면 볼 수 있습니다.", icon=":material/info:")
    else:
        _trees(store, name, kind, model)
