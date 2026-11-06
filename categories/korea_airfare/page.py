"""한국 국내선 항공료: 계절성, 비용·수요 지표, 월별 예측."""
import streamlit as st

from core.ui import header, load, model_tables, notes, store_for

store = store_for(__file__)
summary = header(store)

monthly = load(store, "monthly")
seasonal = load(store, "seasonal_month")
seasonal["월"] = seasonal["월"].astype(str).str.zfill(2) + "월"

st.subheader("월별·계절별 항공료")
left, right = st.columns(2)
left.markdown("**월별 항공료** (연평균 대비)")
left.bar_chart(seasonal.assign(차이=seasonal["항공료 계절 지수"] - 100), x="월", y="차이", y_label="%", alt="월별 항공료의 연평균 대비 차이")
right.markdown("**계절별 평균** (연평균 100)")
right.dataframe(load(store, "seasonal_season").round(1), hide_index=True, alt="계절별 항공료와 여객 계절 지수")
right.caption("봄 3-5월, 여름 6-8월, 가을 9-11월, 겨울 12-2월")

st.dataframe(seasonal.round(1), hide_index=True, alt="월별 계절 지수 상세 표")

st.subheader("추이")
st.markdown("**항공료 지수**")
st.line_chart(monthly, x="월", y=["항공료 지수", "실질 항공료", "국내항공여객 생산자물가"], y_label="2020=100", alt="항공료 지수, 실질 항공료, 생산자물가 추이")
left, right = st.columns(2)
left.markdown("**제트유·원유**")
left.line_chart(monthly, x="월", y=["제트유", "원유(달러)"], y_label="2020=100", alt="제트유와 원유 지수 추이")
right.markdown("**원/달러 환율**")
right.line_chart(monthly, x="월", y="원/달러 환율", y_label="원", alt="원달러 환율 추이")
st.markdown("**국내선 여객**")
st.line_chart(monthly, x="월", y="여객(명)", y_label="명", alt="국내선 월별 여객 추이")

st.subheader("지표가 항공료보다 얼마나 앞서는가")
st.caption("전년 대비 변화율끼리의 상관계수입니다. 시차 2는 지표의 2개월 전 값과 비교한 것입니다.")
st.line_chart(load(store, "lag_corr"), x="시차(개월)", y="상관계수", color="지표", alt="지표별 시차 상관계수")

st.subheader("예측 결과")
predictions = load(store, "predictions")
horizon = st.segmented_control("예측 시점", ["1개월 전", "3개월 전"], default="1개월 전") or "1개월 전"
shown = ["전체", "직전 값 그대로", "작년 같은 달 값"]  # 8개 모델을 모두 그리면 선을 구분할 수 없다. 전체 비교는 아래 표에 있다
chart = predictions[["월", "실제"]].join(predictions[[f"{horizon}: {name}" for name in shown]].set_axis(shown, axis=1))
st.line_chart(chart, x="월", y=["실제", *shown], y_label="지수", alt=f"{horizon} 예측과 실제 항공료 지수")
compare = load(store, "model_compare")
st.dataframe(compare[compare["예측 시점"] == horizon].round(2), hide_index=True, alt="모델별 오차 비교 표")
with st.expander("1개월 뒤 모델의 표준화 계수"):
    st.dataframe(load(store, "coefficients").round(4), hide_index=True, alt="1개월 뒤 모델의 표준화 계수")

st.subheader("모델 성능")
model_tables(summary["models"])
notes(summary)
