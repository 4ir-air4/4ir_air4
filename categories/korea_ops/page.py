"""한국 운항·지연: 인천 편별 지연과 한국공항공사 집계."""
import streamlit as st

from core.ui import header, load, model_tables, notes, prediction_section, store_for

store = store_for(__file__)
summary = header(store)
PERCENT = st.column_config.NumberColumn(format="percent")

st.subheader("인천공항 월별 추이")
monthly = load(store, "icn_monthly")
left, right = st.columns(2)
left.markdown("**지연률**")
left.line_chart(monthly, x="ym", y="지연률", color="direction", x_label="월", y_label="지연률", alt="인천 월별 지연률")
right.markdown("**운항 편수**")
right.line_chart(monthly, x="ym", y="편수", color="direction", x_label="월", y_label="편", alt="인천 월별 운항 편수")
st.caption("2024년 공백 구간은 원본 자료가 없는 기간입니다.")

st.subheader("인천공항 시간대·항공사·국가별 (2025년 이후)")
st.markdown("**예정 시각별 지연률**")
st.bar_chart(load(store, "icn_hour"), x="hour", y="지연률", color="direction", stack=False, x_label="예정 시각", y_label="지연률", alt="시간대별 지연률")
tab = st.segmented_control("인천 분류", ["항공사 코드", "국가"], default="항공사 코드") or "항공사 코드"
name, col = {"항공사 코드": ("icn_airline", "airline"), "국가": ("icn_country", "country")}[tab]
st.dataframe(load(store, name).rename(columns={col: tab, "direction": "방향"}).round(3), hide_index=True,
             column_config={"지연률": PERCENT}, alt=f"인천 {tab}별 지연 통계")

st.subheader("한국공항공사 전체 공항 (월 집계)")
trend = load(store, "kac_trend")
left, right = st.columns(2)
left.markdown("**월별 지연 편수**")
left.line_chart(trend, x="ym", y="지연편", color="국내_국제", x_label="월", y_label="편", alt="월별 지연 편수")
right.markdown("**여객 천 명당 지연 편수**")
right.line_chart(trend, x="ym", y="여객 천 명당 지연 편수", color="국내_국제", x_label="월", y_label="편", alt="여객 천 명당 지연 편수")
st.markdown("**시간대별 누적 지연 편수**")
st.bar_chart(load(store, "kac_hour"), x="시간대", y="지연편", color="국내_국제", stack=False, y_label="편", alt="시간대별 지연 편수")

st.subheader("공항별 지연·결항 현황 (집계 기간 불명)")
left, right = st.columns(2)
left.markdown("**지연**")
left.dataframe(load(store, "airport_delay"), hide_index=True, alt="공항별 지연 현황")
right.markdown("**결항**")
right.dataframe(load(store, "airport_cancel"), hide_index=True, alt="공항별 결항 현황")

st.subheader("모델 성능")
model_tables(summary["models"])
prediction_section(store)
notes(summary)
