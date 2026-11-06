"""유럽 공항: 교통량, ATFM·사전 지연, 추가 시간."""
import streamlit as st

from core.ui import header, load, model_tables, notes, prediction_section, store_for

store = store_for(__file__)
summary = header(store)
NAMES = ["택시아웃", "택시인", "접근 구역(ASMA)"]

st.subheader("교통량")
left, right = st.columns([2, 1])
left.markdown("**월별 운항 편수 (공항별 합계)**")
left.line_chart(load(store, "traffic_month"), x="월", y="편수", y_label="편", alt="월별 운항 편수")
right.markdown("**최다 운항 공항 (최근 12개월)**")
right.bar_chart(load(store, "traffic_top").head(10), x="APT_ICAO", y="편수", sort="-편수", horizontal=True, x_label="편", y_label="", alt="최다 운항 공항")

st.subheader("도착 ATFM 지연")
left, right = st.columns(2)
left.markdown("**월별 도착편당 지연**")
left.line_chart(load(store, "atfm_month"), x="월", y="도착편당 지연(분)", y_label="분/편", alt="월별 도착편당 ATFM 지연")
right.markdown("**원인 비중**")
right.area_chart(load(store, "atfm_cause"), x="YEAR", y="비중", color="원인", x_label="연도", y_label="비중", alt="연도별 ATFM 지연 원인 비중")
st.markdown("**지연이 큰 공항 (최근 12개월, 도착 5천 편 초과)**")
st.bar_chart(load(store, "atfm_top"), x="APT_ICAO", y="도착편당 지연(분)", sort="-도착편당 지연(분)", horizontal=True, x_label="분/편", y_label="", alt="공항별 도착 ATFM 지연")

st.subheader("사전 지연과 항로 지연")
left, right = st.columns(2)
left.markdown("**출발편당 사전 지연**")
left.line_chart(load(store, "pre_year"), x="YEAR", y="출발편당 지연(분)", color="지표", x_label="연도", y_label="분/편", alt="연도별 사전 지연")
right.markdown("**항공편당 항로 ATFM 지연**")
right.line_chart(load(store, "route_year"), x="YEAR", y="항공편당 지연(분)", x_label="연도", y_label="분/편", alt="연도별 항로 ATFM 지연")
st.markdown("**항로 지연이 큰 관제 구역 (최근 12개월)**")
st.bar_chart(load(store, "route_top"), x="ENTITY_NAME", y="항공편당 지연(분)", sort="-항공편당 지연(분)", horizontal=True, x_label="분/편", y_label="", alt="관제 구역별 항로 지연")

st.subheader("추가 시간 (기준 시간 대비)")
st.line_chart(load(store, "add_year"), x="YEAR", y="추가 시간(분/편)", color="지표", x_label="연도", y_label="분/편", alt="연도별 추가 시간")
for col, i in zip(st.columns(3), range(3)):
    col.markdown(f"**{NAMES[i]} 상위 공항 (최근 12개월)**")
    col.bar_chart(load(store, f"add_top_{i}"), x="APT_ICAO", y="추가 시간(분/편)", sort="-추가 시간(분/편)", horizontal=True, x_label="분/편", y_label="", alt=f"{NAMES[i]} 추가 시간 상위 공항")

st.subheader("모델 성능")
model_tables(summary["models"])
prediction_section(store)
notes(summary)
