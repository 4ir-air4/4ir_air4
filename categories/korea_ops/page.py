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

if store.table_path("icn_airline_monthly").exists():  # 파이프라인을 다시 돌리기 전에는 없다
    st.markdown("**항공사별 월 지연률** (2025년 이후 편수 상위 12개 항공사)")
    by_airline = load(store, "icn_airline_monthly")
    choices = list(by_airline.groupby("항공사").편수.sum().sort_values(ascending=False).index)
    left, right = st.columns([3, 1])
    picked = left.multiselect("항공사", choices, default=choices[:4])
    way = right.segmented_control("방향", ["도착", "출발"], default="도착") or "도착"
    shown = by_airline[by_airline.항공사.isin(picked) & (by_airline.direction == way)]
    st.line_chart(shown, x="ym", y="지연률", color="항공사", x_label="월", y_label="지연률", alt=f"항공사별 월 {way} 지연률")
    st.caption("항공사 이름은 인천 실시간 수집 자료의 한글 이름이고, 없으면 OpenFlights 영문 이름입니다. 출발 지연은 중앙값을 뺀 뒤 15분 이상입니다. 편수가 적은 달은 지연률이 크게 흔들립니다.")

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
