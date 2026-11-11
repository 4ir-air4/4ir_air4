"""전 세계 항공: OpenSky 관측, 국가 통계, 참조 자료."""
import streamlit as st

from ldh.core.data import START_YEAR
from ldh.core.ui import header, load, model_tables, notes, prediction_section, store_for, time_chart

store = store_for(__file__)
summary = header(store)

st.subheader(f"OpenSky 관측 항공편 ({START_YEAR}~2022)")
monthly = load(store, "monthly")
time_chart(st, monthly, x="월", y="관측 편수", y_label="편", alt="월별 관측 항공편")
left, right = st.columns(2)
left.markdown("**출발 국가별 (2022년 상위 15개국)**")
left.dataframe(load(store, "country").rename(columns={"country": "국가"}), hide_index=True, alt="국가별 연간 관측 항공편")
right.markdown("**기종별 (2022년)**")
right.bar_chart(load(store, "type_top"), x="typecode", y="n", sort="-n", horizontal=True, x_label="편", y_label="", alt="기종별 관측 편수")
st.markdown("**노선별 (2022년, 공항 ICAO 코드)**")
st.dataframe(load(store, "route_top").rename(columns={"origin": "출발", "destination": "도착", "n": "편수", "year": "연도"}), hide_index=True, alt="노선별 관측 편수")

st.subheader("국가별 항공 통계 (세계은행)")
wb = load(store, "worldbank")
metric = st.segmented_control("지표", sorted(wb.지표.unique()), default="여객(명)") or "여객(명)"
part = wb[wb.지표 == metric]
left, right = st.columns(2)
left.markdown("**세계**")
time_chart(left, part[part.국가 == "세계"], x="연도", y="값", y_label=metric, alt=f"세계 {metric}")
right.markdown("**주요 국가**")
time_chart(right, part[part.국가 != "세계"], x="연도", y="값", color="국가", y_label=metric, alt=f"주요 국가 {metric}")

st.subheader("참조 자료")
a, b, c, d = st.columns(4)
a.markdown("**공항 종류**")
a.dataframe(load(store, "airport_type"), hide_index=True, alt="공항 종류별 수")
b.markdown("**정기편 공항 수 상위 국가**")
b.dataframe(load(store, "airport_country"), hide_index=True, alt="국가별 정기편 공항 수")
c.markdown("**항공기 제조사**")
c.dataframe(load(store, "manufacturer"), hide_index=True, alt="제조사별 등록 대수")
d.markdown("**항공사별 노선 수 (2014년 기준)**")
d.dataframe(load(store, "openflights_airline"), hide_index=True, alt="항공사별 노선 수")

st.subheader("모델 성능")
model_tables(summary["models"])
prediction_section(store)
notes(summary)
