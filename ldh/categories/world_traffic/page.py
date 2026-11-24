"""전 세계 항공: OpenSky 관측, 국가 통계, 참조 자료."""
import altair
import streamlit as st

from ldh.core.data import START_YEAR
from ldh.core.ui import KR_AXIS, header, load, model_tables, notes, prediction_section, store_for, time_chart

store = store_for(__file__)


def table(where, df, alt):
    """숫자 열에 천 단위 쉼표를 넣어 보여 준다 (연도 열은 제외)."""
    fmt = {c: st.column_config.NumberColumn(format="localized") for c in df.select_dtypes("number") if c != "연도"}
    where.dataframe(df, hide_index=True, column_config=fmt, alt=alt)


summary = header(store)

st.subheader(f"OpenSky 관측 항공편 ({START_YEAR}~2022)")
monthly = load(store, "monthly")
time_chart(st, monthly, x="월", y="관측 편수", y_label="편", alt="월별 관측 항공편")
left, right = st.columns(2)
left.markdown("**출발 국가별 (2022년 상위 15개국)**")
table(left, load(store, "country").rename(columns={"country": "국가"}), alt="국가별 연간 관측 항공편")
right.markdown("**기종별 (2022년)**")
right.altair_chart(altair.Chart(load(store, "type_top")).mark_bar().encode(
    x=altair.X("n:Q", title="편", axis=KR_AXIS), y=altair.Y("typecode:N", sort="-x", title=None), tooltip=["typecode", "n"]),
    width="stretch", alt="기종별 관측 편수")
st.markdown("**노선별 (2022년, 공항 ICAO 코드)**")
table(st, load(store, "route_top").rename(columns={"origin": "출발", "destination": "도착", "n": "편수", "year": "연도"}), alt="노선별 관측 편수")

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
table(a, load(store, "airport_type"), alt="공항 종류별 수")
b.markdown("**정기편 공항 수 상위 국가**")
table(b, load(store, "airport_country"), alt="국가별 정기편 공항 수")
c.markdown("**항공기 제조사**")
table(c, load(store, "manufacturer"), alt="제조사별 등록 대수")
d.markdown("**항공사별 노선 수 (2014년 기준)**")
table(d, load(store, "openflights_airline"), alt="항공사별 노선 수")

st.subheader("모델 성능")
model_tables(summary["models"])
prediction_section(store)
notes(summary)
