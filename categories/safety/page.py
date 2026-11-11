"""항공 안전: 미국 NTSB와 국내 사고 통계."""
import streamlit as st

from core.data import START_YEAR
from core.ui import header, load, model_tables, notes, prediction_section, store_for, time_chart

store = store_for(__file__)
summary = header(store)
PERCENT = st.column_config.NumberColumn(format="percent")

st.subheader("미국 사고 추이 (NTSB)")
yearly = load(store, "yearly")
left, right = st.columns(2)
left.markdown("**연간 사고와 사망 사고**")
long = yearly.melt("연도", ["사고", "사망 사고"], var_name="구분", value_name="건수")
time_chart(left, long, x="연도", y="건수", color="구분", y_label="건", alt="연간 사고와 사망 사고")
right.markdown("**사고 중 사망 사고 비율**")
time_chart(right, yearly, x="연도", y="사망 사고 비율", y_label="비율", alt="연간 사망 사고 비율")

st.subheader(f"어떤 조건에서 사망 사고 비율이 높은가 ({START_YEAR}년 이후)")
a, b, c, d = st.columns(4)
for col, name, title in ((a, "light", "조명"), (b, "wx", "기상"), (c, "category", "항공기 종류"), (d, "far", "운항 규정 (FAR)")):
    col.markdown(f"**{title}**")
    col.dataframe(load(store, name), hide_index=True, column_config={"사망 사고 비율": PERCENT}, alt=f"{title}별 사고와 사망 사고 비율")
st.caption("사고 30건 미만인 항목은 제외했습니다.")
st.markdown(f"**월별 사고 건수 ({START_YEAR}년 이후 누적)**")
st.bar_chart(load(store, "month"), x="월", y="사고", y_label="건", alt="월별 사고 건수")

st.subheader("국내·전 세계 사고 (국토교통부)")
korea = load(store, "korea_yearly")
st.markdown("**국내 연간 발생 건수**")
long = korea.melt("연도", ["사고발생건수", "준사고발생건수", "경량사고발생건수", "초경량사고발생건수"], var_name="구분", value_name="건수")
time_chart(st, long, x="연도", y="건수", color="구분", kind="bar", y_label="건", alt="국내 연간 사고 발생 건수")
a, b, c = st.columns(3)
a.markdown("**국내 사고 구분**")
a.dataframe(load(store, "korea_type"), hide_index=True, alt="국내 사고 구분별 건수")
b.markdown("**국내 비행 단계**")
b.dataframe(load(store, "korea_phase"), hide_index=True, alt="국내 비행 단계별 건수")
c.markdown("**전 세계 비행 단계 (최신판)**")
c.dataframe(load(store, "world_phase"), hide_index=True, alt="전 세계 비행 단계별 건수")

st.subheader("모델 성능")
model_tables(summary["models"])
prediction_section(store)
notes(summary)
