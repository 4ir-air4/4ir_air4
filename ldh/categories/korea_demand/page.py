"""한국 수요·수송: 연간·월간 여객과 수요 예측."""
import streamlit as st

from ldh.core.ui import header, load, model_tables, notes, prediction_section, store_for, time_chart

store = store_for(__file__)
summary = header(store)

st.subheader("전국 항공 수송 (연간)")
national = load(store, "national")
left, right = st.columns(2)
left.markdown("**여객**")
time_chart(left, national, x="연도", y="여객", color="구분", y_label="명", alt="연도별 국내·국제 여객")
right.markdown("**운항**")
time_chart(right, national, x="연도", y="운항", color="구분", y_label="편", alt="연도별 국내·국제 운항")

st.subheader("인천공항")
monthly = load(store, "icn_monthly")
st.markdown("**월별 유임 여객**")
time_chart(st, monthly, x="ym", y="passengers", x_label="월", y_label="명", alt="인천 월별 유임 여객")
left, right = st.columns(2)
rename = {"country": "국가", "name": "항공사", "flights": "운항(편)", "passengers": "유임 여객(명)"}
left.markdown("**국가별 (최근 12개월)**")
left.dataframe(load(store, "icn_country").rename(columns=rename), hide_index=True, alt="인천 국가별 여객")
right.markdown("**항공사별 (최근 12개월)**")
right.dataframe(load(store, "icn_airline").rename(columns=rename), hide_index=True, alt="인천 항공사별 여객")

st.subheader("한국공항공사 공항")
left, right = st.columns(2)
rename = {"flights": "운항(편)", "passengers": "여객(명)"}
left.markdown("**공항별 여객 (최근 12개월)**")
left.bar_chart(load(store, "type_airport"), x="공항", y="passengers", sort="-passengers", horizontal=True, x_label="명", y_label="", alt="공항별 여객")
right.markdown("**기종별 (최근 12개월)**")
right.dataframe(load(store, "type_top").rename(columns=rename), hide_index=True, alt="기종별 운항과 여객")

st.subheader("제주공항")
left, right = st.columns(2)
left.markdown("**일별 여객**")
time_chart(left, load(store, "jeju_daily"), x="날짜", y="y", y_label="명", alt="제주공항 일별 여객")
right.markdown("**연간 여객**")
time_chart(right, load(store, "jeju_yearly"), x="연도별", y=["국내선 여객 (명)", "국제선 여객 (명)"], kind="bar", x_label="연도", y_label="명", alt="제주공항 연간 여객")

st.subheader("모델 성능")
model_tables(summary["models"])
prediction_section(store)
notes(summary)
