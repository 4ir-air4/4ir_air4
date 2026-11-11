"""outputs/ 의 결과만 읽어 그린다. 제목은 app.py 가 그리므로 st.title 을 쓰지 않는다."""
import streamlit as st

from core.ui import header, load, model_tables, notes, store_for, time_chart

store = store_for(__file__)
summary = header(store)

st.subheader("연도별 건수")
time_chart(st, load(store, "yearly"), x="연도", y="건수", alt="연도별 건수 추이")

st.subheader("모델 성능")
model_tables(summary["models"])
notes(summary)
