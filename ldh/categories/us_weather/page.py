"""미국 기상과 운항: 날씨별 지연·결항률과 기상 데이터의 예측 기여도."""
import streamlit as st

from ldh.core.ui import header, load, model_tables, notes, prediction_section, store_for

store = store_for(__file__)
summary = header(store)
PERCENT = st.column_config.NumberColumn(format="percent")

st.subheader("날씨별 지연·결항률")
cond = load(store, "condition")
side = st.segmented_control("어느 공항의 날씨", ["출발 공항", "도착 공항"], default="출발 공항") or "출발 공항"
part = cond[cond.공항 == side].sort_values("지연률", ascending=False)
left, right = st.columns(2)
left.markdown("**지연률**")
left.bar_chart(part, x="날씨", y="지연률", sort="-지연률", horizontal=True, x_label="지연률", y_label="", alt="날씨별 지연률")
right.markdown("**결항률**")
right.bar_chart(part, x="날씨", y="결항률", sort="-결항률", horizontal=True, x_label="결항률", y_label="", alt="날씨별 결항률")

st.subheader("시정·풍속·기온 구간별 (출발 공항)")
bins = load(store, "bins")
for col, name in zip(st.columns(3), ["시정", "풍속", "기온"]):
    col.markdown(f"**{name}**")
    col.bar_chart(bins[bins.항목 == name], x="구간", y="지연률", sort=False, x_label="", y_label="지연률", alt=f"{name} 구간별 지연률")

st.subheader("공항별 악천후 영향 (편수 상위 40개)")
a = load(store, "airport").rename(columns={"지연률_악천후": "악천후 지연률", "지연률_그 외": "그 외 지연률", "결항률_악천후": "악천후 결항률", "결항률_그 외": "그 외 결항률"})
cols = ["Origin", "악천후 비중", "악천후 지연률", "그 외 지연률", "악천후 결항률", "그 외 결항률"]
st.dataframe(a[cols].rename(columns={"Origin": "공항"}).round(3), hide_index=True, alt="공항별 악천후 지연·결항률",
             column_config={c: PERCENT for c in cols[1:]})
st.caption("악천후: 뇌우, 눈, 결빙, 안개·박무 중 하나 이상. 표본 집계입니다.")

st.subheader("기상 데이터를 더하면 예측이 얼마나 좋아지는가")
model_tables(summary["models"])
prediction_section(store)
notes(summary)
