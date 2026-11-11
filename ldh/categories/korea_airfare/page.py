"""한국 국내선 항공료: 계절성, 비용·수요 지표, 월별 예측."""
import altair as alt
import streamlit as st

from ldh.core.ui import COVID_NOTE, header, in_covid, load, model_tables, notes, store_for, time_chart

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

if store.table_path("price_year_month").exists():  # 파이프라인을 다시 돌리기 전에는 없다
    st.markdown("**연도별로 보기**")
    measure = st.segmented_control("기준", ["그해 평균 대비(%)", "전년 같은 달 대비(%)"], default="그해 평균 대비(%)") or "그해 평균 대비(%)"
    cells = load(store, "price_year_month").dropna(subset=[measure])
    grid = alt.Chart(cells).mark_rect().encode(
        x=alt.X("월:O", title=None), y=alt.Y("연도:O", title=None),
        color=alt.Color(f"{measure}:Q", scale=alt.Scale(scheme="blueorange", domainMid=0), title="%"),
        tooltip=["연도", "월", alt.Tooltip("지수:Q", format=".1f"), alt.Tooltip(f"{measure}:Q", format="+.1f")])
    covid = cells[in_covid(cells["연도"] + "-" + cells["월"].str[:2]).values]  # 월은 '01월' 형식
    outline = alt.Chart(covid).mark_rect(filled=False, stroke="black", strokeWidth=1.5).encode(x="월:O", y="연도:O")
    st.altair_chart(grid + outline, width="stretch")
    st.caption(f"그해 평균 대비는 같은 해 안의 높낮이, 전년 같은 달 대비는 1년 사이 변화입니다. 2026년은 3월부터 제트유가 급등해 그해 평균 대비만 보면 상반기가 낮게 보입니다. 검은 테두리는 {COVID_NOTE}입니다.")

    st.markdown("**명절 효과** (추세 대비 %)")
    st.dataframe(load(store, "holiday_effect").round(1), hide_index=True, alt="설과 추석이 든 해와 아닌 해의 같은 달 항공료 비교")
    st.caption("설·추석은 해마다 달이 바뀝니다. 같은 달을 명절이 든 해와 아닌 해로 나눠 비교했습니다. 코로나 구간을 빼면 해가 3~4개뿐이라 참고용입니다.")

st.subheader("추이")
st.markdown("**항공료 지수**")
time_chart(st, monthly, x="월", y=["항공료 지수", "실질 항공료", "국내항공여객 생산자물가"], y_label="2020=100", alt="항공료 지수, 실질 항공료, 생산자물가 추이")
left, right = st.columns(2)
left.markdown("**제트유·원유**")
time_chart(left, monthly, x="월", y=["제트유", "원유(달러)"], y_label="2020=100", alt="제트유와 원유 지수 추이")
right.markdown("**원/달러 환율**")
time_chart(right, monthly, x="월", y="원/달러 환율", y_label="원", alt="원달러 환율 추이")
st.markdown("**국내선 여객**")
time_chart(st, monthly, x="월", y="여객(명)", y_label="명", alt="국내선 월별 여객 추이")

if store.table_path("fare_compare").exists():
    st.subheader("다른 교통 요금과 비교")
    st.caption("모두 소비자물가 품목 지수(2020=100)라 2020년 이후 오른 정도를 바로 비교할 수 있습니다. 금액이 아니라 지수입니다.")
    time_chart(st, load(store, "fare_monthly"), x="월", y="지수", color="항목", y_label="2020=100", alt="교통 요금별 소비자물가 지수 추이")
    st.dataframe(load(store, "fare_compare").round(1), hide_index=True, alt="교통 요금별 최근 지수, 2020년 대비, 전년 대비, 계절 변동 폭")
    st.caption("계절 변동 폭은 코로나 구간을 뺀 월별 계절 지수(연평균 100)의 최고와 최저 차이입니다.")

st.subheader("노선별 수요 (운임 지수 보완)")
st.caption("운임은 전국 지수 하나뿐이라 노선별 가격은 알 수 없습니다. 대신 노선별 하루 평균 여객의 계절 지수(연평균 100)로 어느 노선, 어느 달에 수요가 몰리는지 봅니다.")
routes = load(store, "route_summary")
heat = alt.Chart(load(store, "route_month")).mark_rect().encode(
    x=alt.X("월:O", title=None), y=alt.Y("노선:N", sort=list(routes["노선"]), title=None),
    color=alt.Color("여객 계절 지수:Q", scale=alt.Scale(scheme="blueorange", domainMid=100)),
    tooltip=["노선", "월", alt.Tooltip("여객 계절 지수:Q", format=".1f")])
st.altair_chart(heat, width="stretch")
st.dataframe(routes.round(2), hide_index=True, alt="노선별 최근 12개월 여객, 편당 여객, 성수기와 비수기")
st.caption("노선은 최근 12개월 여객이 많은 순입니다. 편당 여객은 최근 12개월 여객 ÷ 운항 편수이고, 항공료 계절 지수와 상관은 월별 12개 값으로 계산했습니다.")

st.subheader("항공사별 국내선 여객")
st.caption("운임은 항공사별로도 공개되지 않아 여객과 운항만 봅니다. 대형은 대한항공·아시아나항공이고 나머지는 저비용 항공사로 묶었습니다. 좌석 수가 없어 탑승률 대신 편당 여객을 씁니다.")
left, right = st.columns(2)
left.markdown("**최근 12개월 항공사별**")
left.dataframe(load(store, "airline_recent").drop(columns="코드").round(1), hide_index=True, alt="항공사별 최근 12개월 국내선 여객, 운항, 편당 여객")
right.markdown("**저비용 항공사 여객 비중**")
time_chart(right, load(store, "airline_lcc_monthly"), x="월", y="저비용 비중(%)", y_label="%", alt="월별 국내선 저비용 항공사 여객 비중")
share = load(store, "route_airline")
top = list(routes["노선"].head(10))
bars = alt.Chart(share[share["노선"].isin(top)]).mark_bar().encode(
    x=alt.X("점유율(%):Q", stack="normalize", axis=alt.Axis(format="%"), title="최근 12개월 여객 점유율"), y=alt.Y("노선:N", sort=top, title=None),
    color=alt.Color("항공사:N"), tooltip=["노선", "항공사", alt.Tooltip("여객:Q", format=","), alt.Tooltip("점유율(%):Q", format=".1f")])
st.markdown("**노선별 항공사 점유율** (여객 상위 10개 노선)")
st.altair_chart(bars, width="stretch")

if store.table_path("route_airline_monthly").exists():  # 파이프라인을 다시 돌리기 전에는 없다
    pick = st.selectbox("노선 자세히 보기", list(routes["노선"]))
    one = load(store, "route_airline_monthly").query("노선 == @pick and 여객 > 0")
    one = one.assign(월=one["월"].str[:4] + "-" + one["월"].str[4:])
    left, right = st.columns([2, 1])
    left.markdown(f"**{pick} 항공사별 점유율 추이**")
    time_chart(left, one, x="월", y="여객", color="항공사", kind="area", stack="normalize", y_label="여객 점유율", alt=f"{pick} 항공사별 여객 점유율 추이")
    last = one[one["월"].isin(sorted(one["월"].unique())[-12:])].groupby("항공사", as_index=False)[["여객", "운항"]].sum()
    last["편당 여객"] = last["여객"] / last["운항"]
    right.markdown("**최근 12개월**")
    right.dataframe(last.sort_values("여객", ascending=False).round(1), hide_index=True, alt=f"{pick} 항공사별 최근 12개월 여객, 운항, 편당 여객")

st.subheader("지표가 항공료보다 얼마나 앞서는가")
st.caption("전년 대비 변화율끼리의 상관계수입니다. 시차 2는 지표의 2개월 전 값과 비교한 것입니다.")
st.line_chart(load(store, "lag_corr"), x="시차(개월)", y="상관계수", color="지표", alt="지표별 시차 상관계수")

st.subheader("예측 결과")
predictions = load(store, "predictions")
horizon = st.segmented_control("예측 시점", ["1개월 전", "3개월 전"], default="1개월 전") or "1개월 전"
shown = ["전체", "직전 값 그대로", "작년 같은 달 값"]  # 8개 모델을 모두 그리면 선을 구분할 수 없다. 전체 비교는 아래 표에 있다
chart = predictions[["월", "실제"]].join(predictions[[f"{horizon}: {name}" for name in shown]].set_axis(shown, axis=1))
time_chart(st, chart, x="월", y=["실제", *shown], y_label="지수", alt=f"{horizon} 예측과 실제 항공료 지수")
compare = load(store, "model_compare")
st.dataframe(compare[compare["예측 시점"] == horizon].round(2), hide_index=True, alt="모델별 오차 비교 표")
with st.expander("1개월 뒤 모델의 표준화 계수"):
    st.dataframe(load(store, "coefficients").round(4), hide_index=True, alt="1개월 뒤 모델의 표준화 계수")

st.subheader("모델 성능")
model_tables(summary["models"])
notes(summary)
