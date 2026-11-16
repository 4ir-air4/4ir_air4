"""한국 운항·지연: 인천 편별 지연과 한국공항공사 집계."""
import streamlit as st

from ldh.core.ui import header, load, model_tables, notes, prediction_section, store_for, time_chart

store = store_for(__file__)
summary = header(store)
PERCENT = st.column_config.NumberColumn(format="percent")

st.subheader("인천공항 월별 추이")
monthly = load(store, "icn_monthly")
left, right = st.columns(2)
left.markdown("**지연률**")
time_chart(left, monthly, x="ym", y="지연률", color="direction", x_label="월", y_label="지연률", alt="인천 월별 지연률")
right.markdown("**운항 편수**")
time_chart(right, monthly, x="ym", y="편수", color="direction", x_label="월", y_label="편", alt="인천 월별 운항 편수")
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
    time_chart(st, shown, x="ym", y="지연률", color="항공사", x_label="월", y_label="지연률", alt=f"항공사별 월 {way} 지연률")
    st.caption("항공사 이름은 인천 실시간 수집 자료의 한글 이름이고, 없으면 OpenFlights 영문 이름입니다. 출발 지연은 중앙값을 뺀 뒤 15분 이상입니다. 편수가 적은 달은 지연률이 크게 흔들립니다.")

st.subheader("한국공항공사 전체 공항 (월 집계)")
trend = load(store, "kac_trend")
left, right = st.columns(2)
left.markdown("**월별 지연 편수**")
time_chart(left, trend, x="ym", y="지연편", color="국내_국제", x_label="월", y_label="편", alt="월별 지연 편수")
right.markdown("**여객 천 명당 지연 편수**")
time_chart(right, trend, x="ym", y="여객 천 명당 지연 편수", color="국내_국제", x_label="월", y_label="편", alt="여객 천 명당 지연 편수")
st.markdown("**시간대별 누적 지연 편수**")
st.bar_chart(load(store, "kac_hour"), x="시간대", y="지연편", color="국내_국제", stack=False, y_label="편", alt="시간대별 지연 편수")

st.subheader("공항별 지연·결항 원인 (집계 기간 불명)")
# 원인 순서와 색은 두 그래프에서 같게 고정한다(같은 원인 = 같은 색). dataviz 기본 범주 팔레트 1~4번, 이웃 색 구분 검증 통과
CAUSES = {"기상": "#2a78d6", "연결편": "#eb6834", "정비": "#1baf7a", "기타": "#eda100"}
CAUSE_WORD = {"기상": "기상", "연결": "연결편", "정비": "정비", "기타": "기타"}  # 원본 열 이름의 단어 -> 원인


def by_airport(table, total_word):
    """원본 열 이름(공개 시점마다 조금 다름)을 원인 4개로 묶고, 국내·국제를 공항별로 더한다."""
    d = load(store, table)
    causes = {c: v for c in d.columns for k, v in CAUSE_WORD.items() if k in c}
    total = next(c for c in d.columns if c.startswith(total_word))
    out = d.rename(columns={**causes, total: total_word}).groupby("공항", as_index=False)[[total_word, *CAUSES]].sum()
    return out[out[total_word] > 0].sort_values(total_word, ascending=False, ignore_index=True)  # 편수 0인 공항(무안 등)은 빈 막대라 뺀다


delay, cancel = by_airport("airport_delay", "지연편수"), by_airport("airport_cancel", "결항편수")
share = lambda t, k: t[k].sum() / t[list(CAUSES)].sum().sum()  # noqa: E731
with st.container(horizontal=True):
    st.metric("지연 중 연결편 원인", f"{share(delay, '연결편'):.0%}", border=True, help="앞 편이 늦게 도착해 다음 편이 늦게 떠나는 경우입니다.")
    st.metric("결항 중 기상 원인", f"{share(cancel, '기상'):.0%}", border=True)
    st.metric("지연 중 기상 원인", f"{share(delay, '기상'):.0%}", border=True)
left, right = st.columns(2)
for col, t, what in ((left, delay, "지연"), (right, cancel, "결항")):
    col.markdown(f"**{what} 원인 구성비** (위쪽일수록 {what} 편수가 많은 공항)")
    col.bar_chart(t, x="공항", y=list(CAUSES), color=list(CAUSES.values()), stack="normalize", horizontal=True, sort=False,
                  x_label="공항", y_label="구성비", height=420, alt=f"한국공항공사 공항별 {what} 원인 구성비(기상, 연결편, 정비, 기타)")
st.caption("지연은 대부분 연결편(앞 편 지연이 이어짐) 때문이고, 결항은 대부분 기상 때문입니다. 인천 지연 모델이 같은 편명의 이전 지연 이력을 쓰고, "
           "결항 모델에서 기상 입력이 성능을 크게 올리는(미국 결항 PR-AUC 0.051 → 0.218) 근거입니다. 자료: 한국공항공사 공항별 지연·결항 현황(국내선·국제선 합계, 공개 자료에 집계 기간이 적혀 있지 않음).")
with st.expander("공항별 편수 표로 보기"):
    left, right = st.columns(2)
    left.dataframe(delay, hide_index=True, alt="공항별 지연 편수와 원인별 건수")
    right.dataframe(cancel, hide_index=True, alt="공항별 결항 편수와 원인별 건수")

st.subheader("모델 성능")
model_tables(summary["models"])
prediction_section(store)
notes(summary)
