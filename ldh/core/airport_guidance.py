"""국내 공항 혼잡 예상 화면. 팀원 sjw 가 내보낸 정적 결과(sjw/outputs/airport_guidance/<생성번호>/)만 읽는다.

원본 CSV·수집기·AI 는 쓰지 않는다(sjw README 의 outputs-only 요청). 읽기 전에 manifest 해시를 확인하고, 시각은 KST 로 바꿔 보여 준다.
새 결과를 받으면 같은 위치에 새 생성번호 폴더를 통째로 넣고 RUN 만 바꾼다.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

RUN = "20261117_122311"
BASE = Path(__file__).resolve().parents[2] / "sjw" / "outputs" / "airport_guidance" / RUN
STATUS = {"N": "보통 (N)", "Y": "혼잡 (Y)"}
COLORS = ["#2a78d6", "#eb6834"]  # dataviz 기본 범주 1·2번(보통, 혼잡). 범례와 표가 함께 있어 색만으로 구분하지 않는다
AOD = {"출발": "D", "도착": "A"}
TOF = {"D": "국내선", "I": "국제선"}


@st.cache_data
def load(base):
    base = Path(base)
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    bad = [f["path"] for f in manifest["files"] if hashlib.sha256((base / f["path"]).read_bytes()).hexdigest() != f["sha256"]]
    if bad:
        raise ValueError(f"해시가 맞지 않는 파일: {bad}")
    read = lambda name: json.loads((base / name).read_text(encoding="utf-8"))  # noqa: E731
    passengers = pd.DataFrame(read("passengers.json")["records"]).astype({"SDT": str, "HH": str})
    return manifest, passengers, read("weather.json")["airports"], read("flights.json")["airports"]


def kst(value):
    return pd.to_datetime(value, utc=True).tz_convert("Asia/Seoul").strftime("%m-%d %H:%M")


try:
    manifest, passengers, weather, flights = load(str(BASE))
except (OSError, ValueError, KeyError) as exc:
    st.error(f"공항 안내 결과 파일을 읽지 못했습니다. 옛 결과를 최신으로 보여 주지 않습니다. ({exc})", icon=":material/error:")
    st.stop()

st.caption(f"자료: 한국공항공사 공항별 예상 여객·운항 현황, 기상청 공항 기상(AMOS). 수집·정리: sjw. 결과 생성 {kst(manifest['metadata']['exported_at'])} KST.")
names = passengers.drop_duplicates("ARP").set_index("ARP").airport_name
order = passengers.ARP.value_counts().index.tolist()
left, mid, right, last = st.columns([2, 1, 1, 1.4], vertical_alignment="bottom")
code = left.selectbox("공항", order, index=order.index("GMP") if "GMP" in order else 0,  # 김포는 예상 여객·기상·운항이 모두 있다
                      format_func=lambda c: f"{names[c]} ({c})")
way = mid.segmented_control("방향", list(AOD), default="출발") or "출발"
here = passengers[(passengers.ARP == code) & (passengers.AOD == AOD[way])]
kinds = sorted(here.TOF.unique())
kind = right.segmented_control("노선", kinds, default=kinds[0], format_func=TOF.get) or kinds[0]
here = here[here.TOF == kind]
days = sorted(here.SDT.unique())
day = last.selectbox("날짜", days, index=len(days) - 1, format_func=lambda s: f"{s[:4]}-{s[4:6]}-{s[6:]}")

st.subheader(f"{names[code]} {TOF[kind]} {way} 시간대별 예상 인원")
today = here[here.SDT == day].sort_values("HH")
peak = today.loc[today.PCT.idxmax()]
with st.container(horizontal=True):
    st.metric("하루 예상 인원", f"{today.PCT.sum():,.0f}명", border=True, help="표시된 시간대(HH)의 예상 인원(PCT)을 더한 값입니다.")
    st.metric("가장 붐비는 시간대", f"{peak.HH}시대 · {peak.PCT:,.0f}명", border=True)
    st.metric("혼잡(Y) 시간대", f"{(today.CONGEST_YN == 'Y').sum()}개 / {len(today)}개", border=True)
bars = today.assign(**{STATUS[k]: today.PCT.where(today.CONGEST_YN == k, 0) for k in STATUS})  # 빈 값이면 축이 음수로 늘어나 0으로 채운다
st.bar_chart(bars, x="HH", y=list(STATUS.values()), color=COLORS, x_label="시간대(HH, 원본 구분)", y_label="예상 인원(명)", height=320,
             alt=f"{names[code]} {TOF[kind]} {way} {day} 시간대별 예상 인원과 제공기관 혼잡 구분")
st.caption(f"{kst(today.collected_at.iloc[0])} KST에 받은 예상치입니다. 실제 탑승 인원이나 대기시간이 아닙니다.")

st.markdown("**날짜별 비교** (같은 시간대의 예상 인원, 굵은 칸은 혼잡 Y)")
table = here.pivot_table(index="HH", columns="SDT", values="PCT").rename(columns=lambda s: f"{s[4:6]}-{s[6:]}")
flag = here.pivot_table(index="HH", columns="SDT", values="CONGEST_YN", aggfunc="first").rename(columns=lambda s: f"{s[4:6]}-{s[6:]}")
st.dataframe(table.style.format("{:,.0f}", na_rep="").apply(lambda _: flag.eq("Y").replace({True: "font-weight: 700", False: ""}), axis=None),
             alt=f"{names[code]} {TOF[kind]} {way} 날짜별 시간대 예상 인원")

rows = pd.DataFrame()
left, right = st.columns(2)
with left:
    st.subheader("최신 공항 기상")
    w = weather.get(code, {})
    if w.get("available") and w.get("records"):
        r = w["records"][0]
        show = lambda v, unit: "자료 없음" if v is None else f"{v:,.1f}{unit}"  # noqa: E731
        with st.container(horizontal=True):
            st.metric("기온", show(r.get("temperature"), "℃"), border=True)
            st.metric("풍속", show(r.get("wind_speed_10m"), "m/s"), border=True)
            st.metric("시정(좌)", show(r.get("l_vis"), "m"), border=True)
        st.caption(f"관측 {kst(r['observed_at'])} KST 1회 값입니다. 위 예상 인원 날짜와 무관한 최신 조회 결과입니다.")
    else:
        st.info("이 공항에 맞는 기상 관측소 자료가 없습니다." + (f" ({', '.join(w.get('reasons', []))})" if w.get("reasons") else ""), icon=":material/cloud_off:")
with right:
    st.subheader(f"운항 현황 ({way})")
    f = flights.get(code, {})
    if f.get("available") and f.get("records"):
        rows = pd.DataFrame(f["records"])
        rows = rows[rows.io == ("O" if way == "출발" else "I")]
        st.dataframe(rows.rmkKor.value_counts().rename_axis("상태").reset_index(name="편수"), hide_index=True, alt="운항 상태별 편수")
        st.caption(f"{kst(f['collected_at'])} KST에 한 번 받은 운항표입니다. 운항일과 실제 지연은 검증되지 않아 계산하지 않습니다.")
    else:
        st.info("운항 자료가 없습니다.", icon=":material/flight:")
if len(rows):
    with st.expander(f"운항표 보기 ({len(rows)}편)"):
        st.dataframe(rows[["airFln", "std", "etd", "rmkKor", "city"]].rename(columns={"airFln": "편명", "std": "예정", "etd": "변경", "rmkKor": "상태", "city": "도시"}),
                     hide_index=True, alt="운항표")

st.caption("읽는 법: 예상 인원(PCT)은 공항공사가 미리 낸 예상치이고, 혼잡 Y/N은 공항공사의 구분값입니다(AI 예측 아님). 7일 자료에서 혼잡 Y가 있는 공항·방향 10개 중 8개는 "
           "예상 인원 기준값 하나로 Y/N이 나뉩니다. 시간대(HH)의 정확한 시각 구간은 확인되지 않았습니다. 기상·운항은 결과 생성 때 한 번 조회한 값이라 날이 지나면 옛 자료입니다.")
