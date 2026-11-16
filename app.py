"""AERO INSIGHT · 항공 교통 관제·수요 예측 통합 대시보드.   실행: streamlit run app.py

이 파일은 팀 작업을 한 화면으로 묶는 '껍데기(셸)'만 담당한다. 각자 작업 폴더는 섞지 않는다.
  - ojh/  : 실시간 항공기 지도, 항공 교통량 예측, 전 세계 공항 수요 예측  (메뉴용 래퍼는 ojh/views/)
  - ldh/  : 한국·미국·유럽·안전 분석 카테고리, 인천 지연 예보, 모델 비교
            (ldh/categories/ 에 폴더를 추가하면 META["group"] 에 맞는 메뉴 그룹에 자동으로 붙는다)

상단 메뉴(왼쪽부터 중요도 순)
  실시간 지도 · 교통량 예측 · 공항 수요 예측   ← 기존 메인 화면 탭 3개 (바로 열림)
  한국 ▾ · 미국 ▾ · 유럽·세계 ▾ · 항공 안전 ▾ · 종합·모델 ▾   ← 누르면 하위 메뉴가 펼쳐짐
왼쪽 사이드바에도 같은 그룹·순서의 접이식 메뉴가 있다 (두 메뉴 모두 같은 페이지로 이동).
"""
import sys
from html import escape
from pathlib import Path

import streamlit as st

from ldh.core.registry import discover

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ojh"))  # ojh 탭 파일들이 같은 폴더 모듈을 이름으로 import 한다

st.set_page_config(page_title="AERO INSIGHT · 항공 통합 대시보드", page_icon="✈️",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# 1. 메뉴 구성
# ---------------------------------------------------------------------------
TOP = ""  # 이 그룹의 페이지는 드롭다운 없이 메뉴 바에 바로 나온다
SECTIONS = [TOP, "한국", "미국", "유럽·세계", "항공 안전", "종합·모델"]
LDH_GROUP = {"한국": "한국", "미국": "미국", "유럽·세계": "유럽·세계", "안전": "항공 안전"}
# 왼쪽 사이드바에서 쓰는 그룹 이름·아이콘 (상단 메뉴와 같은 순서)
SIDE = {TOP: ("실시간 관제", ":material/satellite_alt:"), "한국": ("인천 · 한국", ":material/location_city:"),
        "미국": ("미국", ":material/flag:"), "유럽·세계": ("유럽 · 세계", ":material/public:"),
        "항공 안전": ("항공 안전", ":material/health_and_safety:"), "종합·모델": ("종합 · 모델 연구소", ":material/science:")}

# 같은 그룹 안의 순서(작을수록 위). 적지 않은 ldh 카테고리는 META["order"] + 100 순서로 뒤에 붙는다.
RANK = {
    "live": 0, "traffic_forecast": 1, "airport_demand": 2,
    "icn_risk": 0, "korea_ops": 1, "korea_demand": 2, "korea_airfare": 3,
    "us_flights": 0, "us_weather": 1,
    "world_traffic": 0, "europe": 1,
    "safety": 0,
    "home": 0, "model_compare": 1, "year_2026": 2,
}

# 페이지 상단 배너: (설명, 배지)
INFO = {
    "live":             ("OpenSky Network 실시간 관측으로 전 세계 항공기 위치·고도·기종과 비행 경로를 추적합니다.", ["LIVE", "OpenSky"]),
    "traffic_forecast": ("관측 데이터로 미래 항공 교통량을 예측하고 여러 모델의 성능을 비교합니다.", ["ML 예측", "모델 비교"]),
    "airport_demand":   ("미국·유럽·영국·한국 공항의 월별 승객 수와 GDP 성장률로 공항 수요를 예측합니다.", ["ML 예측", "글로벌"]),
    "icn_risk":         ("인천공항 도착·출발편의 지연 위험을 시간대·기상 조건으로 예보합니다.", ["ML 예측", "인천"]),
    "korea_ops":        ("국내 공항별 운항 실적과 지연 패턴을 분석합니다.", ["통계", "ML"]),
    "korea_demand":     ("국내 항공 여객·화물 수송 실적과 수요 추이를 예측합니다.", ["수요", "ML"]),
    "korea_airfare":    ("국내선 노선별 항공료 흐름과 변동 요인을 살펴봅니다.", ["요금", "통계"]),
    "us_flights":       ("미국 국내선 운항 실적(BTS)으로 지연·결항 패턴을 분석합니다.", ["BTS", "ML"]),
    "us_weather":       ("기상 조건이 미국 공항 운항 지연에 주는 영향을 분석합니다.", ["기상", "ML"]),
    "world_traffic":    ("전 세계 항공 교통 규모와 지역별 성장 흐름을 봅니다.", ["통계", "글로벌"]),
    "europe":           ("유럽 주요 공항의 이용객·운항 추이를 분석합니다.", ["EUROCONTROL", "ML"]),
    "safety":           ("항공 사고·사건 기록으로 안전 동향과 위험 요인을 살펴봅니다.", ["안전", "통계"]),
    "home":             ("모든 분석 카테고리의 핵심 지표와 모델 성능을 한 장에 모았습니다.", ["요약", "KPI"]),
    "model_compare":    ("카테고리별 예측 모델을 같은 기준(시간순 미래 구간 평가)으로 비교합니다.", ["벤치마크"]),
    "year_2026":        ("2026년 자료로만 학습한 구동 확인 결과입니다.", ["실험"]),
}


def build_pages():
    """[(그룹, slug, st.Page)] 를 돌려준다."""
    items = [
        # --- 기존 메인 화면 탭 3개 (ojh) ---
        (TOP, "live", st.Page("ojh/views/live_map.py", title="실시간 지도", icon="🌍", url_path="live", default=True)),
        (TOP, "traffic_forecast", st.Page("ojh/views/traffic_forecast.py", title="교통량 예측", icon="📊",
                                          url_path="traffic_forecast")),
        (TOP, "airport_demand", st.Page("ojh/views/airport_demand.py", title="공항 수요 예측", icon="🛫",
                                        url_path="airport_demand")),
        # --- 팀원 작업 (ldh) ---
        ("한국", "icn_risk", st.Page("ldh/core/icn_risk.py", title="인천 지연 위험 예보",
                                        icon=":material/flight_takeoff:", url_path="icn_risk")),
        ("종합·모델", "home", st.Page("ldh/core/home.py", title="통합 요약", icon=":material/dashboard:", url_path="home")),
        ("종합·모델", "model_compare", st.Page("ldh/core/model_compare.py", title="모델 비교",
                                             icon=":material/compare_arrows:", url_path="model_compare")),
        ("종합·모델", "year_2026", st.Page("ldh/core/year_2026.py", title="2026년 학습 결과",
                                         icon=":material/event:", url_path="year_2026")),
    ]
    for c in discover()[0]:
        items.append((LDH_GROUP.get(c["group"], c["group"]), c["slug"],
                      st.Page(c["page"], title=c["title"], icon=c["icon"], url_path=c["slug"])))
        RANK.setdefault(c["slug"], 100 + c.get("order", 99))
    return sorted(items, key=lambda it: RANK.get(it[1], 999))


# ---------------------------------------------------------------------------
# 2. 스타일
# ---------------------------------------------------------------------------
CSS = """
<style>
/* ---------- 왼쪽 사이드바 ---------- */
[data-testid="stSidebarNav"], [data-testid="stSidebarNavSeparator"] { display: none; }  /* 좁은 화면에서 자동으로 들어오는 기본 메뉴는 숨김 */
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: .4rem; }
.ai-brand {
  margin: 0 0 1rem; padding: 1.05rem 1rem .95rem; border-radius: 14px; color: #fff; position: relative; overflow: hidden;
  background: linear-gradient(135deg, #0b1e3f 0%, #133b7a 55%, #1d6fd8 100%); box-shadow: 0 6px 18px rgba(13,42,92,.28);
}
.ai-brand:after { content: ""; position: absolute; right: -38px; top: -38px; width: 120px; height: 120px; border-radius: 50%;
  border: 1px solid rgba(255,255,255,.18); box-shadow: 0 0 0 16px rgba(255,255,255,.05), 0 0 0 32px rgba(255,255,255,.03); }
.ai-brand .logo { font-size: 1.3rem; font-weight: 800; letter-spacing: .08em; line-height: 1.15; }
.ai-brand .logo span { color: #7cc4ff; }
.ai-brand .tag { font-size: .74rem; opacity: .82; margin-top: .3rem; line-height: 1.4; }
.ai-brand .live { display: inline-flex; align-items: center; gap: .35rem; margin-top: .65rem; font-size: .68rem; font-weight: 700;
  letter-spacing: .06em; padding: .18rem .5rem; border-radius: 999px; background: rgba(34,197,94,.18); color: #86efac;
  border: 1px solid rgba(134,239,172,.35); }
.ai-brand .dot { width: 7px; height: 7px; border-radius: 50%; background: #22c55e; animation: ai-pulse 1.6s infinite; }
@keyframes ai-pulse { 0% { box-shadow: 0 0 0 0 rgba(34,197,94,.7); } 70% { box-shadow: 0 0 0 7px rgba(34,197,94,0); }
                      100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); } }
.ai-menu-title { font-size: .68rem; font-weight: 700; letter-spacing: .14em; color: #6b7a99; margin: 0 0 0 .3rem;
  line-height: 1.4; padding-bottom: 1.15rem; }  /* Streamlit 마크다운의 아래 -16px 여백을 상쇄해 첫 카드와 겹치지 않게 */
section[data-testid="stSidebar"] { border-right: 1px solid #e3e9f3; box-shadow: 2px 0 16px rgba(16,42,90,.04); }
/* 그룹 사이 간격을 한 값으로 통일 */
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] { gap: .5rem; }
/* 그룹 카드: 머리와 본문이 한 장의 카드로 보이게 */
section[data-testid="stSidebar"] [data-testid="stExpander"] details {
  border: 1px solid #e3e9f3; border-radius: 12px; background: #fff; overflow: hidden;
  box-shadow: 0 1px 3px rgba(16,42,90,.04); transition: box-shadow .15s, border-color .15s; }
section[data-testid="stSidebar"] [data-testid="stExpander"] details:hover { border-color: #c9d7ee; box-shadow: 0 3px 10px rgba(16,42,90,.07); }
section[data-testid="stSidebar"] [data-testid="stExpander"] details[open] { border-color: #c9d7ee; }
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
  font-weight: 700; font-size: .9rem; padding: .62rem .8rem; background: transparent !important; color: #13213d; }
section[data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary {
  border-bottom: 1px solid #edf1f7; background: linear-gradient(90deg, rgba(37,99,235,.06), rgba(37,99,235,0)) !important; }
section[data-testid="stSidebar"] [data-testid="stExpanderDetails"] { padding: .4rem .4rem .45rem !important; }
section[data-testid="stSidebar"] [data-testid="stExpanderDetails"] [data-testid="stVerticalBlock"] { gap: .2rem; }
/* Streamlit 이 사이드바 링크에 주는 음수 여백(-6px) 때문에 메뉴가 겹쳐 보이던 것을 없앤다 */
section[data-testid="stSidebar"] [data-testid="stExpanderDetails"] [data-testid="stElementContainer"] { margin: 0 !important; }
/* 하위 메뉴: 그룹 아이콘과 같은 줄에서 시작, 높이 통일 */
section[data-testid="stSidebar"] [data-testid="stPageLink"] a {
  border-radius: 8px; padding: .38rem .5rem; min-height: 2.1rem; margin: 0; align-items: center; }
section[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover { background: #f1f5fc; }
section[data-testid="stSidebar"] [data-testid="stPageLink"] p { white-space: normal; line-height: 1.35; font-size: .88rem; }  /* 긴 이름 잘림 방지 */
section[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {
  background: linear-gradient(90deg, #e7f0ff, #f4f8ff); box-shadow: inset 3px 0 0 #2563eb; }
section[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] p { color: #1b4fb0; font-weight: 700; }
section[data-testid="stSidebar"] hr { margin: .6rem 0 .4rem; }
.ai-credit { font-size: .72rem; opacity: .62; line-height: 1.65; }
.ai-credit b { opacity: .95; }

/* ---------- 상단 메뉴 바 꾸미기 ---------- */
header[data-testid="stHeader"] { border-bottom: 1px solid rgba(127,127,127,.16); box-shadow: 0 2px 12px rgba(13,42,92,.06); }
[data-testid="stTopNav"] { gap: .2rem; }
[data-testid="stTopNavLink"], [data-testid="stTopNavSection"] { border-radius: 999px !important; padding: .28rem .65rem !important;
  font-weight: 600; transition: background .15s, color .15s; }
[data-testid="stTopNavLink"]:hover, [data-testid="stTopNavSection"]:hover { background: rgba(59,130,246,.10) !important; }
[data-testid="stTopNavLink"][aria-current="page"] { background: linear-gradient(120deg, #12366f, #1b5fc0) !important;
  box-shadow: 0 3px 10px rgba(27,95,192,.30); }
[data-testid="stTopNavLink"][aria-current="page"] * { color: #fff !important; }
/* 내 탭 3개와 팀원 그룹 사이 구분선 */
[data-testid="stTopNav"] > div:not(:has([data-testid="stTopNavSection"])) + div:has(> [data-testid="stTopNavSection"]) {
  margin-left: .45rem; padding-left: .65rem; border-left: 1px solid rgba(127,127,127,.28); }

/* ---------- 상단 메뉴 바 아래 여백: 메뉴에 가리지 않을 만큼만 남긴다 */
[data-testid="stMainBlockContainer"] { padding-top: 4.2rem; }
.ai-hero {
  border-radius: 16px; padding: 1.2rem 1.5rem 1.15rem; margin: .25rem 0 1.1rem; color: #fff;
  background: radial-gradient(circle at 92% 20%, rgba(124,196,255,.35) 0, rgba(124,196,255,0) 38%),
              linear-gradient(120deg, #0b1e3f 0%, #12366f 50%, #1b5fc0 100%);
  box-shadow: 0 8px 24px rgba(13, 42, 92, .22); position: relative; overflow: hidden;
}
.ai-hero:before { content: "✈"; position: absolute; right: 1.6rem; top: 50%; transform: translateY(-50%) rotate(-12deg);
  font-size: 4.4rem; opacity: .10; pointer-events: none; }
.ai-hero .crumb { font-size: .74rem; letter-spacing: .1em; font-weight: 700; color: #9dd2ff; }
.ai-hero .crumb b { color: #fff; letter-spacing: .14em; }
.ai-hero h1 { color: #fff; font-size: 1.75rem; font-weight: 800; margin: .2rem 0 .3rem; padding: 0; line-height: 1.25; }
.ai-hero p { margin: 0; opacity: .86; font-size: .95rem; max-width: 52rem; padding-right: 5rem; }
.ai-hero .badges { margin-top: .7rem; display: flex; gap: .4rem; flex-wrap: wrap; }
.ai-hero .badge { font-size: .7rem; font-weight: 700; letter-spacing: .04em; padding: .17rem .55rem; border-radius: 999px;
  background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); }
.ai-hero .badge.live { background: rgba(34,197,94,.22); border-color: rgba(134,239,172,.5); color: #bbf7d0; }
@media (max-width: 640px) { .ai-hero { padding: 1rem 1.1rem; } .ai-hero h1 { font-size: 1.35rem; }
  .ai-hero p { padding-right: 0; } .ai-hero:before { display: none; } }

/* ---------- 본문 전체 꾸미기 (모든 페이지 공통) ---------- */
[data-testid="stMain"] {
  background: radial-gradient(1100px 480px at 88% -8%, rgba(59,130,246,.13), transparent 62%),
              radial-gradient(900px 420px at -8% 38%, rgba(14,165,233,.07), transparent 60%),
              linear-gradient(180deg, #f5f8fd 0%, #eef3fa 100%);
}
header[data-testid="stHeader"] { background: rgba(245,248,253,.86) !important; backdrop-filter: blur(10px); }
/* 소제목: 왼쪽 강조 막대 */
[data-testid="stMain"] [data-testid="stHeading"] h2 { position: relative; padding-left: .9rem; color: #13213d; }
[data-testid="stMain"] [data-testid="stHeading"] h2:before { content: ""; position: absolute; left: 0; top: .28em; bottom: .28em;
  width: 5px; border-radius: 4px; background: linear-gradient(180deg, #1b5fc0, #38bdf8); }
[data-testid="stMain"] [data-testid="stHeading"] h3 { position: relative; padding-left: .75rem; color: #1a2b4f; }
[data-testid="stMain"] [data-testid="stHeading"] h3:before { content: ""; position: absolute; left: 0; top: .3em; bottom: .3em;
  width: 3px; border-radius: 3px; background: #3b82f6; }
/* 지표 카드 */
[data-testid="stMain"] [data-testid="stMetric"] {
  background: #fff; border: 1px solid #e1e8f3 !important; border-radius: 14px; padding: .95rem 1.1rem .85rem;
  box-shadow: 0 2px 10px rgba(16,42,90,.06); position: relative; overflow: hidden; transition: transform .15s, box-shadow .15s; }
[data-testid="stMain"] [data-testid="stMetric"]:before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 3px;
  background: linear-gradient(90deg, #1b5fc0, #38bdf8); }
[data-testid="stMain"] [data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(16,42,90,.10); }
[data-testid="stMain"] [data-testid="stMetricLabel"] p { color: #5b6b8c; font-weight: 600; }
[data-testid="stMain"] [data-testid="stMetricValue"] { color: #10295c; font-weight: 700; }
[data-testid="stMain"] [data-testid="stMetricValue"], [data-testid="stMain"] [data-testid="stMetricValue"] * {  /* 긴 값(기간 등)은 '...' 대신 줄바꿈 */
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important; overflow-wrap: anywhere; line-height: 1.2; }
/* 차트 · 표 · 지도 카드 */
[data-testid="stMain"] [data-testid="stVegaLiteChart"], [data-testid="stMain"] [data-testid="stPlotlyChart"],
[data-testid="stMain"] [data-testid="stPyplot"] {
  background: #fff; border: 1px solid #e1e8f3; border-radius: 14px; padding: .7rem 0 .35rem; overflow: hidden;
  box-shadow: 0 2px 10px rgba(16,42,90,.05); }  /* 좌우 여백을 주면 차트가 카드보다 넓어져 오른쪽이 잘린다 */
[data-testid="stMain"] [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid #e1e8f3;
  box-shadow: 0 2px 10px rgba(16,42,90,.05); background: #fff; }
[data-testid="stMain"] iframe { border-radius: 14px; box-shadow: 0 6px 22px rgba(16,42,90,.12); }
/* 탭: 알약 모양 세그먼트 */
[data-testid="stMain"] [data-testid="stTabs"] [role="tablist"] { gap: .25rem; background: #fff; padding: .3rem; border-radius: 12px;
  border: 1px solid #e1e8f3; box-shadow: 0 1px 4px rgba(16,42,90,.04); width: fit-content; max-width: 100%; overflow-x: auto; }
[data-testid="stMain"] [data-testid="stTab"] { border-radius: 9px; padding: .42rem .95rem !important; height: auto;
  border: none !important; transition: background .15s; }
[data-testid="stMain"] [data-testid="stTab"]:hover { background: #f1f5fc; }
[data-testid="stMain"] [data-testid="stTab"][aria-selected="true"] { background: linear-gradient(120deg, #12366f, #1b5fc0);
  box-shadow: 0 3px 10px rgba(27,95,192,.25); }
[data-testid="stMain"] [data-testid="stTab"][aria-selected="true"] p { color: #fff !important; font-weight: 700; }
[data-testid="stMain"] [data-testid="stTabs"] [role="tablist"] > div:not([data-testid="stTab"]) { display: none; }  /* 기본 밑줄 */
/* 테두리 상자(st.container(border=True)): 흰 카드 */
[data-testid="stMain"] [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {
  background: #fff; border-color: #e1e8f3 !important; border-radius: 14px; box-shadow: 0 2px 10px rgba(16,42,90,.05);
  transition: transform .15s, box-shadow .15s; }
[data-testid="stMain"] [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]:hover {
  transform: translateY(-2px); box-shadow: 0 8px 20px rgba(16,42,90,.09); }
/* 접이식 · 알림 · 버튼 · 입력 */
[data-testid="stMain"] [data-testid="stExpander"] details { background: #fff; border-radius: 12px; border-color: #e1e8f3; }
[data-testid="stMain"] [data-testid="stAlert"] > div { border-radius: 12px; }
[data-testid="stMain"] .stButton button, [data-testid="stMain"] .stDownloadButton button {
  border-radius: 10px; font-weight: 600; border-color: #d3deef; box-shadow: 0 1px 3px rgba(16,42,90,.06); }
[data-testid="stMain"] .stButton button:hover { border-color: #2563eb; color: #1b4fb0; }
[data-testid="stMain"] [data-baseweb="select"] > div { background: #fff; border-radius: 10px; }
[data-testid="stMain"] hr { border-color: #dfe6f2; }
/* 페이지 맨 아래 */
.ai-footer { margin-top: 2.5rem; padding: 1rem 0 .5rem; border-top: 1px solid #dfe6f2; display: flex; justify-content: space-between;
  flex-wrap: wrap; gap: .5rem; font-size: .75rem; color: #7484a3; }
.ai-footer b { color: #12366f; letter-spacing: .08em; }
</style>
"""


def sidebar(items, order, current):
    """왼쪽 메뉴: 상단 메뉴와 같은 그룹·순서를 접이식으로 보여 준다. 지금 보고 있는 그룹은 펼쳐 둔다."""
    with st.sidebar:
        st.markdown(
            '<div class="ai-brand"><div class="logo">AERO <span>INSIGHT</span></div>'
            '<div class="tag">항공 교통 관제 · 수요 예측 통합 플랫폼</div>'
            '<div class="live"><span class="dot"></span>LIVE TRACKING</div></div>'
            '<div class="ai-menu-title">NAVIGATION</div>',
            unsafe_allow_html=True)
        for i, g in enumerate([g for g in order if any(it[0] == g for it in items)], 1):
            label, icon = SIDE.get(g, (g, ":material/folder:"))
            pages = [p for gg, _, p in items if gg == g]
            here = any(p.title == current.title for p in pages)
            with st.expander(f"{i:02d}  ·  {label}", icon=icon, expanded=(g == TOP) or here):
                for p in pages:
                    st.page_link(p)
        st.divider()
        st.markdown(
            '<div class="ai-credit"><b>TEAM 4IR-AIR4</b><br>'
            '실시간 관제 · 교통량 · 공항 수요 예측 — <b>ojh</b><br>'
            '한국 · 미국 · 유럽 · 안전 분석 — <b>ldh</b><br>'
            '데이터: OpenSky · BTS · EUROCONTROL · Eurostat 외</div>',
            unsafe_allow_html=True)


def mark_top_section(order, groups, section):
    """팀원 페이지를 보고 있으면 상단 메뉴에서 그 그룹 버튼(예: 미국 ▾)도 강조한다."""
    if section == TOP:
        return
    shown = [g for g in order if groups.get(g) and g != TOP]
    nth = len(groups.get(TOP, [])) + shown.index(section) + 1  # stTopNav 의 몇 번째 칸인지
    st.markdown(
        f'<style>[data-testid="stTopNav"] > div:nth-child({nth}) [data-testid="stTopNavSection"] {{'
        'background: rgba(59,130,246,.13) !important; color: #1b5fc0 !important; box-shadow: inset 0 0 0 1px rgba(27,95,192,.35); }'
        '</style>', unsafe_allow_html=True)


def hero(section, slug, page):
    desc, badges = INFO.get(slug, ("", []))
    crumb = "<b>AERO INSIGHT</b>" + (f" · {escape(section)}" if section else " · 실시간 관제")
    badge_html = "".join(
        f'<span class="badge{" live" if b == "LIVE" else ""}">{"● " if b == "LIVE" else ""}{escape(b)}</span>' for b in badges)
    st.markdown(
        f'<div class="ai-hero"><div class="crumb">{crumb}</div><h1>{escape(page.title)}</h1>'
        + (f"<p>{escape(desc)}</p>" if desc else "")
        + (f'<div class="badges">{badge_html}</div>' if badge_html else "") + "</div>",
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 3. 실행
# ---------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)
items = build_pages()
order = SECTIONS + [g for g in dict.fromkeys(g for g, _, _ in items) if g not in SECTIONS]  # 새 ldh 그룹은 뒤에
groups = {s: [p for g, _, p in items if g == s] for s in order}
current = st.navigation({k: v for k, v in groups.items() if v}, position="top")
sidebar(items, order, current)
sec, slug = next((g, k) for g, k, p in items if p.title == current.title)
mark_top_section(order, groups, sec)
hero(sec, slug, current)
current.run()
st.markdown('<div class="ai-footer"><span><b>AERO INSIGHT</b> · 항공 교통 관제 · 수요 예측 통합 플랫폼</span>'
            '<span>TEAM 4IR-AIR4 · ojh · ldh</span></div>', unsafe_allow_html=True)
