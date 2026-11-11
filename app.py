"""항공 데이터 통계·예측 대시보드.  실행: streamlit run app.py

메뉴는 categories/ 폴더에서 자동으로 만든다. 새 카테고리 추가 방법은 categories/README.md 참고.
"""
import streamlit as st

from core.registry import discover

st.set_page_config(page_title="항공 데이터 대시보드", page_icon="✈️", layout="wide")

categories, _ = discover()
live = st.Page("ojh/live_app.py", title="실시간 항공기", icon=":material/radar:", url_path="live", default=True)
groups = {"": [live,
               st.Page("core/home.py", title="전체 요약", icon=":material/dashboard:", url_path="home"),
               st.Page("core/icn_risk.py", title="인천 지연 위험 예보", icon=":material/flight_takeoff:", url_path="icn_risk"),
               st.Page("core/model_compare.py", title="모델 비교", icon=":material/compare_arrows:", url_path="model_compare"),
               st.Page("core/year_2026.py", title="2026년 학습 결과", icon=":material/event:", url_path="year_2026")]}
for c in categories:
    groups.setdefault(c["group"], []).append(
        st.Page(c["page"], title=c["title"], icon=c["icon"], url_path=c["slug"]))

page = st.navigation(groups)
if page.title != live.title:  # ojh/live_app.py 는 자기 제목을 직접 그린다. 기본 페이지는 url_path 가 비어 title 로 비교
    st.title(page.title)
page.run()
