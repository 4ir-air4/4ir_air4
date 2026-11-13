"""통합 앱 메인 페이지: 실시간 항공기 지도 (ojh/live_app.py 의 첫 번째 탭).

live_app.py 의 탭 3개는 통합 앱 상단 메뉴의 맨 앞 3칸(실시간 지도 · 교통량 예측 · 공항 수요 예측)이 된다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ojh/

import streamlit as st  # noqa: E402
import live_app  # noqa: E402

st.set_page_config(page_title="AERO INSIGHT · 실시간 항공기 지도", page_icon="✈️", layout="wide")  # live_app 이 바꾼 탭 제목 되돌림
live_app.show_live_map()
