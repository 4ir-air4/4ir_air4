"""통합 앱 메뉴: 전 세계 공항 이용객 수요 예측 (ojh/_3_airport_tab.py)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ojh/

from _3_airport_tab import show_airport_demand  # noqa: E402

show_airport_demand()
