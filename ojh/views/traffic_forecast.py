"""통합 앱 메뉴: 항공 교통량 예측 및 모델 비교 (ojh/_2_traffic_tab.py)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ojh/

from _2_traffic_tab import show_traffic_prediction_tab  # noqa: E402

show_traffic_prediction_tab()
