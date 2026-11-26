"""통합 앱 메뉴: 항공 교통량 예측 및 모델 비교 (ojh/_2_traffic_tab.py)"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ojh/

import _2_traffic_tab as traffic  # noqa: E402

# 파일이 바뀌었으면 앱을 재시작하지 않아도 새 버전을 읽는다 (서버가 예전 모듈을 메모리에 들고 있는 경우 대비)
_mtime = Path(traffic.__file__).stat().st_mtime
if getattr(traffic, "_loaded_mtime", None) != _mtime:
    traffic = importlib.reload(traffic)
    traffic._loaded_mtime = _mtime

traffic.show_traffic_prediction_tab()
