"""통합 앱 메뉴: 활주로·계류장 혼잡도 (ojh/_5_apron_tab.py)"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ojh/

import _5_apron_tab as apron  # noqa: E402

# 파일이 바뀌었으면 앱을 재시작하지 않아도 새 버전을 읽는다 (서버가 예전 모듈을 메모리에 들고 있는 경우 대비)
_mtime = Path(apron.__file__).stat().st_mtime
if getattr(apron, "_loaded_mtime", None) != _mtime:
    apron = importlib.reload(apron)
    apron._loaded_mtime = _mtime

apron.show_apron_congestion()
