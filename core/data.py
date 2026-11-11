"""여러 카테고리가 같이 쓰는 데이터 읽기 도구."""
import io
from pathlib import Path

import pandas as pd

from core.registry import ROOT

REF = ROOT / "reference_data"
CACHE = ROOT / "data" / "cache"
START_YEAR = 2020  # 통계·학습·예측에 쓰는 첫 해. 큰 데이터 캐시는 폴더 이름에 이 값이 들어가 값을 바꾸면 새로 만든다
COVID = (pd.Period("2020-02", "M"), pd.Period("2022-04", "M"))  # 코로나 시기. 국내선 항공료가 계산에서 빼고, 모든 화면이 차트에 표시한다


def read_csv_bytes(raw, **kw):
    """국내 CSV는 CP949와 UTF-8이 섞여 있다. 실패한 시도의 읽기 위치가 남지 않게 매번 새 버퍼로 읽는다."""
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding, **kw)
        except UnicodeDecodeError:
            continue
    raise ValueError("UTF-8, CP949 모두 해석할 수 없는 파일")


def read_csv_any(path, **kw):
    return read_csv_bytes(Path(path).read_bytes(), **kw)


def to_number(s):
    """'"1,234"' 같은 문자열 숫자를 숫자로 바꾼다."""
    return pd.to_numeric(s.astype("string").str.replace(r'[",\s]', "", regex=True), errors="coerce")
