"""bts_data/ontime_YYYY_MM.zip -> data/ontime_YYYY_MM.parquet (학습에 필요한 컬럼만 추출).

사용법: python prepare_data.py [시작연도]   예) python prepare_data.py 2022
이미 변환된 파일은 건너뛰므로 중단 후 다시 실행해도 됩니다.
"""
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

SRC, DST = Path("bts_data"), Path("data")
# 오래된 연도는 일부 컬럼(지연 원인, TaxiOut 등)이 비어 있어 타입을 고정해야 연도 간 concat 이 안전함
DTYPES = {
    "Year": "int64", "Month": "int64", "DayOfWeek": "int64", "FlightDate": "string",
    "Reporting_Airline": "string", "Origin": "string", "Dest": "string", "CancellationCode": "string",
    "Tail_Number": "string",
    **{c: "float64" for c in [
        "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "Distance",
        "ArrDel15", "ArrDelay", "DepDel15", "DepDelay", "Cancelled", "Diverted",
        "TaxiOut", "TaxiIn", "AirTime",
        "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
    ]},
}


def up_to_date(path):
    """DTYPES에 컬럼이 추가되면 예전 parquet은 다시 만든다."""
    return path.exists() and set(DTYPES) <= set(pq.read_schema(path).names)


def convert(zp):
    out = DST / zp.with_suffix(".parquet").name
    with zipfile.ZipFile(zp) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(name) as f:
            # 옛 파일의 Tail_Number 에 UTF-8 이 아닌 바이트가 있다. 필요한 값은 모두 ASCII 라 latin-1 로 읽어도 안전하다
            df = pd.read_csv(f, usecols=list(DTYPES), dtype=DTYPES, engine="pyarrow", encoding="latin-1")
    tmp = out.with_suffix(".tmp")  # 중간에 끊겨도 깨진 parquet이 남지 않도록
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    return f"{out}  {len(df):,} rows"


if __name__ == "__main__":
    from_year = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    DST.mkdir(exist_ok=True)
    todo = [zp for zp in sorted(SRC.glob("ontime_*.zip"))
            if not up_to_date(DST / zp.with_suffix(".parquet").name) and int(zp.stem.split("_")[1]) >= from_year]
    with ProcessPoolExecutor(max_workers=3) as ex:  # 파일당 메모리 ~2GB. 많이 띄우면 메모리 한도로 프로세스가 죽는다
        for msg in ex.map(convert, todo):
            print(msg, flush=True)
