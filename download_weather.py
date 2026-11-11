"""train_enhanced.py 에 필요한 참고 자료와 BTS 전체 공항의 기상 관측(METAR)을 받습니다.

사용법: python download_weather.py   (prepare_data.py 실행 후)
저장 위치: reference_data/  이미 받은 파일은 건너뜁니다.
"""
import sys
import time
import urllib.parse
from pathlib import Path

import pandas as pd

from download_refs import FILES, fetch

REF = Path("reference_data")
YEARS = {2023: "2024-01-01", 2024: "2025-01-01", 2025: "2026-01-01", 2026: "2026-09-02"}
REF_FILES = ["ourairports/airports.csv", "ourairports/runways.csv", "openflights/airports.dat",
             "opensky/aircraftDatabase.csv"]
# OurAirports에 PBI 코드가 없는 팜비치(DJT로 표기)와 OpenFlights에 시간대가 없는 공항
PBI_IDENT = "KPBI"
TZ_FIXES = {"BIH": "America/Los_Angeles", "DJT": "America/New_York", "EAR": "America/Chicago",
            "HEZ": "America/Chicago", "XWA": "America/Chicago"}


def airport_idents():
    """IATA 코드 -> ICAO 식별자."""
    oa = pd.read_csv(REF / "ourairports/airports.csv", usecols=["ident", "iata_code"], dtype="string")
    ident = oa.dropna().drop_duplicates("iata_code").set_index("iata_code").ident
    ident["PBI"] = PBI_IDENT
    return ident


def stations(codes):
    """BTS 공항 코드 -> IEM 관측소 ID. 미국 본토는 K를 뺀 3글자, 그 외는 ICAO 그대로 사용한다."""
    ident = airport_idents().reindex(codes)
    ids = ident.where(~ident.fillna("").str.fullmatch(r"K\w{3}"), ident.str[1:])
    return ids.replace({"PBI": "DJT"})  # 공항 개명으로 IEM 관측소 ID는 전 기간 DJT


def airport_timezones():
    of = pd.read_csv(REF / "openflights/airports.dat", header=None, na_values="\\N",
                     usecols=[4, 11], names=["iata", "tz"], dtype="string")
    tz = of.dropna().drop_duplicates("iata").set_index("iata").tz
    return pd.concat([tz, pd.Series(TZ_FIXES, dtype="string")])


def weather_url(ids, year, end):
    y2, m2, d2 = map(int, end.split("-"))
    return "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" + urllib.parse.urlencode({
        "station": ",".join(ids), "data": "tmpf,sknt,gust,vsby,p01i,skyc1,skyl1,wxcodes",
        "year1": year, "month1": 1, "day1": 1, "year2": y2, "month2": m2, "day2": d2,
        "tz": "Etc/UTC", "format": "onlycomma", "latlon": "no", "missing": "empty", "trace": "empty",
        "report_type": "3,4"})


def main():
    failed = [n for n in REF_FILES if not fetch(FILES[n], REF / n)]
    if failed:
        sys.exit(f"실패: {failed}")
    files = sorted(Path("data").glob("ontime_202[3-6]_*.parquet"))
    routes = pd.concat(pd.read_parquet(f, columns=["Origin", "Dest"]) for f in files)
    codes = sorted(set(routes.Origin.dropna()) | set(routes.Dest.dropna()))
    ids = stations(codes)
    assert ids.notna().all(), f"관측소 없는 공항: {ids[ids.isna()].index.tolist()}"
    for year, end in YEARS.items():  # IEM 제한(요청당 1000 관측소-년)을 넘지 않도록 연도별로 받는다
        if not fetch(weather_url(sorted(set(ids)), year, end), REF / f"weather_asos/all_{year}.csv"):
            failed.append(year)
        time.sleep(2)
    if failed:
        sys.exit(f"실패: {failed}")


if __name__ == "__main__":
    main()
