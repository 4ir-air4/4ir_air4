"""공항 코드표: IATA -> ICAO, 기상 관측소 ID, 시간대. 받아 둔 참고 자료(OurAirports, OpenFlights)를 읽는다."""
import pandas as pd

from ldh.core.data import REF

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
