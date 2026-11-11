"""공항별 월간 이용객 수 + 국가별 GDP 성장률 수집기 (미국 · 유럽 · 한국).

[공항 이용객 탭] 전용 데이터는 모두 data/airport_demand/ 안에 모읍니다.
(data/raw/ 의 btstranstats_counts.csv, opensky_counts.csv 등 '항공기 수' 파일과 섞이지 않습니다)

만드는 파일
    data/airport_demand/airport_demand_passengers_monthly.csv   airport, airport_name, country, region, date, passengers, source
    data/airport_demand/airport_demand_gdp_growth.csv           country, year, gdp_growth

입력 데이터 (없는 지역은 건너뜁니다)  -- data/airport_demand/input/ 아래
    us_bts/*.csv|zip          BTS TranStats > T-100 Market (All Carriers) 에서 받은 CSV (연도별로 여러 개 가능)
                              필수 컬럼: YEAR, MONTH, ORIGIN, DEST, PASSENGERS, ORIGIN_COUNTRY, DEST_COUNTRY
    europe_eurostat/*.tsv|csv|json   Eurostat avia_paoa. 비어 있으면 API로 연도별 자동 다운로드
    uk_caa/*.csv              영국 CAA 월별 공항 통계(Table 09). 없는 달만 caa.co.uk 에서 자동 다운로드
                              (브렉시트 이후 Eurostat에 영국 데이터가 없어서 2020-11 이후는 이걸로 채움)
    korea_airportal/          에어포탈(airportal.go.kr) 시계열 통계. 2009년부터 연도별로 자동 다운로드 (airportal_YYYY.json)
                              직접 받은 시계열 엑셀(.xlsx/.xls)을 넣어도 함께 읽음

사용법
    python ojh/collect/fetch_airports.py              # 전체 수집 (예전 위치의 파일은 자동으로 옮겨줍니다)
    python ojh/collect/fetch_airports.py --gdp-only   # GDP만 다시 받기
"""

import gzip
import io
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[1]  # ojh/collect -> ojh (data/, models/ 가 있는 곳)
RAW_DIR = BASE_DIR / "data" / "raw"
# 공항 이용객 탭 전용 폴더: 기존 항공기 수 CSV(btstranstats_counts.csv 등)와 섞이지 않도록 따로 둔다.
AIRPORT_DIR = BASE_DIR / "data" / "airport_demand"
INPUT_DIR = AIRPORT_DIR / "input"
T100_DIR = INPUT_DIR / "us_bts"                 # 미국  : BTS T-100 (직접 받은 CSV)
EUROSTAT_DIR = INPUT_DIR / "europe_eurostat"    # 유럽  : Eurostat (API로 받은 연도별 파일)
UK_CAA_DIR = INPUT_DIR / "uk_caa"                # 영국  : CAA 월별 공항 통계 (자동 다운로드)
KOREA_DIR = INPUT_DIR / "korea_airportal"       # 한국  : 에어포탈 시계열 엑셀 (직접 받아서 넣기)
OUT_PASSENGERS = AIRPORT_DIR / "airport_demand_passengers_monthly.csv"
OUT_GDP = AIRPORT_DIR / "airport_demand_gdp_growth.csv"

# 예전 위치 -> 새 위치 (처음 실행할 때 자동으로 옮긴다. 이미 받은 데이터를 다시 받을 필요 없음)
LEGACY_MOVES = [
    (RAW_DIR / "airport_passengers_monthly.csv", OUT_PASSENGERS),
    (RAW_DIR / "gdp_growth.csv", OUT_GDP),
    (RAW_DIR / "us_bts", T100_DIR),
    (RAW_DIR / "europe_eurostat", EUROSTAT_DIR),
    (RAW_DIR / "extra_manual", KOREA_DIR),
    (INPUT_DIR / "extra_manual", KOREA_DIR),       # 예전 한국용 폴더 이름
    (RAW_DIR / "t100", T100_DIR),
    (RAW_DIR / "eurostat", EUROSTAT_DIR),
    (RAW_DIR / "extra", KOREA_DIR),
]


def migrate_legacy_files():
    """data/raw/ 에 있던 공항 이용객 관련 파일·폴더를 data/airport_demand/ 로 옮긴다."""
    import shutil

    for old_path, new_path in LEGACY_MOVES:
        if not old_path.exists() or old_path == new_path:
            continue
        new_path.parent.mkdir(parents=True, exist_ok=True)
        if old_path.is_dir():
            new_path.mkdir(parents=True, exist_ok=True)
            for item in list(old_path.iterdir()):
                target = new_path / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
            try:
                old_path.rmdir()
            except OSError:
                pass
        elif not new_path.exists():
            shutil.move(str(old_path), str(new_path))
        else:
            continue
        print(f"[정리] {old_path.relative_to(BASE_DIR)} -> {new_path.relative_to(BASE_DIR)}")

START_YEAR = 2000
# Eurostat 주소는 두 종류다. format 값이 서로 다르니 섞어 쓰면 'Invalid value for wsOutputFormat' 400 오류가 난다.
#   SDMX 2.1  : format = TSV | SDMX-CSV | JSON ...   시간 필터 = startPeriod / endPeriod   (키: freq.unit.tra_meas.rep_airp)
#   Statistics: format = JSON 만 지원                시간 필터 = sinceTimePeriod / untilTimePeriod (연도 단위)
EUROSTAT_SDMX_API = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/avia_paoa"
EUROSTAT_STATS_API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/avia_paoa"
WORLDBANK_API = "https://api.worldbank.org/v2/country/{codes}/indicator/NY.GDP.MKTP.KD.ZG"

# Eurostat 국가코드 -> World Bank(ISO2) 국가코드
COUNTRY_FIX = {"EL": "GR", "UK": "GB"}
EUROPE = {
    "AT", "BE", "BG", "CH", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GB", "GR", "HR", "HU",
    "IE", "IS", "IT", "LT", "LU", "LV", "ME", "MK", "MT", "NL", "NO", "PL", "PT", "RO", "RS", "SE",
    "SI", "SK", "TR", "AL", "BA", "XK", "LI",
}

AIRPORT_NAMES = {
    # 미국 (IATA)
    "ATL": "애틀랜타 하츠필드-잭슨", "ORD": "시카고 오헤어", "DFW": "댈러스/포트워스", "DEN": "덴버",
    "LAX": "로스앤젤레스", "JFK": "뉴욕 JFK", "SFO": "샌프란시스코", "CLT": "샬럿", "LAS": "라스베이거스",
    "PHX": "피닉스", "MCO": "올랜도", "MIA": "마이애미", "SEA": "시애틀", "EWR": "뉴어크", "MSP": "미니애폴리스",
    "DTW": "디트로이트", "BOS": "보스턴", "PHL": "필라델피아", "LGA": "뉴욕 라과디아", "FLL": "포트로더데일",
    "IAH": "휴스턴 조지 부시", "BWI": "볼티모어", "SLC": "솔트레이크시티", "SAN": "샌디에이고", "HNL": "호놀룰루",
    "IAD": "워싱턴 덜레스", "DCA": "워싱턴 레이건", "TPA": "탬파", "PDX": "포틀랜드", "AUS": "오스틴",
    # 유럽 (ICAO)
    "EGLL": "런던 히스로", "LFPG": "파리 샤를드골", "EDDF": "프랑크푸르트", "EHAM": "암스테르담 스키폴",
    "LEMD": "마드리드 바라하스", "LEBL": "바르셀로나", "EDDM": "뮌헨", "LIRF": "로마 피우미치노",
    "LTFM": "이스탄불", "LTFJ": "이스탄불 사비하 괵첸", "EGKK": "런던 개트윅", "EIDW": "더블린",
    "LSZH": "취리히", "LOWW": "빈", "EKCH": "코펜하겐", "ENGM": "오슬로", "ESSA": "스톡홀름 알란다",
    "EFHK": "헬싱키", "LPPT": "리스본", "EBBR": "브뤼셀", "LFPO": "파리 오를리", "EDDB": "베를린 브란덴부르크",
    "EDDL": "뒤셀도르프", "EDDH": "함부르크", "LEPA": "팔마데마요르카", "LEMG": "말라가", "LGAV": "아테네",
    "EPWA": "바르샤바", "LKPR": "프라하", "LHBP": "부다페스트", "EGSS": "런던 스탠스테드",
    "EGCC": "맨체스터", "LIMC": "밀라노 말펜사", "EGGW": "런던 루턴", "EGPH": "에든버러", "EGBB": "버밍엄",
    # 한국 (IATA)
    "ICN": "인천국제공항", "GMP": "김포", "PUS": "김해(부산)", "CJU": "제주", "TAE": "대구", "CJJ": "청주",
    "KWJ": "광주", "MWX": "무안", "RSU": "여수", "USN": "울산", "HIN": "사천", "KPO": "포항경주",
    "KUV": "군산", "WJU": "원주", "YNY": "양양",
}


LEGACY_DIR_NAMES = {"us_bts": "t100", "europe_eurostat": "eurostat"}


def _list_files(folder, patterns):
    """새 폴더와 예전 이름 폴더(t100/eurostat/extra)에 있는 파일을 함께 찾는다."""
    folders = [Path(folder)]
    legacy = LEGACY_DIR_NAMES.get(Path(folder).name)
    if legacy:
        folders.append(Path(folder).parent / legacy)
    return sorted(f for d in folders if d.exists() for pat in patterns for f in d.glob(pat))


def _read_text(path):
    return Path(path).read_text(encoding="utf-8-sig", errors="replace")


def _is_us(series):
    s = series.astype(str).str.strip().str.upper()
    return (s == "US") | s.str.startswith("UNITED STATES")


# ---------------------------------------------------------
# 1. 미국: BTS T-100 Market (All Carriers)
# ---------------------------------------------------------
def load_t100():
    files = _list_files(T100_DIR, ("*.csv", "*.zip"))
    if not files:
        print(f"[T-100] {T100_DIR} 에 파일이 없어 미국 데이터는 건너뜁니다.")
        return pd.DataFrame()

    frames = []
    for path in files:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            print(f"[T-100] {path.name} 읽기 실패: {exc}")
            continue
        df.columns = [str(c).strip().upper() for c in df.columns]
        canon = {n.replace("_", ""): n for n in ("YEAR", "MONTH", "ORIGIN", "DEST", "PASSENGERS", "ORIGIN_COUNTRY", "DEST_COUNTRY", "ORIGIN_COUNTRY_NAME", "DEST_COUNTRY_NAME")}
        df.columns = [canon.get(c.replace("_", ""), c) for c in df.columns]

        oc = next((c for c in ("ORIGIN_COUNTRY", "ORIGIN_COUNTRY_NAME") if c in df.columns), None)
        dc = next((c for c in ("DEST_COUNTRY", "DEST_COUNTRY_NAME") if c in df.columns), None)
        need = ["YEAR", "MONTH", "ORIGIN", "DEST", "PASSENGERS"]
        missing = [c for c in need if c not in df.columns] + ([] if oc else ["ORIGIN_COUNTRY"]) + ([] if dc else ["DEST_COUNTRY"])
        if missing:
            print(f"[T-100] {path.name}: 필요한 컬럼이 없어 건너뜁니다 -> {missing}")
            print(f"        현재 컬럼: {list(df.columns)[:30]}")
            continue

        df["PASSENGERS"] = pd.to_numeric(df["PASSENGERS"], errors="coerce").fillna(0)
        df["date"] = pd.to_datetime(
            {"year": pd.to_numeric(df["YEAR"], errors="coerce"), "month": pd.to_numeric(df["MONTH"], errors="coerce"), "day": 1},
            errors="coerce",
        )
        df = df.dropna(subset=["date"])

        # 공항 이용객 = 출발 승객 + 도착 승객 (미국 공항만)
        dep = df[_is_us(df[oc])].groupby(["ORIGIN", "date"])["PASSENGERS"].sum()
        arr = df[_is_us(df[dc])].groupby(["DEST", "date"])["PASSENGERS"].sum()
        dep.index.set_names(["airport", "date"], inplace=True)
        arr.index.set_names(["airport", "date"], inplace=True)
        total = dep.add(arr, fill_value=0).rename("passengers").reset_index()
        total["source"] = "BTS T-100"
        frames.append(total)
        print(f"[T-100] {path.name}: 공항-월 {len(total):,}개")

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["airport", "date"], keep="last")
    out["airport_name"] = out["airport"].map(AIRPORT_NAMES).fillna(out["airport"])
    out["country"] = "US"
    out["region"] = "US"
    return out


# ---------------------------------------------------------
# 2. 유럽: Eurostat avia_paoa
# ---------------------------------------------------------
def _clean_number(series):
    return pd.to_numeric(series.astype(str).str.extract(r"^\s*(-?\d+(?:\.\d+)?)")[0], errors="coerce")


def _pick_dimension_values(df, dims):
    """airport/time 이외의 차원(freq, unit, tra_meas, schedule 등)에서 '총 승객' 행만 남긴다."""
    prefer = {"unit": ["PAS"], "freq": ["M"], "tra_meas": ["PAS_CRD", "PAS_PAS_CRD"], "tra_cov": ["TOTAL", "TOT"], "schedule": ["TOT", "TOTAL"]}
    for dim in dims:
        if dim not in df.columns or dim in ("rep_airp", "TIME_PERIOD"):
            continue
        values = list(df[dim].dropna().unique())
        if len(values) <= 1:
            continue
        chosen = next((v for v in prefer.get(dim, ["TOT", "TOTAL"]) if v in values), None)
        if chosen is None and dim == "tra_meas":
            chosen = next((v for v in values if "PAS_CRD" in str(v)), None)
        if chosen is None:
            print(f"[Eurostat] 차원 '{dim}'에 값이 여러 개({values[:8]})인데 고를 기준을 못 찾았습니다. 데이터 내려받을 때 필터를 하나로 맞춰주세요.")
            return None
        df = df[df[dim] == chosen]
    return df


def _long_from_jsonstat(payload):
    """Eurostat JSON-stat(Statistics API, format=JSON) -> rep_airp, TIME_PERIOD, passengers"""
    ids = list(payload["id"])
    sizes = [int(n) for n in payload["size"]]
    codes = {}
    for dim in ids:
        index = payload["dimension"][dim]["category"]["index"]
        if isinstance(index, list):
            index = {c: i for i, c in enumerate(index)}
        codes[dim] = [c for c, _ in sorted(index.items(), key=lambda kv: kv[1])]
    values = payload.get("value") or {}
    if isinstance(values, list):
        values = {i: v for i, v in enumerate(values) if v is not None}
    time_dim = next((d for d in ids if d.lower() in ("time", "time_period")), None)
    if not values or time_dim is None or "rep_airp" not in ids:
        return pd.DataFrame()

    flat = np.array([int(k) for k in values.keys()])
    positions = np.unravel_index(flat, sizes)
    df = pd.DataFrame({dim: np.array(codes[dim], dtype=object)[pos] for dim, pos in zip(ids, positions)})
    df["passengers"] = [np.nan if v is None else float(v) for v in values.values()]
    df = df.rename(columns={time_dim: "TIME_PERIOD"})
    df = _pick_dimension_values(df, [d for d in ids if d not in ("rep_airp", time_dim)])
    if df is None:
        return pd.DataFrame()
    df = df[df["TIME_PERIOD"].astype(str).str.fullmatch(r"\d{4}-\d{2}")]  # 월 단위만
    return df[["rep_airp", "TIME_PERIOD", "passengers"]]


def parse_eurostat_text(text):
    """Eurostat TSV(와이드) / SDMX-CSV(롱) / JSON-stat 텍스트 -> airport, country, date, passengers"""
    stripped = text.lstrip()
    if stripped.startswith("{"):  # JSON-stat (또는 오류 본문)
        payload = json.loads(stripped)
        if "error" in payload:
            print(f"[Eurostat] 오류 응답입니다: {str(payload['error'])[:200]}")
            return pd.DataFrame()
        long = _long_from_jsonstat(payload)
        if long.empty:
            return pd.DataFrame()
    elif "\t" in text.split("\n", 1)[0] and "\\" in text.split("\n", 1)[0]:  # 예) freq,unit,tra_meas,rep_airp\TIME_PERIOD <TAB> 2019-01 <TAB> ...
        raw = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
        head = raw.columns[0]
        dims = head.split("\\")[0].split(",")
        month_cols = [c for c in raw.columns[1:] if re.fullmatch(r"\d{4}-\d{2}", c.strip())]
        keys = raw[head].str.split(",", expand=True)
        keys.columns = dims
        wide = pd.concat([keys, raw[month_cols]], axis=1)
        wide.columns = list(dims) + [c.strip() for c in month_cols]
        wide = _pick_dimension_values(wide, dims)
        if wide is None:
            return pd.DataFrame()
        long = wide.melt(id_vars=["rep_airp"], value_vars=[c.strip() for c in month_cols], var_name="TIME_PERIOD", value_name="v")
        long["passengers"] = _clean_number(long["v"])
    else:
        df = pd.read_csv(io.StringIO(text), dtype=str)
        cols = {c.lower(): c for c in df.columns}
        need = ["rep_airp", "time_period", "obs_value"]
        if any(n not in cols for n in need):
            print(f"[Eurostat] 알 수 없는 CSV 형식입니다. 컬럼: {list(df.columns)[:15]}")
            return pd.DataFrame()
        df = df.rename(columns={cols["rep_airp"]: "rep_airp", cols["time_period"]: "TIME_PERIOD", cols["obs_value"]: "obs_value"})
        dims = [c for c in df.columns if c.lower() in ("freq", "unit", "tra_meas", "tra_cov", "schedule")]
        df = _pick_dimension_values(df, dims)
        if df is None:
            return pd.DataFrame()
        long = df[df["TIME_PERIOD"].str.fullmatch(r"\d{4}-\d{2}")].copy()
        long["passengers"] = pd.to_numeric(long["obs_value"], errors="coerce")

    long = long.dropna(subset=["passengers"])
    long["date"] = pd.to_datetime(long["TIME_PERIOD"].str.strip() + "-01", errors="coerce")
    code = long["rep_airp"].astype(str).str.split("_", n=1, expand=True)
    long["country"] = code[0].replace(COUNTRY_FIX)
    long["airport"] = code[1] if code.shape[1] > 1 else code[0]
    return long[["airport", "country", "date", "passengers"]].dropna()


def _eurostat_attempts(year):
    """한 해치를 받는 방법 3가지. 주소마다 format 값과 시간 필터 이름이 다르다 (위 상수 설명 참고)."""
    sdmx_url = f"{EUROSTAT_SDMX_API}/M.PAS.PAS_CRD."  # 키 = freq.unit.tra_meas.rep_airp (마지막 빈 칸 = 모든 공항)
    window = {"startPeriod": f"{year}-01", "endPeriod": f"{year}-12", "compressed": "false"}
    return [
        ("SDMX 2.1 / TSV", "tsv", sdmx_url, {"format": "TSV", **window}),
        ("SDMX 2.1 / CSV", "csv", sdmx_url, {"format": "SDMX-CSV", **window}),
        ("Statistics / JSON", "json", EUROSTAT_STATS_API, {
            "format": "JSON", "freq": "M", "unit": "PAS", "tra_meas": "PAS_CRD",
            "sinceTimePeriod": str(year), "untilTimePeriod": str(year),
        }),
    ]


def _download_eurostat_year(year):
    """한 해치(12개월)만 요청한다. 방법을 바꿔 가며 최대 3번 시도한다.

    반환: (파일 경로 또는 None, 상태)   상태 = "ok" | "nodata"(그 해 데이터가 없음) | "error"(요청 자체가 거부됨)
    """
    errors, nodata = [], 0
    attempts = _eurostat_attempts(year)
    for label, ext, url, params in attempts:
        try:
            res = requests.get(url, params=params, timeout=300)
        except Exception as exc:
            errors.append(f"{label}: 연결 실패 {exc}")
            continue
        if res.status_code == 404:  # 해당 기간 데이터가 없는 경우
            nodata += 1
            continue
        if res.status_code != 200:
            errors.append(f"{label}: HTTP {res.status_code} {res.text[:200].strip()}")
            continue
        content = res.content
        if content[:2] == b"\x1f\x8b":  # gzip으로 온 경우
            content = gzip.decompress(content)
        text = content.decode("utf-8-sig", errors="replace")
        if len(text) < 200:
            nodata += 1
            continue
        target = EUROSTAT_DIR / f"avia_paoa_{year}.{ext}"
        target.write_text(text, encoding="utf-8")
        return target, "ok"
    if nodata == len(attempts) or (nodata and not errors):
        return None, "nodata"
    print(f"[Eurostat] {year}년 실패")
    for line in errors:
        print(f"      - {line}")
    return None, "error"


def download_eurostat_by_year():
    EUROSTAT_DIR.mkdir(parents=True, exist_ok=True)
    files, errors_in_a_row, skipped = [], 0, []
    for year in range(START_YEAR, date.today().year + 1):
        cached = next((p for p in EUROSTAT_DIR.glob(f"avia_paoa_{year}.*") if p.stat().st_size > 200), None)
        if cached is not None:  # 이미 받은 해는 건너뛴다 (중간에 끊겨도 이어받기 가능)
            files.append(cached)
            continue
        path, status = _download_eurostat_year(year)
        if status == "nodata":  # 오래된 해에는 월별 데이터가 없을 수 있다. 오류가 아니라 건너뛴다.
            skipped.append(year)
            continue
        if status == "error":
            errors_in_a_row += 1
            if not files and errors_in_a_row >= 3:  # 한 해도 못 받았는데 요청이 3번 연속 거부되면 주소/파라미터 문제로 보고 멈춘다
                print("[Eurostat] 처음 3개 연도의 요청이 모두 거부되어 중단합니다. 위의 오류 내용을 확인해주세요.")
                break
            continue
        errors_in_a_row = 0
        files.append(path)
        print(f"[Eurostat] {year}년 저장 ({path.stat().st_size / 1e6:.1f} MB)")
    if skipped:
        print(f"[Eurostat] 데이터가 없어 건너뛴 해: {skipped[0]}~{skipped[-1]} ({len(skipped)}개 연도)")
    return files


def load_eurostat():
    files = _list_files(EUROSTAT_DIR, ("*.tsv", "*.csv", "*.json"))
    if not files:
        print(f"[Eurostat] 로컬 파일이 없어 {START_YEAR}년부터 연도별로 내려받습니다...")
        files = download_eurostat_by_year()
        if not files:
            print("   -> 내려받기에 실패했습니다. 위의 오류 내용을 확인하거나, 직접 받아서 data/airport_demand/input/europe_eurostat/ 에 넣어주세요:")
            print("      Eurostat Data Browser의 'avia_paoa' 표에서 주기=월(M), 단위=Passenger, 측정=Passengers carried 로 필터 후 TSV 다운로드")
            return pd.DataFrame()

    frames = []
    for path in files:
        try:
            part = parse_eurostat_text(_read_text(path))
        except Exception as exc:
            print(f"[Eurostat] {path.name} 처리 실패: {exc}")
            continue
        print(f"[Eurostat] {path.name}: 공항-월 {len(part):,}개")
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["airport", "date"], keep="last")
    out = out[out["date"].dt.year >= START_YEAR]
    out["airport_name"] = out["airport"].map(AIRPORT_NAMES).fillna(out["airport"])
    out["region"] = "Europe"
    out["source"] = "Eurostat"
    return out


# ---------------------------------------------------------
# 2-1. 영국 공항 (UK CAA) -- 브렉시트 이후 Eurostat에 영국이 빠져서 따로 받는다
# ---------------------------------------------------------
# CAA는 API가 없고, 달마다 페이지가 있고 그 안에 'Table 09 Terminal and Transit Passengers' CSV가 있다.
#   연도 페이지 -> 월 페이지 -> Table 09 CSV 링크 순서로 찾아 내려받는다.
# terminal_pax 값이 Eurostat의 영국 'Passengers carried' 값과 거의 같아서(예: 히스로 2015-01 5,454,900) 그대로 이어 붙일 수 있다.
UK_CAA_BASE = "https://www.caa.co.uk"
UK_CAA_INDEX = UK_CAA_BASE + "/data-and-analysis/uk-aviation-market/airports/uk-airport-data/uk-airport-data-{year}/"
UK_CAA_START_YEAR = 2015          # 이 해부터 CSV가 있다 (그 이전은 스캔 PDF)
UK_CAA_HEADERS = {"User-Agent": "Mozilla/5.0 (airport-demand-research)"}
UK_CAA_DELAY_SEC = 0.5            # 사이트에 부담을 주지 않도록 요청 사이에 쉬는 시간
MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august",
               "september", "october", "november", "december"]

# CAA 공항 이름(부분 문자열) -> ICAO 코드. 위에서부터 먼저 맞는 것을 쓴다 (예: 'BELFAST CITY'가 'BELFAST'보다 먼저).
UK_CAA_ICAO = [
    ("HEATHROW", "EGLL"), ("GATWICK", "EGKK"), ("STANSTED", "EGSS"), ("LUTON", "EGGW"), ("LONDON CITY", "EGLC"),
    ("SOUTHEND", "EGMC"), ("MANCHESTER", "EGCC"), ("BIRMINGHAM", "EGBB"), ("EDINBURGH", "EGPH"),
    ("PRESTWICK", "EGPK"), ("GLASGOW", "EGPF"), ("BRISTOL", "EGGD"), ("BELFAST CITY", "EGAC"),
    ("BELFAST INTERNATIONAL", "EGAA"), ("NEWCASTLE", "EGNT"), ("LIVERPOOL", "EGGP"), ("EAST MIDLANDS", "EGNX"),
    ("LEEDS BRADFORD", "EGNM"), ("ABERDEEN", "EGPD"), ("SOUTHAMPTON", "EGHI"), ("CARDIFF", "EGFF"),
    ("DONCASTER", "EGCN"), ("EXETER", "EGTE"), ("INVERNESS", "EGPE"), ("BOURNEMOUTH", "EGHH"),
    ("NORWICH", "EGSH"), ("NEWQUAY", "EGHQ"), ("SUMBURGH", "EGPB"), ("CITY OF DERRY", "EGAE"),
    ("HUMBERSIDE", "EGNJ"), ("KIRKWALL", "EGPA"), ("TEESSIDE", "EGNV"), ("DURHAM TEES", "EGNV"),
    ("STORNOWAY", "EGPO"), ("WICK", "EGPC"), ("ISLAY", "EGPI"), ("BENBECULA", "EGPL"), ("DUNDEE", "EGPN"),
    ("BLACKPOOL", "EGNH"), ("BARRA", "EGPR"), ("TIREE", "EGPU"), ("CAMPBELTOWN", "EGEC"), ("LERWICK", "EGET"),
    ("BIGGIN HILL", "EGKB"), ("ISLES OF SCILLY", "EGHE"), ("LANDS END", "EGHC"), ("FARNBOROUGH", "EGLF"),
    ("OXFORD", "EGTK"), ("CAMBRIDGE", "EGSC"), ("COVENTRY", "EGBE"), ("GLOUCESTERSHIRE", "EGBJ"),
    ("SCATSTA", "EGPM"), ("PLYMOUTH", "EGHD"), ("MANSTON", "EGMH"), ("LYDD", "EGMD"),
]


def _uk_icao(name):
    name = str(name).upper()
    return next((code for key, code in UK_CAA_ICAO if key in name), None)


def _caa_get(url):
    time.sleep(UK_CAA_DELAY_SEC)
    res = requests.get(url if url.startswith("http") else UK_CAA_BASE + url, headers=UK_CAA_HEADERS, timeout=60)
    res.raise_for_status()
    return res


def download_uk_caa():
    """아직 받지 않은 달의 Table 09 CSV만 내려받아 uk_caa/caa_t09_YYYYMM.csv 로 저장한다."""
    UK_CAA_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today()
    saved, failed = 0, []
    for year in range(UK_CAA_START_YEAR, today.year + 1):
        months = [m for m in range(1, 13) if (year, m) < (today.year, today.month)]
        missing = [m for m in months if not (UK_CAA_DIR / f"caa_t09_{year}{m:02d}.csv").exists()]
        if not missing:
            continue
        try:
            index_html = _caa_get(UK_CAA_INDEX.format(year=year)).text
        except Exception as exc:
            print(f"[UK CAA] {year}년 목록 페이지 실패: {exc}")
            continue
        for m in missing:
            name = MONTH_NAMES[m - 1]
            link = re.search(rf'href="([^"]*uk-airport-data-{year}/{name}-{year}/?)"', index_html)
            if not link:  # 아직 공개되지 않은 달 (CAA는 보통 2달 정도 늦게 올린다)
                continue
            try:
                page = _caa_get(link.group(1)).text
                csv_link = re.search(r'href="([^"]+)"[^>]*>\s*<span>\s*Table 09[^<]*\(CSV', page)
                if not csv_link:
                    failed.append(f"{year}-{m:02d} (CSV 링크 없음)")
                    continue
                content = _caa_get(csv_link.group(1)).content
            except Exception as exc:
                failed.append(f"{year}-{m:02d} ({exc})")
                continue
            (UK_CAA_DIR / f"caa_t09_{year}{m:02d}.csv").write_bytes(content)
            saved += 1
            print(f"[UK CAA] {year}-{m:02d} 저장")
    if saved:
        print(f"[UK CAA] 새로 받은 달: {saved}개")
    if failed:
        print(f"[UK CAA] 실패한 달: {', '.join(failed)}")


def load_uk_caa(download=True):
    if download:
        try:
            download_uk_caa()
        except Exception as exc:
            print(f"[UK CAA] 다운로드 중단: {exc} (이미 받은 파일만 사용합니다)")
    files = sorted(UK_CAA_DIR.glob("caa_t09_*.csv"))
    if not files:
        print("[UK CAA] 파일이 없어 영국 2020-11 이후 데이터는 빠집니다.")
        return pd.DataFrame()

    rows = []
    for path in files:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as exc:
            print(f"[UK CAA] {path.name} 읽기 실패: {exc}")
            continue
        df.columns = [c.strip().lower() for c in df.columns]
        need = {"this_period", "last_period", "reporting_airport_name", "terminal_pax_this_period", "terminal_pax_last_period"}
        if not need.issubset(df.columns):
            print(f"[UK CAA] {path.name}: 형식이 달라 건너뜁니다. (컬럼 {list(df.columns)[:8]})")
            continue
        if "reporting_airport_group_name" in df.columns:  # 저지·건지·맨섬 등 영국 밖 공항 제외
            df = df[~df["reporting_airport_group_name"].astype(str).str.contains("Non UK", case=False)]
        for period_col, pax_col, priority in (("this_period", "terminal_pax_this_period", 1), ("last_period", "terminal_pax_last_period", 0)):
            part = df[["reporting_airport_name", period_col, pax_col]].copy()
            part.columns = ["name", "period", "passengers"]
            part["priority"] = priority  # 같은 달이 두 파일에 있으면 그 달 파일의 값(this_period)을 우선
            rows.append(part)
    if not rows:
        return pd.DataFrame()

    data = pd.concat(rows, ignore_index=True)
    data["airport"] = data["name"].map(_uk_icao)
    unknown = sorted(set(data.loc[data["airport"].isna(), "name"].astype(str)))
    if unknown:
        print(f"[UK CAA] ICAO 코드를 모르는 공항 {len(unknown)}개는 제외: {unknown[:10]}")
    data = data.dropna(subset=["airport"])
    data["date"] = pd.to_datetime(data["period"].astype(str).str[:6], format="%Y%m", errors="coerce")
    data["passengers"] = pd.to_numeric(data["passengers"], errors="coerce")
    data = data.dropna(subset=["date", "passengers"])
    data = data.sort_values("priority").drop_duplicates(["airport", "date", "name"], keep="last")
    data = data.groupby(["airport", "date"], as_index=False)["passengers"].sum()

    data["country"] = "GB"
    data["region"] = "Europe"
    data["airport_name"] = data["airport"].map(AIRPORT_NAMES).fillna(data["airport"])
    data["source"] = "UK CAA"
    print(f"[UK CAA] 공항 {data['airport'].nunique()}개 | {data['date'].min():%Y-%m} ~ {data['date'].max():%Y-%m} | {len(data):,}행")
    return data


# ---------------------------------------------------------
# 3. 한국 공항 (에어포탈 시계열 엑셀)
# ---------------------------------------------------------
# 에어포탈 > 항공통계 > 상세조회 > 시계열 (https://www.airportal.go.kr/stats/transport/chartDetail5.do)
#   구분=공항, 여객 으로 조회 후 '엑셀 다운로드'한 파일은 세로형이다 (한 번에 최대 1년치):
#       공항명      기간           여객(명)
#       전체 합계   2026년 01월    11648121
#       인천(ICN)   2026년 01월    6875632
#   (화면에 보이는 표처럼 월이 옆으로 나열된 가로형 '공항명 | 2026-07 | 2026-08 ...' 도 읽는다)
#   -> 공항명 안의 괄호 속 3글자가 공항 코드(IATA).
# 주의: 에어포탈 기준은 '국내선 출발 + 국제선 출발·도착'이라 국내선 승객은 출발만 센다.
#       (인천은 거의 국제선이라 영향이 작지만 김포·제주는 실제보다 작게 잡힌다)
KOREA_MONTH_RE = re.compile(r"^\s*(\d{4})\s*[-./년]\s*(\d{1,2})\s*월?\s*(?:[-./]\s*\d{1,2})?(?:\s+00:00:00)?\s*$")
KOREA_CODE_RE = re.compile(r"\(([A-Z]{3})\)")


def _cell_month(value):
    if isinstance(value, (pd.Timestamp, date)):
        return value.year, value.month
    m = KOREA_MONTH_RE.match(str(value))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _read_korea_sheets(path):
    """엑셀(xlsx/xls), 엑셀처럼 저장된 HTML 표, CSV 를 모두 '머리글 없는 표' 목록으로 읽는다."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm"):
        return list(pd.read_excel(path, header=None, sheet_name=None, engine="openpyxl").values())
    if suffix == ".xls":
        try:
            return list(pd.read_excel(path, header=None, sheet_name=None, engine="xlrd").values())
        except Exception:
            return pd.read_html(path, header=None, thousands=",")  # 정부 사이트가 HTML 표를 .xls 로 주는 경우
    if suffix == ".csv":
        for enc in ("utf-8-sig", "cp949"):
            try:
                return [pd.read_csv(path, header=None, encoding=enc)]
            except UnicodeDecodeError:
                continue
    return []


def _parse_korea_long(raw, month_col):
    """세로형: 공항명 | 기간(2026년 01월) | 여객(명)  -- 에어포탈 시계열 '엑셀 다운로드' 파일이 이 모양이다."""
    name_col = next(
        (j for j in range(raw.shape[1]) if j != month_col and raw.iloc[:, j].astype(str).str.contains(r"\([A-Z]{3}\)").any()),
        None,
    )
    if name_col is None:
        return pd.DataFrame()
    others = [j for j in range(raw.shape[1]) if j not in (month_col, name_col)]
    # 값 칸: 위쪽 머리글에 '여객'이 있는 칸, 없으면 숫자가 가장 많은 칸
    head = raw.head(5).astype(str)
    value_col = next((j for j in others if head.iloc[:, j].str.contains("여객").any()), None)
    if value_col is None and others:
        numeric = {j: pd.to_numeric(raw.iloc[:, j].astype(str).str.replace(",", ""), errors="coerce").notna().sum() for j in others}
        value_col = max(numeric, key=numeric.get)
    if value_col is None:
        return pd.DataFrame()

    rows = []
    for _, r in raw.iterrows():
        code = KOREA_CODE_RE.search(str(r.iloc[name_col]))
        ym = _cell_month(r.iloc[month_col])
        if not code or not ym:  # 머리글, '전체 합계' 줄은 건너뜀
            continue
        value = pd.to_numeric(str(r.iloc[value_col]).replace(",", "").strip(), errors="coerce")
        rows.append({"airport": code.group(1), "date": pd.Timestamp(year=ym[0], month=ym[1], day=1), "passengers": value})
    return pd.DataFrame(rows)


def _parse_korea_sheet(raw):
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    # 한 칸(열)에 월이 여러 개 세로로 있으면 세로형, 아니면 월이 머리글 줄에 가로로 있는 가로형
    month_counts = {j: raw.iloc[:, j].map(lambda v: _cell_month(v) is not None).sum() for j in range(raw.shape[1])}
    if month_counts and max(month_counts.values()) >= 2:
        return _parse_korea_long(raw, max(month_counts, key=month_counts.get))

    header, months = None, {}
    for i in range(min(len(raw), 30)):
        found = {j: _cell_month(v) for j, v in enumerate(raw.iloc[i]) if _cell_month(v)}
        if found:
            header, months = i, found
            break
    if header is None:
        return pd.DataFrame()

    body = raw.iloc[header + 1:]
    name_col = next(
        (j for j in range(raw.shape[1]) if j not in months and body.iloc[:, j].astype(str).str.contains(r"\([A-Z]{3}\)").any()),
        None,
    )
    if name_col is None:
        return pd.DataFrame()

    rows = []
    for _, r in body.iterrows():
        code = KOREA_CODE_RE.search(str(r.iloc[name_col]))
        if not code:  # '전체 합계' 같은 줄은 건너뜀
            continue
        for j, (y, m) in months.items():
            value = pd.to_numeric(str(r.iloc[j]).replace(",", "").strip(), errors="coerce")
            rows.append({"airport": code.group(1), "date": pd.Timestamp(year=y, month=m, day=1), "passengers": value})
    return pd.DataFrame(rows)


# 에어포탈 시계열 화면이 '조회'를 누를 때 부르는 주소. 로그인 없이 JSON으로 POST 하면 공항x월 여객 수를 돌려준다.
# (화면에서는 1년까지만 고를 수 있어서 1년씩 나눠 요청한다)
AIRPORTAL_API = "https://www.airportal.go.kr/stats/transport/getDetailedAirTransportStats2Time.do"
AIRPORTAL_HEADERS = {
    "Content-Type": "application/json;charset=UTF-8",
    "request-call": "web-rest",
    "Referer": "https://www.airportal.go.kr/stats/transport/chartDetail5.do",
    "User-Agent": "Mozilla/5.0 (airport-demand-research)",
}
AIRPORTAL_START_YEAR = 2009       # 에어포탈에서 고를 수 있는 가장 이른 해
AIRPORTAL_DELAY_SEC = 1.0         # 사이트에 부담을 주지 않도록 요청 사이에 쉬는 시간
AIRPORTAL_NAME_RE = re.compile(r"\(([A-Z]{3})\)\s*(\d{4})\s*년\s*(\d{1,2})\s*월")


def download_korea_airportal():
    """연도별로 airportal_YYYY.json 을 저장한다. 끝난 해는 한 번만 받고, 올해·작년은 숫자가 갱신될 수 있어 매번 다시 받는다."""
    KOREA_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today()
    saved = 0
    for year in range(AIRPORTAL_START_YEAR, today.year + 1):
        target = KOREA_DIR / f"airportal_{year}.json"
        if target.exists() and year < today.year - 1:
            continue
        payload = {
            "last_yearmonth": f"{year}01", "this_yearmonth": f"{year}12",
            "di_gubun": "total",        # 국내선 + 국제선
            "arvl_type": "IA_D",        # 국내선 출발 + 국제선 출도착 (에어포탈 기준)
            "sn_gubun": "total", "pass_gubun": "4",  # 운항 전체 / 여객 = 유임+환승
            "carge_gubun": "total", "airport_gubun": "total", "pyn_gubun": "total",
        }
        time.sleep(AIRPORTAL_DELAY_SEC)
        try:
            res = requests.post(AIRPORTAL_API, data=json.dumps(payload), headers=AIRPORTAL_HEADERS, timeout=60)
            res.raise_for_status()
            rows = res.json().get("content") or []
        except Exception as exc:
            print(f"[한국] {year}년 요청 실패: {exc}")
            continue
        if not rows:
            continue
        target.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        saved += 1
        print(f"[한국] 에어포탈 {year}년 저장 ({len(rows)}행)")
    if saved:
        print(f"[한국] 에어포탈에서 새로 받은 해: {saved}개")


def _parse_airportal_json(path):
    rows = []
    for rec in json.loads(Path(path).read_text(encoding="utf-8")):
        m = AIRPORTAL_NAME_RE.search(str(rec.get("airport_name", "")))  # 예: '인천(ICN) 2015년 01월' ('전체 합계'는 코드가 없어 빠짐)
        if not m:
            continue
        value = pd.to_numeric(str(rec.get("pass", "")).replace(",", ""), errors="coerce")
        rows.append({"airport": m.group(1), "date": pd.Timestamp(year=int(m.group(2)), month=int(m.group(3)), day=1), "passengers": value})
    return pd.DataFrame(rows)


def load_korea(download=True):
    if download:
        try:
            download_korea_airportal()
        except Exception as exc:
            print(f"[한국] 에어포탈 자동 다운로드 중단: {exc} (이미 받은 파일만 사용합니다)")
    files = _list_files(KOREA_DIR, ("*.json", "*.xlsx", "*.xlsm", "*.xls", "*.csv"))
    files = [f for f in files if not f.name.startswith("~$")]  # 엑셀이 열려 있을 때 생기는 임시 파일 제외
    if not files:
        print(f"[한국] {KOREA_DIR.relative_to(BASE_DIR)} 에 파일이 없어 한국 공항은 빠집니다.")
        return pd.DataFrame()

    frames = []
    for path in files:
        try:
            if path.suffix.lower() == ".json":
                parsed = [_parse_airportal_json(path)]
            else:  # 직접 받은 엑셀도 그대로 읽는다
                parsed = [_parse_korea_sheet(sh) for sh in _read_korea_sheets(path)]
        except Exception as exc:
            print(f"[한국] {path.name} 읽기 실패: {exc}")
            continue
        parsed = [p for p in parsed if not p.empty]
        if not parsed:
            print(f"[한국] {path.name}: 'YYYY-MM' 월 칸이나 '인천(ICN)' 같은 공항명을 찾지 못했습니다. "
                  "에어포탈 '시계열' 화면에서 구분=공항, 여객 으로 받은 파일인지 확인해주세요.")
            continue
        part = pd.concat(parsed, ignore_index=True)
        part["file"] = path.name
        frames.append(part)
        print(f"[한국] {path.name}: 공항 {part['airport'].nunique()}개 | {part['date'].min():%Y-%m} ~ {part['date'].max():%Y-%m}")
    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True).dropna(subset=["passengers"])
    data = data.drop_duplicates(["airport", "date"], keep="last")  # 기간이 겹치면 이름순으로 뒤 파일 값 사용
    data["country"] = "KR"
    data["region"] = "Korea"
    data["airport_name"] = data["airport"].map(AIRPORT_NAMES).fillna(data["airport"])
    data["source"] = "Airportal"
    print(f"[한국] 합계: 공항 {data['airport'].nunique()}개 | {data['date'].min():%Y-%m} ~ {data['date'].max():%Y-%m} | {len(data):,}행")
    return data[["airport", "airport_name", "country", "region", "date", "passengers", "source"]]


# ---------------------------------------------------------
# 4. 국가별 GDP 성장률 (World Bank)
# ---------------------------------------------------------
def parse_worldbank_json(payload):
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        return pd.DataFrame(columns=["country", "year", "gdp_growth"])
    rows = [
        {"country": r["country"]["id"], "year": int(r["date"]), "gdp_growth": r["value"]}
        for r in payload[1]
        if r.get("value") is not None
    ]
    return pd.DataFrame(rows, columns=["country", "year", "gdp_growth"])


def fetch_gdp(countries):
    codes = sorted({c for c in countries if isinstance(c, str) and len(c) == 2})
    if not codes:
        return pd.DataFrame()
    url = WORLDBANK_API.format(codes=";".join(codes))
    params = {"format": "json", "per_page": 20000, "date": f"{START_YEAR - 1}:{date.today().year}"}
    try:
        res = requests.get(url, params=params, timeout=60)
        res.raise_for_status()
        gdp = parse_worldbank_json(res.json())
    except Exception as exc:
        print(f"[GDP] World Bank API 실패: {exc}")
        return pd.DataFrame()
    print(f"[GDP] {gdp['country'].nunique()}개 국가, {len(gdp):,}행")
    return gdp


# ---------------------------------------------------------
# 5. 합치기
# ---------------------------------------------------------
def main():
    migrate_legacy_files()
    for folder in (AIRPORT_DIR, T100_DIR, EUROSTAT_DIR, UK_CAA_DIR, KOREA_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    gdp_only = "--gdp-only" in sys.argv

    if gdp_only and OUT_PASSENGERS.exists():
        countries = pd.read_csv(OUT_PASSENGERS, usecols=["country"])["country"].unique()
    else:
        # 순서 중요: 같은 공항·같은 달이 겹치면 뒤의 출처가 이긴다 (영국은 Eurostat보다 CAA 값을 우선)
        parts = [p for p in (load_t100(), load_eurostat(), load_uk_caa(), load_korea()) if p is not None and not p.empty]
        if not parts:
            print("\n수집된 데이터가 없습니다. data/airport_demand/input/ 아래 us_bts, europe_eurostat, uk_caa, korea_airportal 폴더를 확인해주세요.")
            return
        cols = ["airport", "airport_name", "country", "region", "date", "passengers", "source"]
        data = pd.concat([p[cols] for p in parts], ignore_index=True)
        data = data[(data["passengers"] > 0) & (data["date"].dt.year >= START_YEAR)]
        data = data.drop_duplicates(["airport", "date"], keep="last").sort_values(["airport", "date"])
        data.to_csv(OUT_PASSENGERS, index=False, encoding="utf-8-sig")
        countries = data["country"].unique()

        print("\n=== 수집 결과 ===")
        print(f"저장: {OUT_PASSENGERS}")
        print(f"공항 {data['airport'].nunique():,}개 | 행 {len(data):,}개 | 기간 {data['date'].min():%Y-%m} ~ {data['date'].max():%Y-%m}")
        print(data.groupby("region")["airport"].nunique().rename("공항 수").to_string())

    gdp = fetch_gdp(countries)
    if not gdp.empty:
        gdp.to_csv(OUT_GDP, index=False, encoding="utf-8-sig")
        print(f"저장: {OUT_GDP}")
    elif not OUT_GDP.exists():
        print("[GDP] 파일이 없어 GDP 피처는 평균값으로 대체됩니다. 인터넷 연결 후 `python ojh/collect/fetch_airports.py --gdp-only` 로 다시 받으세요.")


if __name__ == "__main__":
    main()