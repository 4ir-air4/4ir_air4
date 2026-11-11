import io
import gzip
import sys
import tarfile
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# 프로젝트 경로
BASE_DIR = Path(__file__).resolve().parents[1]  # ojh/collect -> ojh (data/, models/ 가 있는 곳)
DATA_DIR = BASE_DIR / "data" / "raw"
EXTERNAL_DATA_PATH = DATA_DIR / "opensky_counts.csv"

# True : 5분 구간 내 고유 항공기(icao24) 수
# False: 5분 구간 내 상태 레코드 수 (기존 방식)
COUNT_UNIQUE_AIRCRAFT = True

# 수집 기간 (명령행 인자로 덮어쓸 수 있음: python fetch_opensky.py 2017-01-01 2022-12-31)
START_DATE = "2017-01-01"
END_DATE = "2022-12-31"

# 이미 날씨까지 저장된 날짜는 건너뛴다. 다시 받으려면 True.
FORCE_REFETCH = False

# ---------------------------------------------------------
# 날씨 설정 (Open-Meteo 과거 날씨 API, 무료/키 불필요)
# ---------------------------------------------------------
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

# 위경도를 GRID_DEG도 격자로 묶고, 격자 중심 좌표의 날씨를 사용한다.
# 격자가 작을수록 정확하지만 API 호출량이 커진다.
GRID_DEG = 10

# 하루에 날씨를 가져올 격자 수 상한 (항공기가 많은 격자 순)
MAX_CELLS = 150

# 한 번의 API 요청에 넣을 격자 수
CELLS_PER_REQUEST = 50

WEATHER_VARS = [
    "temperature_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
]

# 비가 오는 격자로 볼 시간당 강수량 기준(mm)
RAIN_THRESHOLD_MM = 0.1

# 학습에 쓰는 최종 날씨 컬럼 (train.py와 이름이 같아야 한다)
WEATHER_COLUMNS = [
    "temp_mean",
    "precip_mean",
    "cloud_mean",
    "wind_mean",
    "gust_mean",
    "precip_max",
    "gust_max",
    "rain_share",
]


# =========================================================
# OpenSky 파일 읽기
# =========================================================

def read_opensky_bytes(content_bytes: bytes) -> pd.DataFrame:
    """
    OpenSky 시간별 파일을 DataFrame으로 변환합니다.
    다음 형태를 모두 처리합니다.
      - 순수 tar 안에 .csv.gz 가 들어 있는 경우 (현재 케이스)
      - gzip 으로 감싼 tar
      - tar 안의 순수 CSV
      - 압축 없는 CSV
    """
    data = content_bytes

    # 바깥이 gzip이면 먼저 해제
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)

    # tar이면 첫 번째 파일 멤버를 꺼냄
    if tarfile.is_tarfile(io.BytesIO(data)):
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            if not members:
                raise ValueError("tar 안에 파일이 없습니다.")
            data = tar.extractfile(members[0]).read()

        # 멤버 자체가 gzip(.csv.gz)인 경우 한 번 더 해제
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)

    return pd.read_csv(
        io.BytesIO(data),
        usecols=lambda c: c in ["time", "icao24", "lat", "lon"],
        low_memory=False,
    )


# =========================================================
# 시간별 파일 -> 5분 항공기 수 + 격자별 항공기 수
# =========================================================

def add_grid_cell(df: pd.DataFrame) -> pd.DataFrame:
    """위경도를 GRID_DEG도 격자 중심 좌표(lat_c, lon_c)로 바꿉니다."""
    df = df.copy()
    lat = pd.to_numeric(df["lat"], errors="coerce")
    lon = pd.to_numeric(df["lon"], errors="coerce")
    valid = lat.between(-90, 90) & lon.between(-180, 180)

    df["lat_c"] = np.where(
        valid, np.floor(lat / GRID_DEG) * GRID_DEG + GRID_DEG / 2, np.nan
    )
    df["lon_c"] = np.where(
        valid, np.floor(lon / GRID_DEG) * GRID_DEG + GRID_DEG / 2, np.nan
    )
    return df


def summarize_hour(df_hour: pd.DataFrame):
    """
    시간별 파일 하나를 두 결과로 요약합니다.
      1) 5분 단위 항공기 수 (timestamp, aircraft_count)
      2) 시간 x 격자별 고유 항공기 수 (hour, lat_c, lon_c, n)  <- 날씨 가중치용
    """
    df = df_hour.copy()
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
    df = df.dropna(subset=["timestamp"])

    indexed = df.set_index("timestamp")
    if COUNT_UNIQUE_AIRCRAFT and "icao24" in indexed.columns:
        counts = indexed.resample("5min")["icao24"].nunique()
    else:
        counts = indexed.resample("5min").size()
    counts_5min = counts.reset_index(name="aircraft_count")

    if {"lat", "lon", "icao24"}.issubset(df.columns):
        gridded = add_grid_cell(df).dropna(subset=["lat_c", "lon_c"])
        gridded["hour"] = gridded["timestamp"].dt.floor("h")
        cell_hour = (
            gridded.groupby(["hour", "lat_c", "lon_c"])["icao24"]
            .nunique()
            .reset_index(name="n")
        )
    else:
        cell_hour = pd.DataFrame(columns=["hour", "lat_c", "lon_c", "n"])

    return counts_5min, cell_hour


# =========================================================
# 날씨 가져오기 / 요약
# =========================================================

def _get_json_with_retry(url, params, attempts=4):
    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, timeout=120)
            if response.status_code == 200:
                return response.json()
            last_error = RuntimeError(
                f"Status {response.status_code}: {response.text[:200]}"
            )
        except requests.RequestException as error:
            last_error = error
        time.sleep(2 ** attempt)
    raise last_error


def fetch_weather(date_str: str, cells: pd.DataFrame) -> pd.DataFrame:
    """
    격자 중심 좌표들의 하루치 시간별 날씨를 가져옵니다.
    cells: lat_c, lon_c 컬럼을 가진 DataFrame
    반환: hour, lat_c, lon_c + WEATHER_VARS
    """
    frames = []
    cell_rows = list(cells[["lat_c", "lon_c"]].itertuples(index=False, name=None))

    for start in range(0, len(cell_rows), CELLS_PER_REQUEST):
        chunk = cell_rows[start:start + CELLS_PER_REQUEST]

        params = {
            "latitude": ",".join(f"{lat:g}" for lat, _ in chunk),
            "longitude": ",".join(f"{lon:g}" for _, lon in chunk),
            "start_date": date_str,
            "end_date": date_str,
            "hourly": ",".join(WEATHER_VARS),
            "timezone": "GMT",
        }

        payload = _get_json_with_retry(WEATHER_URL, params)

        # 위치가 하나면 dict, 여러 개면 입력 순서대로 list 로 온다.
        if isinstance(payload, dict):
            payload = [payload]

        for (lat_c, lon_c), location in zip(chunk, payload):
            hourly = location.get("hourly")
            if not hourly:
                continue
            weather = pd.DataFrame(hourly)
            weather["hour"] = pd.to_datetime(weather.pop("time"))
            weather["lat_c"] = lat_c
            weather["lon_c"] = lon_c
            frames.append(weather)

        time.sleep(0.5)

    if not frames:
        return pd.DataFrame(columns=["hour", "lat_c", "lon_c"] + WEATHER_VARS)

    return pd.concat(frames, ignore_index=True)


def summarize_weather(cell_hour: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    격자별 날씨를 '그 시간에 그 격자에 있던 항공기 수'로 가중 평균해서
    시간당 날씨 요약 한 줄로 만듭니다.
    """
    merged = cell_hour.merge(weather, on=["hour", "lat_c", "lon_c"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["hour"] + WEATHER_COLUMNS)

    by_hour = merged.groupby("hour")
    result = pd.DataFrame(index=by_hour.size().index)

    weighted = {
        "temp_mean": "temperature_2m",
        "precip_mean": "precipitation",
        "cloud_mean": "cloud_cover",
        "wind_mean": "wind_speed_10m",
        "gust_mean": "wind_gusts_10m",
    }

    for out_name, column in weighted.items():
        valid = merged[column].notna()
        weights = merged["n"].where(valid, 0)
        numerator = (merged[column].where(valid, 0) * weights).groupby(merged["hour"]).sum()
        denominator = weights.groupby(merged["hour"]).sum()
        result[out_name] = numerator / denominator.replace(0, np.nan)

    # 교통이 있는 격자 중 가장 심한 날씨
    result["precip_max"] = by_hour["precipitation"].max()
    result["gust_max"] = by_hour["wind_gusts_10m"].max()

    # 비가 오는 격자에 있던 항공기 비율
    rainy = (merged["precipitation"] >= RAIN_THRESHOLD_MM).astype(float)
    total = merged["n"].groupby(merged["hour"]).sum()
    rainy_n = (merged["n"] * rainy).groupby(merged["hour"]).sum()
    result["rain_share"] = rainy_n / total.replace(0, np.nan)

    return result.reset_index()[["hour"] + WEATHER_COLUMNS]


# =========================================================
# 하루 단위 처리
# =========================================================

def hour_url(date_str: str, hour: int) -> str:
    hour_str = f"{hour:02d}"
    return (
        f"https://s3.opensky-network.org/data-samples/states/"
        f".{date_str}/{hour_str}/states_{date_str}-{hour_str}.csv.tar"
    )


def day_exists(date_str: str) -> bool:
    """00시 파일이 있는지 가볍게 확인해서, 없는 날짜는 24번 시도하지 않고 건너뜁니다."""
    try:
        response = requests.head(hour_url(date_str, 0), timeout=30)
        return response.status_code == 200
    except requests.RequestException:
        return False


def fetch_one_day(date_str: str):
    """하루치(00~23시)를 받아 5분 항공기 수 + 날씨 요약 DataFrame 으로 반환합니다."""
    all_counts = []
    all_cells = []

    for hour in range(24):
        url = hour_url(date_str, hour)
        print(f"  [{hour + 1}/24] {url}")

        try:
            response = requests.get(url, timeout=60)
            if response.status_code != 200:
                print(f"     -> 파일 없음 (Status: {response.status_code})")
                continue

            df_hour = read_opensky_bytes(response.content)

            if df_hour.empty:
                print("     -> 데이터가 비어있습니다.")
                continue

            counts_5min, cell_hour = summarize_hour(df_hour)
            all_counts.append(counts_5min)
            all_cells.append(cell_hour)

        except Exception as error:
            print(f"     -> 오류: {type(error).__name__}: {error}")

    if not all_counts:
        return None

    day_df = pd.concat(all_counts, ignore_index=True)
    day_df = day_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    cell_hour = pd.concat(all_cells, ignore_index=True)
    cell_hour = cell_hour.drop_duplicates(subset=["hour", "lat_c", "lon_c"])

    # 항공기가 많이 지나간 격자 위주로 날씨를 가져온다.
    top_cells = (
        cell_hour.groupby(["lat_c", "lon_c"])["n"]
        .sum()
        .nlargest(MAX_CELLS)
        .reset_index()[["lat_c", "lon_c"]]
    )
    cell_hour = cell_hour.merge(top_cells, on=["lat_c", "lon_c"], how="inner")

    try:
        weather = fetch_weather(date_str, top_cells)
        hourly_summary = summarize_weather(cell_hour, weather)
        print(f"  날씨 격자 {len(top_cells)}개, 시간별 요약 {len(hourly_summary)}행")
    except Exception as error:
        print(f"  날씨를 가져오지 못했습니다 ({type(error).__name__}: {error})")
        hourly_summary = pd.DataFrame(columns=["hour"] + WEATHER_COLUMNS)

    day_df["hour"] = day_df["timestamp"].dt.floor("h")
    day_df = day_df.merge(hourly_summary, on="hour", how="left").drop(columns="hour")

    for column in WEATHER_COLUMNS:
        if column not in day_df.columns:
            day_df[column] = np.nan

    return day_df


# =========================================================
# 저장 / 기간 순회
# =========================================================

def save_merged(new_df: pd.DataFrame) -> pd.DataFrame:
    """새 데이터를 기존 CSV와 병합해 저장합니다(같은 timestamp는 새 값으로 덮어씀)."""
    if EXTERNAL_DATA_PATH.exists():
        existing_df = pd.read_csv(EXTERNAL_DATA_PATH)
        existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"])
        new_df = pd.concat([existing_df, new_df], ignore_index=True)

    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])
    new_df = (
        new_df.drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
    )

    EXTERNAL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(EXTERNAL_DATA_PATH, index=False, encoding="utf-8-sig")
    return new_df


def done_dates() -> set:
    """이미 날씨까지 저장된 날짜 집합."""
    if FORCE_REFETCH or not EXTERNAL_DATA_PATH.exists():
        return set()

    df = pd.read_csv(EXTERNAL_DATA_PATH)
    if not set(WEATHER_COLUMNS).issubset(df.columns):
        return set()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    has_weather = df.dropna(subset=WEATHER_COLUMNS)
    return set(has_weather["timestamp"].dt.strftime("%Y-%m-%d"))


def daterange(start: str, end: str):
    current = date.fromisoformat(start)
    last = date.fromisoformat(end)
    while current <= last:
        yield current.isoformat()
        current += timedelta(days=1)


def fetch_period(start: str = START_DATE, end: str = END_DATE):
    """start~end 각 날짜를 하루씩 받아 저장합니다. 중단돼도 다시 실행하면 이어서 진행합니다."""
    finished = done_dates()
    saved_days = 0

    for date_str in daterange(start, end):
        if date_str in finished:
            continue

        if not day_exists(date_str):
            continue

        print(f"\n=== {date_str} ===")
        day_df = fetch_one_day(date_str)

        if day_df is None or day_df.empty:
            print("  수집된 데이터가 없습니다.")
            continue

        merged = save_merged(day_df)
        saved_days += 1
        print(f"  저장 완료: {EXTERNAL_DATA_PATH} (누적 {len(merged):,}행)")

    print(f"\n[완료] 이번 실행에서 저장한 날짜: {saved_days}일")


if __name__ == "__main__":
    start_arg = sys.argv[1] if len(sys.argv) > 1 else START_DATE
    end_arg = sys.argv[2] if len(sys.argv) > 2 else END_DATE
    fetch_period(start_arg, end_arg)