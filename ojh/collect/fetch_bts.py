import io
import os
import sys
import zipfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------
# 프로젝트 경로 설정
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # ojh/collect -> ojh (data/, models/ 가 있는 곳)
DATA_DIR = BASE_DIR / "data" / "raw"
EXTERNAL_DATA_PATH = DATA_DIR / "btstranstats_counts.csv"  # train.py 연동용 저장 경로

# 수집 연도 범위
START_YEAR = 2000
END_YEAR = 2026

# 날씨 API 재시도 횟수 (실패하면 대기 후 다시 시도). 더 끈질기게 하려면 숫자를 늘리세요.
MAX_WEATHER_RETRIES = 8

# ---------------------------------------------------------
# 1. 날씨 설정 (Open-Meteo 과거 날씨 아카이브 API)
# ---------------------------------------------------------
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_VARS = [
    "temperature_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
]
WEATHER_COLUMNS = [
    "temp_mean",
    "precip_mean",
    "cloud_mean",
    "wind_mean",
    "gust_mean",
    "precip_max",
    "gust_max",
    "rain_share",
    "severe_weather_idx",
]

# 미국 주요 공항 위경도 매핑 사전 (출발 공항 기준 정밀 매핑)
AIRPORT_COORDS = {
    "ATL": (33.6407, -84.4277), "ORD": (41.9742, -87.9073), "DFW": (32.8998, -97.0403),
    "DEN": (39.8561, -104.6737), "LAX": (33.9416, -118.4085), "JFK": (40.6413, -73.7781),
    "SFO": (37.6213, -122.3790), "CLT": (35.2140, -80.9431), "LAS": (36.0840, -115.1537),
    "PHX": (33.4342, -112.0116), "MCO": (28.4312, -81.3081), "MIA": (25.7959, -80.2870),
    "SEA": (47.4502, -122.3088), "EWR": (40.6895, -74.1745), "MSP": (44.8848, -93.2223),
    "DTW": (42.2162, -83.3554), "BOS": (42.3656, -71.0096), "PHL": (39.8721, -75.2411)
}


def fetch_weather_for_coords_and_day(lat: float, lon: float, date_str: str):
    """
    지정한 위경도 좌표와 날짜를 기반으로 해당 위치의 시간별 과거 날씨 수집.
    실패(HTTP 오류, 429 제한, 네트워크 오류, 빈 응답)하면 대기 후 재시도하고,
    끝내 못 받으면 None을 반환한다. (빈 DataFrame이 아님 -> 호출 측에서 실패로 구분)
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "UTC",
    }

    for attempt in range(1, MAX_WEATHER_RETRIES + 1):
        reason = ""
        wait = min(2 ** attempt, 60)  # 2, 4, 8, ... 최대 60초
        try:
            res = requests.get(WEATHER_URL, params=params, timeout=30)
            if res.status_code == 200:
                hourly = res.json().get("hourly")
                if hourly and hourly.get("time"):
                    wdf = pd.DataFrame(hourly)
                    wdf["hour"] = pd.to_datetime(wdf.pop("time"), utc=True)

                    wdf["temp_mean"] = wdf["temperature_2m"]
                    wdf["precip_mean"] = wdf["precipitation"]
                    wdf["cloud_mean"] = wdf["cloud_cover"]
                    wdf["wind_mean"] = wdf["wind_speed_10m"]
                    wdf["gust_mean"] = wdf["wind_gusts_10m"]
                    wdf["precip_max"] = wdf["precipitation"]
                    wdf["gust_max"] = wdf["wind_gusts_10m"]
                    wdf["rain_share"] = (wdf["precipitation"] >= 0.1).astype(float)
                    wdf["severe_weather_idx"] = wdf["wind_gusts_10m"] * wdf["precipitation"]

                    return wdf[["hour"] + WEATHER_COLUMNS]
                reason = "응답에 hourly 데이터가 없음"
            else:
                reason = f"HTTP {res.status_code}"
                if res.status_code == 429:
                    wait = 60  # 요청 제한이면 좀 길게 쉬었다가 재시도
        except Exception as e:
            reason = str(e)

        print(f"   [날씨 수집 경고] {date_str} ({attempt}/{MAX_WEATHER_RETRIES}) {reason} -> {wait}초 후 재시도")
        time.sleep(wait)

    return None


# ---------------------------------------------------------
# 2. 미국 주요 공휴일 생성 함수
# ---------------------------------------------------------
def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """미국 주요 연방 공휴일 및 연휴 전후 특성(is_holiday) 추가"""
    df = df.copy()
    dates = df["timestamp"].dt.date
    years = df["timestamp"].dt.year.unique()
    holiday_dates = set()

    for y in years:
        holiday_dates.update([
            date(y, 1, 1),   # 새해
            date(y, 7, 4),   # 독립기념일
            date(y, 11, 11), # 참전용사의 날
            date(y, 12, 25), # 크리스마스
            date(y, 12, 31), # 연말
        ])
        thanksgiving = date(y, 11, 1) + timedelta(days=(3 - date(y, 11, 1).weekday() + 7) % 7 + 21)
        holiday_dates.update([thanksgiving, thanksgiving - timedelta(days=1), thanksgiving + timedelta(days=1)])

    df["is_holiday"] = dates.isin(holiday_dates).astype(int)
    return df


# ---------------------------------------------------------
# 3. BTS 데이터 처리 및 5분 단위 고급 집계
# ---------------------------------------------------------
def fetch_bts_monthly_data(year: int, month: int):
    """
    BTS TranStats 월별 데이터를 다운로드하여 
    항공기 수, 지연/결항/거리, 누적 롤링 지표, 위경도 기반 날씨, 공휴일 피처 통합
    날씨를 끝내 못 받으면 None을 반환(저장하지 않음 -> 다음 실행 때 다시 시도)
    """
    url = f"https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
    print(f"\n=== BTS 데이터 처리 중: {year}년 {month:02d}월 ===")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    try:
        res = requests.get(url, headers=headers, stream=True, timeout=120)
        if res.status_code != 200:
            print(f"   -> 해당 월 데이터가 없거나 접근 실패 (Status: {res.status_code})")
            return None

        # Zip 압축 해제 및 필요한 컬럼(Origin 포함) 추출
        with zipfile.ZipFile(io.BytesIO(res.content)) as z:
            csv_name = [f for f in z.namelist() if f.endswith(".csv")][0]
            with z.open(csv_name) as f:
                use_cols = [
                    "FlightDate", "DepTime", "Origin", "DepDelay", 
                    "Cancelled", "Diverted", "Distance"
                ]
                df = pd.read_csv(f, usecols=lambda c: c in use_cols, low_memory=False)

        print(f"   -> 원본 항공편 레코드 수: {len(df):,}행")

        # 데이터 정제 및 타임스탬프 생성
        df = df.dropna(subset=["FlightDate", "DepTime"])
        df["DepTime"] = df["DepTime"].astype(str).str.split(".").str[0].str.zfill(4)
        df["DepTime"] = df["DepTime"].replace("2400", "2359")

        df["timestamp_str"] = df["FlightDate"] + " " + df["DepTime"]
        df["timestamp"] = pd.to_datetime(df["timestamp_str"], format="%Y-%m-%d %H%M", errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])

        # 수치형 변수 정리
        df["Cancelled"] = pd.to_numeric(df["Cancelled"], errors="coerce").fillna(0)
        df["Diverted"] = pd.to_numeric(df["Diverted"], errors="coerce").fillna(0)
        df["DepDelay"] = pd.to_numeric(df["DepDelay"], errors="coerce").fillna(0)
        df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").fillna(0)
        df["Origin"] = df["Origin"].astype(str).str.upper()

        # 출발 공항 위경도 매핑 (사전에 없으면 미국 중심 좌표 39.8283, -98.5795 적용)
        coords_list = df["Origin"].map(lambda x: AIRPORT_COORDS.get(x, (39.8283, -98.5795)))
        df["lat"] = [c[0] for c in coords_list]
        df["lon"] = [c[1] for c in coords_list]

        # 해당 월 데이터의 대표 위경도 산출 (중앙값)
        rep_lat = float(df["lat"].median())
        rep_lon = float(df["lon"].median())

        # 5분 단위 집계
        df = df.set_index("timestamp")

        agg_dict = {
            "FlightDate": "size",      # 항공기 수 (aircraft_count)
            "DepDelay": "mean",        # 평균 출발 지연
            "Cancelled": "sum",        # 결항 수
            "Diverted": "sum",         # 회항 수
            "Distance": "mean"         # 평균 운항 거리
        }

        resampled = df.resample("5min").agg(agg_dict).reset_index()
        resampled.rename(columns={
            "FlightDate": "aircraft_count",
            "DepDelay": "avg_dep_delay",
            "Cancelled": "cancelled_count",
            "Diverted": "diverted_count",
            "Distance": "avg_distance"
        }, inplace=True)

        # 5분 빈(Bin)에서 항공편이 없어 NaN이 된 경우 결측 보정
        resampled["avg_dep_delay"] = resampled["avg_dep_delay"].fillna(0.0)
        resampled["avg_distance"] = resampled["avg_distance"].fillna(800.0)

        # 롤링 피처 생성
        resampled = resampled.set_index("timestamp").sort_index()
        resampled["rolling_delay_30m"] = resampled["avg_dep_delay"].rolling(6, min_periods=1).mean()
        resampled["rolling_cancel_1h"] = resampled["cancelled_count"].rolling(12, min_periods=1).sum()
        resampled = resampled.reset_index()

        # 공휴일 특성 추가
        resampled = add_holiday_features(resampled)

        print(f"   -> 5분 단위 집계 및 파생 피처 생성 완료: {len(resampled):,}행")

        # 일별 위경도 기반 날씨 데이터 결합 (실패 시 재시도, 끝내 실패하면 이 달은 저장 안 함)
        resampled["date_str"] = resampled["timestamp"].dt.strftime("%Y-%m-%d")
        unique_dates = resampled["date_str"].unique()

        weather_frames = []
        for d_str in unique_dates:
            w_df = fetch_weather_for_coords_and_day(rep_lat, rep_lon, d_str)
            if w_df is None:
                print(f"   -> {d_str} 날씨 수집 최종 실패: 이 달은 저장하지 않고 건너뜀 (다음 실행 때 다시 시도)")
                return None
            weather_frames.append(w_df)
            time.sleep(0.02)

        all_weather = pd.concat(weather_frames, ignore_index=True)
        resampled["hour"] = resampled["timestamp"].dt.floor("h")
        resampled = resampled.merge(all_weather, on="hour", how="left").drop(columns=["hour", "date_str"])
        for col in WEATHER_COLUMNS:
            resampled[col] = resampled[col].fillna(0.0)

        return resampled

    except Exception as e:
        print(f"   -> 오류 발생: {e}")
        return None


# ---------------------------------------------------------
# 4. 파일 저장 및 병합
# ---------------------------------------------------------
def save_and_merge(new_df: pd.DataFrame):
    """기존 데이터 파일에 누적 병합"""
    if EXTERNAL_DATA_PATH.exists():
        existing_df = pd.read_csv(EXTERNAL_DATA_PATH)
        existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"], utc=True)
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], utc=True)
        merged = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        merged = new_df

    # 같은 timestamp는 새로 받은 값(keep="last")으로 덮어씀 -> 날씨 빠진 달을 다시 받으면 교체됨
    merged = merged.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")

    EXTERNAL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(EXTERNAL_DATA_PATH, index=False, encoding="utf-8-sig")
    print(f"   [저장 완료] 경로: {EXTERNAL_DATA_PATH} (총 누적 행 수: {len(merged):,})")


def load_completed_months() -> set:
    """
    이미 저장된 CSV를 훑어서 '날씨까지 정상으로 받은 달'의 집합("YYYY-MM")을 반환.
    날씨 컬럼 9개가 전부 0인 행이 한 개라도 있으면 날씨 수집에 실패한 달로 보고 완료 목록에서 제외.
    (기온·구름·풍속이 동시에 정확히 0일 수는 사실상 없으므로 실패 판별 기준으로 사용)
    """
    if not EXTERNAL_DATA_PATH.exists():
        return set()

    print("기존 CSV에서 날씨 수집 완료 월을 확인하는 중...")
    total_rows = {}
    no_weather_rows = {}
    cols = ["timestamp"] + WEATHER_COLUMNS

    for chunk in pd.read_csv(EXTERNAL_DATA_PATH, usecols=lambda c: c in cols, chunksize=500_000):
        ym = pd.to_datetime(chunk["timestamp"], utc=True).dt.strftime("%Y-%m")
        missing = chunk[WEATHER_COLUMNS].fillna(0.0).abs().sum(axis=1) == 0

        for key, n in ym.value_counts().items():
            total_rows[key] = total_rows.get(key, 0) + int(n)
        for key, n in ym[missing].value_counts().items():
            no_weather_rows[key] = no_weather_rows.get(key, 0) + int(n)

    completed = {k for k in total_rows if no_weather_rows.get(k, 0) == 0}
    redo = sorted(k for k in total_rows if k not in completed)
    print(f"   -> 완료된 달: {len(completed)}개 / 날씨 누락으로 다시 받을 달: {len(redo)}개 {redo[:12]}{' ...' if len(redo) > 12 else ''}")
    return completed


def main():
    completed = load_completed_months()
    failed = []

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            if year == 2026 and month > 10:
                break

            key = f"{year}-{month:02d}"
            if key in completed:
                print(f"[건너뜀] {key} (이미 날씨까지 수집 완료)")
                continue

            monthly_df = fetch_bts_monthly_data(year, month)
            if monthly_df is not None and not monthly_df.empty:
                save_and_merge(monthly_df)
            else:
                failed.append(key)

    if failed:
        print(f"\n수집 실패/미존재 월 ({len(failed)}개): {failed}")
        print("-> 다시 실행하면 이 달들만 재시도합니다.")
    else:
        print("\n모든 달 수집 완료")


if __name__ == "__main__":
    main()