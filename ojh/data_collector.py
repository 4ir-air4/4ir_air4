import time
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests

try:
    import tomllib
except ImportError:
    import pip._vendor.tomli as tomllib

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
LOCAL_DATA_PATH = DATA_DIR / "aircraftlive_counts.csv"
# 비밀키는 저장소 최상위 .streamlit/secrets.toml 에 있다 (streamlit run app.py 를 최상위에서 실행하므로).
# ojh/ 안에 따로 둔 경우도 찾는다.
SECRETS_PATH = next(
    (p for p in (BASE_DIR / ".streamlit" / "secrets.toml", BASE_DIR.parent / ".streamlit" / "secrets.toml") if p.exists()),
    BASE_DIR.parent / ".streamlit" / "secrets.toml",
)

OPENSKY_API_URL = "https://opensky-network.org/api/states/all"
OPENSKY_AUTH_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_COLUMNS = [
    "temp_mean", "precip_mean", "cloud_mean", "wind_mean",
    "gust_mean", "precip_max", "gust_max", "rain_share", "severe_weather_idx"
]

cached_token = None
token_expiry_time = 0

def load_opensky_secrets():
    if not SECRETS_PATH.exists():
        print(f"   [오류] {SECRETS_PATH} 파일을 찾을 수 없습니다. 경로를 확인해 주세요.")
        return None, None
    try:
        with open(SECRETS_PATH, "rb") as f:
            secrets = tomllib.load(f)
        client_id = secrets.get("OPENSKY_CLIENT_ID") or secrets.get("opensky_client_id")
        client_secret = secrets.get("OPENSKY_CLIENT_SECRET") or secrets.get("opensky_client_secret")
        return client_id, client_secret
    except Exception as e:
        print(f"   [설정 로드 오류] secrets.toml 읽기 실패: {e}")
        return None, None

def get_opensky_access_token(client_id, client_secret):
    global cached_token, token_expiry_time
    if cached_token and time.time() < (token_expiry_time - 60):
        return cached_token
    try:
        data = {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret}
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        res = requests.post(OPENSKY_AUTH_URL, data=data, headers=headers, timeout=10)
        if res.status_code == 200:
            token_data = res.json()
            cached_token = token_data.get("access_token")
            token_expiry_time = time.time() + token_data.get("expires_in", 1800)
            return cached_token
    except Exception:
        pass
    return None

def get_live_opensky_states():
    client_id, client_secret = load_opensky_secrets()
    headers = {}
    if client_id and client_secret:
        token = get_opensky_access_token(client_id, client_secret)
        if token:
            headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.get(OPENSKY_API_URL, headers=headers, timeout=15)
        remaining_credits = response.headers.get("X-Rate-Limit-Remaining", "미확인")
        if response.status_code == 200:
            data = response.json()
            states = data.get("states", [])
            return states, remaining_credits
    except Exception:
        pass
    return [], "미확인"

def get_weather_for_coordinates(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,precipitation,cloud_cover,wind_speed_10m,wind_gusts_10m",
        "timezone": "UTC"
    }
    try:
        res = requests.get(WEATHER_FORECAST_URL, params=params, timeout=10)
        if res.status_code == 200:
            current = res.json().get("current", {})
            temp = current.get("temperature_2m", np.nan)
            precip = current.get("precipitation", 0.0) or 0.0
            cloud = current.get("cloud_cover", np.nan)
            wind = current.get("wind_speed_10m", np.nan)
            gust = current.get("wind_gusts_10m", np.nan) or np.nan
            rain_share = 1.0 if precip >= 0.1 else 0.0
            severe_idx = gust * precip if not np.isnan(gust) else 0.0

            return {
                "temp_mean": temp,
                "precip_mean": precip,
                "cloud_mean": cloud,
                "wind_mean": wind,
                "gust_mean": gust,
                "precip_max": precip,
                "gust_max": gust,
                "rain_share": rain_share,
                "severe_weather_idx": severe_idx
            }
    except Exception:
        pass
    return {col: np.nan for col in WEATHER_COLUMNS}

def check_is_holiday(dt: datetime) -> int:
    d = dt.date()
    y = d.year
    holiday_dates = {date(y, 1, 1), date(y, 7, 4), date(y, 11, 11), date(y, 12, 25), date(y, 12, 31)}
    thanksgiving = date(y, 11, 1) + timedelta(days=(3 - date(y, 11, 1).weekday() + 7) % 7 + 21)
    holiday_dates.update([thanksgiving, thanksgiving - timedelta(days=1), thanksgiving + timedelta(days=1)])
    return 1 if d in holiday_dates else 0

def collect_and_save():
    now_utc = datetime.now(timezone.utc)
    print(f"\n[{now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC] 실시간 항공 트래픽 수집 중...")

    states, remaining_credits = get_live_opensky_states()
    if not states:
        print("   -> 데이터를 가져오지 못해 이번 수집 주기를 건너뜁니다.")
        return

    valid_states = [s for s in states if s[5] is not None and s[6] is not None]
    if not valid_states:
        return

    active_aircrafts = {s[0] for s in valid_states if s[0]}
    aircraft_count = len(active_aircrafts)

    lats = [s[6] for s in valid_states]
    lons = [s[5] for s in valid_states]
    center_lat = float(np.median(lats))
    center_lon = float(np.median(lons))

    minute_floor = (now_utc.minute // 5) * 5
    timestamp = now_utc.replace(minute=minute_floor, second=0, microsecond=0)

    weather_data = get_weather_for_coordinates(center_lat, center_lon)

    # 의미 없는 0원인 지연/결항 더미 칼럼들을 제거하고 실제 수집 가능한 데이터 위주로 구성
    new_row = {
        "timestamp": timestamp.isoformat(),
        "aircraft_count": aircraft_count,
        "is_holiday": check_is_holiday(now_utc),
        **weather_data
    }

    new_df = pd.DataFrame([new_row])

    if LOCAL_DATA_PATH.exists():
        existing_df = pd.read_csv(LOCAL_DATA_PATH)
        existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"], utc=True)
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], utc=True)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")

    LOCAL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(LOCAL_DATA_PATH, index=False, encoding="utf-8-sig")

    print(f"   [수집 완료] 위치({center_lat:.2f}, {center_lon:.2f}) | 누적 행 수: {len(combined):,}개 (현재 항공기: {aircraft_count:,}대 / 남은 크레딧: {remaining_credits})")

if __name__ == "__main__":
    print("항공 트래픽 실시간 수집기를 시작합니다. (Ctrl+C로 중지)")
    try:
        while True:
            collect_and_save()
            time.sleep(300)
    except KeyboardInterrupt:
        print("\n수집기가 안전하게 중지되었습니다.")