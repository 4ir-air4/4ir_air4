from pathlib import Path
from datetime import datetime
import importlib.util
from html import escape
import json
import math
import re
import time

import folium
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from folium.plugins import Fullscreen, FastMarkerCluster
from streamlit_folium import st_folium


# =========================================================
# 1. 기본 설정
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models" / "saved"
TRAIN_SCRIPT = ROOT_DIR / "models" / "train.py"
LIVE_DATA_PATH = ROOT_DIR / "data" / "raw" / "aircraftlive_counts.csv"
HISTORICAL_DATA_PATH = ROOT_DIR / "data" / "raw" / "btstranstats_counts.csv"
STATES_CACHE_PATH = ROOT_DIR / "data" / "cache" / "opensky_states_last.json"

# 마지막 수집 후 이 시간(분)이 지나면 수집기가 멈춘 것으로 보고 경고한다
LIVE_STALE_MINUTES = 15
# 학습 샘플이 이보다 적으면(약 1주일치 미만) 성능 수치가 불안정하다고 안내한다
RECOMMENDED_SAMPLES = 2000
# 실시간 탭에서 보여줄 미래 예측 시점(분). 5분 단위 배수로 자유롭게 바꿔도 된다.
HORIZON_MINUTES = [5, 10, 30, 60, 180]

# 모델은 데이터 성격별로 따로 학습한다 (models/saved/<key>/ 아래에 저장)
DATASETS = {
    "현재 (실시간 수집)": {
        "key": "current",
        "is_live": True,
        "caption": (
            "실시간 API 스냅샷에서 위치가 확인되는 항공기 수를 5분마다 "
            "수집한 데이터로 학습한 모델입니다. "
            "실제 승객 수나 항공사 예약 수요를 예측하는 값은 아닙니다."
        ),
    },
    "과거 (BTS TranStats 샘플)": {
        "key": "historical",
        "is_live": False,
        "caption": (
            "BTS TranStats 과거 시간별 데이터에서 5분 구간별 고유 항공기(icao24) 수를 "
            "집계해 학습한 모델입니다. 측정 방식이 실시간 수집 데이터와 달라 "
            "두 결과의 수치를 직접 비교할 수는 없습니다."
        ),
    },
}

MODEL_SLUGS = {
    "Linear Regression": "linear_regression",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
}


def get_model_path(dataset_key, model_name):
    return MODEL_DIR / dataset_key / f"model_{MODEL_SLUGS[model_name]}.pkl"


def get_comparison_file(dataset_key):
    return MODEL_DIR / dataset_key / "model_comparison.csv"

OPENSKY_BASE_URL = "https://opensky-network.org/api"

# 항공기 클릭 위치와 실제 항공기 좌표의 최대 허용 거리
CLICK_DISTANCE_LIMIT_KM = 3

st.set_page_config(
    page_title="항공 교통 모니터링 및 수요 예측",
    page_icon="✈️",
    layout="wide",
)


# =========================================================
# 2. OpenSky 인증
# =========================================================

def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return ""


@st.cache_data(ttl=600, show_spinner=False)
def get_access_token(client_id, client_secret):

    token_url = (
        "https://auth.opensky-network.org/auth/realms/opensky-network"
        "/protocol/openid-connect/token"
    )

    response = requests.post(
        token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=20,
    )

    response.raise_for_status()

    token = response.json().get("access_token")

    if not token:
        raise RuntimeError("OpenSky 인증 토큰을 가져오지 못했습니다.")

    return token


def get_opensky_headers():

    client_id = get_secret("OPENSKY_CLIENT_ID")
    client_secret = get_secret("OPENSKY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError(
            "OpenSky 인증 정보가 없습니다.\n"
            ".streamlit/secrets.toml을 확인해주세요."
        )

    token = get_access_token(client_id, client_secret)

    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }


# =========================================================
# 3. 실시간 항공기 데이터
# =========================================================

@st.cache_data(ttl=600, show_spinner=False)
def load_aircraft_data():

    headers = get_opensky_headers()

    # 전 세계 상태 벡터(extended)는 응답이 커서 느릴 수 있다.
    # 읽기 제한을 길게 잡고, 실패하면 몇 번 다시 시도한다.
    result = None
    last_error = None

    for attempt in range(3):

        try:

            response = requests.get(
                f"{OPENSKY_BASE_URL}/states/all",
                headers={**headers, "Accept-Encoding": "gzip"},
                params={"extended": 1},
                timeout=(10, 120),  # (연결, 읽기) 초
            )

            response.raise_for_status()

            remaining_credits = response.headers.get(
                "X-Rate-Limit-Remaining", "정보 없음"
            )
            print(f"📌 [OpenSky] 오늘 남은 크레딧(X-Rate-Limit-Remaining): {remaining_credits}")

            result = response.json()

            # 다음 실패에 대비해 마지막 성공 응답을 파일로 남겨 둔다
            try:
                STATES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                STATES_CACHE_PATH.write_text(
                    json.dumps(result), encoding="utf-8"
                )
            except Exception:
                pass

            break

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:

            last_error = exc
            time.sleep(2 * (attempt + 1))

    if result is None:

        # 모두 실패하면 마지막으로 받아 둔 데이터를 보여 준다 (기준 시각 문구로 오래된 데이터인지 알 수 있다)
        if STATES_CACHE_PATH.exists():

            print(f"⚠️ [OpenSky] 조회 실패, 저장된 마지막 데이터를 사용합니다: {last_error}")

            result = json.loads(STATES_CACHE_PATH.read_text(encoding="utf-8"))

        else:

            raise last_error

    api_timestamp = result.get("time")
    states = result.get("states") or []

    columns = [
        "icao24",
        "callsign",
        "origin_country",
        "time_position",
        "last_contact",
        "longitude",
        "latitude",
        "baro_altitude",
        "on_ground",
        "velocity",
        "true_track",
        "vertical_rate",
        "sensors",
        "geo_altitude",
        "squawk",
        "spi",
        "position_source",
        "category",
    ]

    df = pd.DataFrame(states, columns=columns)

    if df.empty:
        return api_timestamp, df

    df["icao24"] = (
        df["icao24"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    df["callsign"] = (
        df["callsign"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df.dropna(
        subset=["latitude", "longitude"]
    )

    # 공항 근처 송신 장비 테스트용 신호(TEST1234)는 실제 항공기가 아니므로 제외
    df = df[
        ~df["callsign"].str.upper().str.startswith("TEST1234")
    ]

    return api_timestamp, df


# =========================================================
# 4. 항공기 이동경로 조회
# =========================================================

@st.cache_data(ttl=600, show_spinner=False)
def load_aircraft_track(icao24):

    headers = get_opensky_headers()

    response = requests.get(
        f"{OPENSKY_BASE_URL}/tracks/all",
        headers=headers,
        params={
            "icao24": icao24.lower(),
            "time": 0,
        },
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


# ---------------------------------------------------------
# 4-1. 궤적 조회용 로컬 프록시 (브라우저 CORS 차단 회피)
# ---------------------------------------------------------

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

TRACK_PROXY_PORT = 8765
TRACK_PROXY_CACHE_SEC = 120


def _make_track_handler(client_id, client_secret):
    token_state = {"token": None, "exp": 0.0}
    token_lock = threading.Lock()
    cache = {}

    def get_token():
        with token_lock:
            if token_state["token"] and time.time() < token_state["exp"]:
                return token_state["token"]
            r = requests.post(
                "https://auth.opensky-network.org/auth/realms/opensky-network"
                "/protocol/openid-connect/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
                timeout=20,
            )
            r.raise_for_status()
            body = r.json()
            token_state["token"] = body["access_token"]
            token_state["exp"] = time.time() + max(60, body.get("expires_in", 1800) - 60)
            return token_state["token"]

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body=b"{}"):
            self.send_response(status)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            qs = parse_qs(urlparse(self.path).query)
            icao24 = (qs.get("icao24") or [""])[0].lower()
            if not re.fullmatch(r"[0-9a-f]{6}", icao24):
                return self._send(400)

            hit = cache.get(icao24)
            if hit and time.time() - hit[0] < TRACK_PROXY_CACHE_SEC:
                return self._send(hit[1], hit[2])

            try:
                r = requests.get(
                    f"{OPENSKY_BASE_URL}/tracks/all",
                    headers={"Authorization": f"Bearer {get_token()}"},
                    params={"icao24": icao24, "time": 0},
                    timeout=30,
                )
                status, body = r.status_code, r.content
            except Exception:
                status, body = 502, b"{}"

            if status in (200, 404):  # 성공/데이터 없음만 캐시 (429·502는 다시 시도)
                cache[icao24] = (time.time(), status, body)
            self._send(status, body)

        def log_message(self, *args):
            pass

    return Handler


@st.cache_resource(show_spinner=False)
def start_track_proxy(client_id, client_secret):
    try:
        server = ThreadingHTTPServer(
            ("127.0.0.1", TRACK_PROXY_PORT),
            _make_track_handler(client_id, client_secret),
        )
    except OSError:
        return None  # 이미 실행 중이면(코드 수정 후 재실행 등) 기존 서버를 그대로 사용
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# =========================================================
# 5. 항공기 정보 가공
# =========================================================

def format_altitude(value):

    if pd.isna(value):
        return "정보 없음"

    return f"{value:,.0f} m"


def format_speed(value):

    if pd.isna(value):
        return "정보 없음"

    return f"{value * 3.6:,.1f} km/h"


# 고도 구간별 색상 (상한 m, 색상, 범례 문구). 낮은 고도일수록 붉은색, 높을수록 보라색
ALT_BANDS = [
    (300, "#e41a1c", "300m 미만"),
    (900, "#ff7f00", "300 ~ 900m"),
    (1500, "#ffd400", "900 ~ 1,500m"),
    (3000, "#a6d96a", "1,500 ~ 3,000m"),
    (4500, "#1a9850", "3,000 ~ 4,500m"),
    (6000, "#00b3b3", "4,500 ~ 6,000m"),
    (8000, "#2c7fb8", "6,000 ~ 8,000m"),
    (10000, "#5e4fa2", "8,000 ~ 10,000m"),
]
ALT_ABOVE = ("#8e24aa", "10,000m 이상")
ALT_GROUND_COLOR = "#7f7f7f"
ALT_NONE_COLOR = "#bdbdbd"


def get_altitude_color(altitude, on_ground=False):

    if on_ground:
        return ALT_GROUND_COLOR

    if pd.isna(altitude):
        return ALT_NONE_COLOR

    for limit, color, _ in ALT_BANDS:
        if altitude < limit:
            return color

    return ALT_ABOVE[0]


def get_aircraft_name(row):

    callsign = str(row.get("callsign", "")).strip()

    if callsign:
        return callsign

    return "콜사인 없음"


# =========================================================
# 6. OpenSky 경로 데이터 가공
# =========================================================

def create_track_points(track_data):

    path = track_data.get("path") or []

    points = []

    for point in path:

        if len(point) < 3:
            continue

        latitude = point[1]
        longitude = point[2]

        if latitude is None or longitude is None:
            continue

        points.append([
            latitude,
            longitude,
        ])

    return points


# =========================================================
# 7. 항공기 마커 (기종 크기/방향/고도 색상 + 클릭 시 사진·경로·실제 궤적)
# =========================================================

# OpenSky category 값 -> (표시 이름, 아이콘 크기(px), 아이콘 모양)
CATEGORY_INFO = {
    0: ("정보 없음", 20, "plane"),
    1: ("분류 정보 없음", 20, "plane"),
    2: ("경량기 (7t 미만)", 14, "plane"),
    3: ("소형기 (7~34t, 리저널/비즈니스)", 18, "plane"),
    4: ("중형기 (34~136t, A320/B737급)", 23, "plane"),
    5: ("중대형기 (B757급)", 26, "plane"),
    6: ("대형기 (136t 초과, 광동체)", 31, "plane"),
    7: ("고성능기", 20, "plane"),
    8: ("헬리콥터", 20, "heli"),
    9: ("글라이더", 18, "plane"),
    10: ("기구/비행선", 16, "dot"),
    11: ("낙하산", 12, "dot"),
    12: ("초경량 항공기", 14, "plane"),
    14: ("무인기(UAV)", 14, "plane"),
    15: ("우주 비행체", 20, "plane"),
    16: ("지상 차량(긴급)", 10, "dot"),
    17: ("지상 차량(서비스)", 10, "dot"),
    18: ("장애물", 10, "dot"),
    19: ("장애물(군집)", 10, "dot"),
    20: ("장애물(선형)", 10, "dot"),
    21: ("초대형기 (A380급)", 38, "plane"),
}


def build_alt_config():
    """고도 색상 구간을 JS에서도 같은 값으로 쓰도록 JSON으로 만든다."""

    return json.dumps(
        {
            "bands": [[limit, color] for limit, color, _ in ALT_BANDS],
            "above": ALT_ABOVE[0],
            "ground": ALT_GROUND_COLOR,
            "none": ALT_NONE_COLOR,
        }
    )


MARKER_CALLBACK = """
function (row) {
    // row: 0 lat, 1 lon, 2 콜사인, 3 icao24, 4 색상, 5 국가, 6 기압고도, 7 기하고도,
    //      8 속도, 9 상태, 10 진행방향(도), 11 아이콘 크기, 12 분류 이름, 13 아이콘 모양
    var S = row[11];
    var SHAPES = {
        plane: "<path d='M12 1.5 L14 9 L22.5 14.2 L22.5 16.4 L14 13.6 L13.6 19.2 L16.5 21.4 L16.5 22.6 L12 21.4 L7.5 22.6 L7.5 21.4 L10.4 19.2 L10 13.6 L1.5 16.4 L1.5 14.2 L10 9 Z'/>",
        heli: "<circle cx='12' cy='13' r='4.5'/><rect x='11' y='15' width='2' height='8'/><rect x='2' y='11.8' width='20' height='1.6'/>",
        dot: "<circle cx='12' cy='12' r='6'/>"
    };
    var svg = "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' width='" + S + "' height='" + S +
              "' style='display:block; overflow:visible; transform:rotate(" + row[10] + "deg);'>" +
              "<g fill='" + row[4] + "' stroke='#111' stroke-width='1.1' stroke-linejoin='round'>" +
              (SHAPES[row[13]] || SHAPES.plane) + "</g></svg>";
    var icon = L.divIcon({
        className: '',
        html: svg,
        iconSize: [S, S],
        iconAnchor: [S / 2, S / 2],
        popupAnchor: [0, -S / 2]
    });
    var marker = L.marker(new L.LatLng(row[0], row[1]), {icon: icon});

    function buildPopup() { return "<div style='width:240px; font-family:Arial; font-size:13px; line-height:1.4;'>" +
                       "<h4 style='margin:0 0 5px 0; color:#1f77b4;'>✈️ " + row[2] + "</h4>" +
                       "<div class='ac-photo' style='min-height:20px; margin-bottom:6px; color:#888;'>사진 불러오는 중...</div>" +
                       "<hr style='margin:5px 0;'>" +
                       "<b>ICAO24:</b> " + row[3] + "<br>" +
                       "<b>기종 분류:</b> " + row[12] + "<br>" +
                       "<b>국가:</b> " + row[5] + "<br>" +
                       "<b>기종:</b> <span class='ac-type'>불러오는 중...</span><br>" +
                       "<b>출발지:</b> <span class='ac-origin'>불러오는 중...</span><br>" +
                       "<b>도착지:</b> <span class='ac-dest'>불러오는 중...</span><br>" +
                       "<b>기압고도:</b> " + row[6] + "<br>" +
                       "<b>기하고도:</b> " + row[7] + "<br>" +
                       "<b>속도:</b> " + row[8] + "<br>" +
                       "<b>상태:</b> " + row[9] +
                       "<div class='ac-track' style='margin-top:4px; font-size:11px; color:#666;'></div></div>"; }
    marker.bindPopup(buildPopup);
    marker.bindTooltip(row[2] + " (" + row[3] + ")");

    // 팝업을 열 때만 브라우저에서 사진/경로/궤적을 조회한다 (수천 대를 한꺼번에 조회하지 않기 위해)
    marker.on('popupopen', function (e) {
        var root = e.popup.getElement();
        if (!root) { return; }
        var photoEl = root.querySelector('.ac-photo');
        var originEl = root.querySelector('.ac-origin');
        var destEl = root.querySelector('.ac-dest');
        var trackEl = root.querySelector('.ac-track');
        var typeEl = root.querySelector('.ac-type');
        var hex = String(row[3]).toLowerCase();
        var callsign = String(row[2]).trim();
        var map = e.popup._map || e.target._map;
        var cur = e.target.getLatLng();
        var ALT = __ALT_CONFIG__;
        var TOKEN = __TOKEN__;

        window.__acCache = window.__acCache || {photo: {}, route: {}, track: {}};
        var id = window.__acPopupId = (window.__acPopupId || 0) + 1;
        var state = {route: undefined, track: undefined};

        function altColor(m, ground) {
            if (ground) { return ALT.ground; }
            if (m === null || m === undefined || isNaN(m)) { return ALT.none; }
            for (var i = 0; i < ALT.bands.length; i++) {
                if (m < ALT.bands[i][0]) { return ALT.bands[i][1]; }
            }
            return ALT.above;
        }

        function fmtAirport(a) {
            if (!a) { return '정보 없음'; }
            var code = a.iata_code || a.icao_code || '';
            var place = a.municipality || a.name || '';
            var text = (place + (code ? ' (' + code + ')' : '')).trim();
            return text || '정보 없음';
        }

        function hasXY(a) { return a && a.latitude != null && a.longitude != null; }

        // 두 지점 사이의 대권(최단 경로) 좌표열. 경도는 시작점에서 끊기지 않게 이어 붙인다.
        function gcPath(lat1, lon1, lat2, lon2, n) {
            var r = Math.PI / 180;
            var p1 = lat1 * r, l1 = lon1 * r, p2 = lat2 * r, l2 = lon2 * r;
            var d = 2 * Math.asin(Math.sqrt(
                Math.pow(Math.sin((p2 - p1) / 2), 2) +
                Math.cos(p1) * Math.cos(p2) * Math.pow(Math.sin((l2 - l1) / 2), 2)));
            if (!d) { return [[lat1, lon1], [lat2, lon2]]; }
            var pts = [[lat1, lon1]];
            var prev = lon1;
            for (var i = 1; i <= n; i++) {
                var f = i / n;
                var A = Math.sin((1 - f) * d) / Math.sin(d);
                var B = Math.sin(f * d) / Math.sin(d);
                var x = A * Math.cos(p1) * Math.cos(l1) + B * Math.cos(p2) * Math.cos(l2);
                var y = A * Math.cos(p1) * Math.sin(l1) + B * Math.cos(p2) * Math.sin(l2);
                var z = A * Math.sin(p1) + B * Math.sin(p2);
                var lat = Math.atan2(z, Math.sqrt(x * x + y * y)) / r;
                var lon = Math.atan2(y, x) / r;
                lon += 360 * Math.round((prev - lon) / 360);
                pts.push([lat, lon]);
                prev = lon;
            }
            return pts;
        }

        function clearRouteLayer() {
            if (window.__acRouteLayer) {
                window.__acRouteLayer.remove();
                window.__acRouteLayer = null;
            }
        }

        // 지도를 좌우로 넘겨도 선이 보이도록 -360/0/+360 위치에 같은 선을 그린다
        var OFFSETS = [-360, 0, 360];

        function addLine(layer, pts, opts) {
            OFFSETS.forEach(function (o) {
                L.polyline(pts.map(function (p) { return [p[0], p[1] + o]; }), opts).addTo(layer);
            });
        }

        function addDot(layer, pt, color, tip) {
            OFFSETS.forEach(function (o) {
                L.circleMarker([pt[0], pt[1] + o], {radius: 6, color: '#111', weight: 1, fillColor: color, fillOpacity: 1})
                    .bindTooltip(tip).addTo(layer);
            });
        }

        function redraw() {
            if (window.__acPopupId !== id || !map) { return; }
            clearRouteLayer();
            var layer = L.layerGroup();
            var fr = state.route;
            var tr = state.track;
            var o = fr ? fr.origin : null;
            var dst = fr ? fr.destination : null;

            if (tr && tr.length > 1) {
                // 실제로 날아온 궤적: 고도 색상으로 구간별 표시
                for (var i = 1; i < tr.length; i++) {
                    addLine(layer, [tr[i - 1], tr[i]], {
                        color: altColor(tr[i][2], tr[i][3]), weight: 4, opacity: 0.95
                    });
                }
                if (hasXY(o)) {
                    var oLon = o.longitude + 360 * Math.round((tr[0][1] - o.longitude) / 360);
                    // 출발 공항과 궤적 시작점이 멀면 OpenSky가 수신하지 못한 구간 -> 회색 점선으로 이어서 표시
                    if (kmBetween(o.latitude, oLon, tr[0][0], tr[0][1]) > 50) {
                        addLine(layer, gcPath(o.latitude, oLon, tr[0][0], tr[0][1], 32),
                                {color: '#888', weight: 2, opacity: 0.8, dashArray: '2 6'});
                        trackEl.textContent = '실선: 실제 이동 궤적 (고도 색상) · 회색 점선: 수신 공백(추정)';
                    }
                    addDot(layer, [o.latitude, oLon], '#2ca02c', '출발: ' + fmtAirport(o));
                }
            } else if (hasXY(o)) {
                // 궤적을 못 가져오면 출발 공항~현재 위치를 이은 추정선(점선)
                var done = gcPath(cur.lat, cur.lng, o.latitude, o.longitude, 64);
                addLine(layer, done, {color: '#1f77b4', weight: 3, opacity: 0.8, dashArray: '2 6'});
                var oEnd = done[done.length - 1];
                addDot(layer, oEnd, '#2ca02c', '출발: ' + fmtAirport(o));
            }

            // 남은 구간: 현재 위치 -> 도착 공항 (점선)
            if (hasXY(dst)) {
                var left = gcPath(cur.lat, cur.lng, dst.latitude, dst.longitude, 64);
                addLine(layer, left, {color: '#ff7f0e', weight: 3, opacity: 0.9, dashArray: '8 8'});
                var dEnd = left[left.length - 1];
                addDot(layer, dEnd, '#d62728', '도착: ' + fmtAirport(dst));
            }

            layer.addTo(map);
            window.__acRouteLayer = layer;
        }

        // 클러스터가 줌/이동 때마다 화면 밖 마커를 지우면서 팝업이 자동으로 닫히는데,
        // 그때 경로까지 지우면 안 되므로 popupclose가 아니라 '지도 빈 곳 클릭'으로만 지운다.
        if (map) {
            if (window.__acMapClickHandler) { map.off('click', window.__acMapClickHandler); }
            var onMapClick = function () {
                clearRouteLayer();
                map.off('click', onMapClick);
                if (window.__acMapClickHandler === onMapClick) { window.__acMapClickHandler = null; }
            };
            window.__acMapClickHandler = onMapClick;
            map.on('click', onMapClick);
        }

        // ---- 사진 (Planespotters) ----
        function showPhoto(p) {
            photoEl.textContent = '';
            if (!p) { photoEl.textContent = '사진 없음'; return; }
            var link = document.createElement('a');
            link.href = p.link;
            link.target = '_blank';
            link.rel = 'noopener';
            var img = document.createElement('img');
            img.src = (p.thumbnail_large || p.thumbnail).src;
            img.style.cssText = 'width:100%; border-radius:4px; display:block;';
            link.appendChild(img);
            photoEl.appendChild(link);
            var credit = document.createElement('div');
            credit.style.cssText = 'font-size:11px; color:#666; margin-top:2px;';
            credit.textContent = '© ' + (p.photographer || 'Planespotters.net') + ' · Planespotters.net';
            photoEl.appendChild(credit);
        }

        if (hex in window.__acCache.photo) {
            showPhoto(window.__acCache.photo[hex]);
        } else {
            fetch('https://api.planespotters.net/pub/photos/hex/' + encodeURIComponent(hex))
                .then(function (r) { return r.json(); })
                .then(function (d) {
                    var p = (d.photos && d.photos.length) ? d.photos[0] : null;
                    window.__acCache.photo[hex] = p;
                    showPhoto(p);
                })
                .catch(function () { photoEl.textContent = '사진 없음'; });
        }

        // ---- 기종 (adsbdb 항공기 DB) ----
        function showType(ac) {
            if (!ac) { typeEl.textContent = '정보 없음'; return; }
            var name = ((ac.manufacturer || '') + ' ' + (ac.type || '')).trim();
            var code = ac.icao_type ? ' (' + ac.icao_type + ')' : '';
            var reg = ac.registration ? ' · ' + ac.registration : '';
            typeEl.textContent = (name + code + reg).trim() || '정보 없음';
        }

        window.__acCache.type = window.__acCache.type || {};
        if (hex in window.__acCache.type) {
            showType(window.__acCache.type[hex]);
        } else {
            fetch('https://api.adsbdb.com/v0/aircraft/' + encodeURIComponent(hex))
                .then(function (r) { return r.json(); })
                .then(function (d) {
                    var ac = (d.response && d.response.aircraft) ? d.response.aircraft : null;
                    window.__acCache.type[hex] = ac;
                    showType(ac);
                })
                .catch(function () { showType(null); });
        }

        // ---- 출발지/도착지 (adsbdb 콜사인 경로 DB) ----
        function showRoute(fr) {
            state.route = fr || null;
            originEl.textContent = fr ? fmtAirport(fr.origin) : '정보 없음';
            destEl.textContent = fr ? fmtAirport(fr.destination) : '정보 없음';
            redraw();
        }

        if (!callsign || callsign === '콜사인 없음') {
            showRoute(null);
        } else if (callsign in window.__acCache.route) {
            showRoute(window.__acCache.route[callsign]);
        } else {
            fetch('https://api.adsbdb.com/v0/callsign/' + encodeURIComponent(callsign))
                .then(function (r) { return r.json(); })
                .then(function (d) {
                    var fr = (d.response && d.response.flightroute) ? d.response.flightroute : null;
                    window.__acCache.route[callsign] = fr;
                    showRoute(fr);
                })
                .catch(function () { showRoute(null); });
        }

        // ---- 실제 이동 궤적 (OpenSky /tracks/all) ----
        function kmBetween(lat1, lon1, lat2, lon2) {
            var r = Math.PI / 180;
            var dLat = (lat2 - lat1) * r, dLon = (lon2 - lon1) * r;
            var a = Math.pow(Math.sin(dLat / 2), 2) +
                Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.pow(Math.sin(dLon / 2), 2);
            return 12742 * Math.asin(Math.sqrt(a));
        }

        function buildTrack(d) {
            var path = (d && d.path) || [];
            var pts = [];
            var prev = null;
            path.forEach(function (p) {
                if (p[1] == null || p[2] == null) { return; }
                var lon = p[2];
                if (prev !== null) { lon += 360 * Math.round((prev - lon) / 360); }
                prev = lon;
                pts.push([p[1], lon, p[3], p[5]]);
            });
            if (pts.length < 2) { return null; }
            var shift = 360 * Math.round((cur.lng - pts[pts.length - 1][1]) / 360);
            pts.forEach(function (p) { p[1] += shift; });

            // 항공기 위치는 최대 10분 전 스냅샷이라 궤적이 아이콘보다 앞서 있을 수 있다.
            // 아이콘과 가장 가까운 점까지만 사용해서 선이 아이콘 앞으로 튀어나오지 않게 한다.
            var nearest = 0, best = Infinity;
            pts.forEach(function (p, i) {
                var dd = kmBetween(cur.lat, cur.lng, p[0], p[1]);
                if (dd <= best) { best = dd; nearest = i; }
            });
            pts = pts.slice(0, nearest + 1);

            var last = pts[pts.length - 1];
            pts.push([cur.lat, cur.lng, last[2], last[3]]);
            return pts;
        }

        function showTrack(pts, err) {
            state.track = pts || null;
            if (pts) {
                trackEl.textContent = '실선: 실제 이동 궤적 (고도 색상)';
            } else if (err) {
                trackEl.textContent = '궤적 조회 실패 (' + err + ') - 점선은 추정 경로입니다';
            } else {
                trackEl.textContent = '궤적 좌표가 부족합니다 - 점선은 추정 경로입니다';
            }
            redraw();
        }

        if (hex in window.__acCache.track) {
            showTrack(window.__acCache.track[hex]);
        } else {
            fetch('http://127.0.0.1:8765/track?icao24=' + encodeURIComponent(hex))
                .then(function (r) { if (!r.ok) { throw new Error('HTTP ' + r.status); } return r.json(); })
                .then(function (d) {
                    var pts = buildTrack(d);
                    window.__acCache.track[hex] = pts;
                    showTrack(pts);
                })
                .catch(function (e) { showTrack(null, e.message); });
        }
    });
    return marker;
}
"""


# ---------------------------------------------------------
# 기종 분류 보강: OpenSky 항공기 DB(icao24 -> 기종)로 category 빈칸을 채운다
# ---------------------------------------------------------

# OpenSky 항공기 데이터베이스 CSV (https://opensky-network.org/datasets/metadata/ 에서 받은 파일)
AIRCRAFT_DB_PATH = ROOT_DIR / "data" / "raw" / "aircraft_database.csv"
AIRCRAFT_DB_CACHE = ROOT_DIR / "data" / "raw" / "aircraft_category_cache.pkl"

# 지도 좌우 반복 표시용 마커 복제 위치. 속도가 더 중요하면 () 로 바꾸면 데이터가 1/3로 줄어든다
WRAP_COPIES = (-360, 360)

_TYPE_SUPER = re.compile(r"^(A388|A225)$")
_TYPE_HEAVY = re.compile(
    r"^(B74\w|B77\w|B78\w|B76\w|A30\w|A310|A33\w|A34\w|A35\w|MD11|DC10|L101|"
    r"IL96|IL86|IL76|A124|C5M?|C17|B52|K35R|KC10)$"
)
_TYPE_MID_HEAVY = re.compile(r"^(B75\w)$")
_TYPE_MEDIUM = re.compile(
    r"^(B73\w|B37\w|B38\w|B39\w|A31[89]|A19N|A20N|A21N|A32\w|A22\w|BCS\d|B71\w|"
    r"B72\w|B46\w|MD8\w|MD9\w|DC9\w?|F100|F70|E17\w|E19\w|E29\w|CRJ[79X]|C919|SU95|RJ\w\w)$"
)
_TYPE_SMALL = re.compile(
    r"^(CRJ[12]|E13\w|E14\w|E45X|AT\w\w|DH8\w|SF34|SB20|F50|JS\d\d|D328|B190|"
    r"GLF\w|GL\w\w|CL\d\d|FA\d\w|E55P|C68A|C750|H25\w|C56X|PC24)$"
)


def classify_aircraft_type(typecode, icao_type, description):
    """DB의 기종 정보를 OpenSky category 코드(2~9, 21=초대형)로 바꾼다. 모르면 None."""

    typecode = (typecode or "").strip().upper()
    icao_type = (icao_type or "").strip().upper()
    description = (description or "").strip().lower()

    if typecode:
        if _TYPE_SUPER.match(typecode):
            return 21
        if _TYPE_HEAVY.match(typecode):
            return 6
        if _TYPE_MID_HEAVY.match(typecode):
            return 5
        if _TYPE_MEDIUM.match(typecode):
            return 4
        if _TYPE_SMALL.match(typecode):
            return 3

    if description:
        for keyword, code in (
            ("high vortex", 5),
            ("heavy", 6),
            ("large", 4),
            ("small", 3),
            ("light", 2),
            ("rotorcraft", 8),
            ("glider", 9),
            ("lighter", 10),
            ("parachut", 11),
            ("ultralight", 12),
            ("unmanned", 14),
            ("uav", 14),
            ("space", 15),
            ("high performance", 7),
        ):
            if keyword in description:
                return code

    # ICAO 기종 분류 코드 예) L2J = 육상기/엔진 2개/제트, H1T = 헬리콥터
    if len(icao_type) == 3:
        if icao_type[0] in ("H", "G"):
            return 8
        if icao_type[2] == "P":
            return 2
        if icao_type[2] == "T":
            return 2 if icao_type[1] == "1" else 3
        if icao_type[2] == "J":
            return 3

    return None


@st.cache_resource(show_spinner=False)
def load_aircraft_categories():
    """icao24 -> category 코드 딕셔너리. DB 파일이 없으면 빈 딕셔너리."""

    if not AIRCRAFT_DB_PATH.exists():
        return {}

    if (
        AIRCRAFT_DB_CACHE.exists()
        and AIRCRAFT_DB_CACHE.stat().st_mtime >= AIRCRAFT_DB_PATH.stat().st_mtime
    ):
        try:
            return joblib.load(AIRCRAFT_DB_CACHE)
        except Exception:
            pass

    with open(AIRCRAFT_DB_PATH, encoding="utf-8", errors="replace") as f:
        first_line = f.readline()

    # OpenSky CSV는 작은따옴표로 감싼 파일도 있고 큰따옴표인 파일도 있다
    quotechar = "'" if first_line.startswith("'") else '"'
    wanted = {"icao24", "typecode", "icaoaircrafttype", "categoryDescription"}

    db = pd.read_csv(
        AIRCRAFT_DB_PATH,
        quotechar=quotechar,
        dtype=str,
        usecols=lambda c: c.strip("'\"") in wanted,
        on_bad_lines="skip",
        low_memory=False,
    )
    db.columns = [c.strip("'\"") for c in db.columns]
    db = db.fillna("")

    for column in wanted:
        if column not in db.columns:
            db[column] = ""

    result = {}

    for hex_code, typecode, icao_type, description in zip(
        db["icao24"],
        db["typecode"],
        db["icaoaircrafttype"],
        db["categoryDescription"],
    ):

        code = classify_aircraft_type(typecode, icao_type, description)

        if code is not None and hex_code:
            result[hex_code.strip().lower()] = code

    try:
        joblib.dump(result, AIRCRAFT_DB_CACHE)
    except Exception:
        pass

    return result


@st.cache_data(ttl=600, show_spinner=False)
def build_marker_rows(df):
    """지도 마커용 데이터를 만든다. 결과는 데이터와 같은 주기(10분)로 캐시한다."""

    category_db = load_aircraft_categories()

    rows = []

    for (
        lat,
        lon,
        callsign_raw,
        icao24_raw,
        country_raw,
        baro_alt,
        geo_alt,
        velocity,
        on_ground,
        heading,
        category,
    ) in zip(
        df["latitude"].tolist(),
        df["longitude"].tolist(),
        df["callsign"].tolist(),
        df["icao24"].tolist(),
        df["origin_country"].tolist(),
        df["baro_altitude"].tolist(),
        df["geo_altitude"].tolist(),
        df["velocity"].tolist(),
        df["on_ground"].tolist(),
        df["true_track"].tolist(),
        df["category"].tolist(),
    ):

        on_ground = bool(on_ground)

        altitude_for_color = baro_alt if not pd.isna(baro_alt) else geo_alt
        color = get_altitude_color(altitude_for_color, on_ground)

        callsign = escape(str(callsign_raw).strip() or "콜사인 없음")
        icao24 = escape(str(icao24_raw))
        country = escape(str(country_raw or ""))

        heading = 0.0 if pd.isna(heading) else round(float(heading), 1)

        live_category = 0 if pd.isna(category) else int(category)
        db_category = category_db.get(str(icao24_raw).lower())

        if live_category in (0, 1) and db_category is not None:
            final_category = db_category
        elif live_category == 6 and db_category == 21:
            final_category = 21
        else:
            final_category = live_category

        category_label, icon_size, icon_shape = CATEGORY_INFO.get(
            final_category, ("정보 없음", 20, "plane")
        )

        base_item = [
            float(lat),
            float(lon),
            callsign,
            icao24,
            color,
            country,
            format_altitude(baro_alt),
            format_altitude(geo_alt),
            format_speed(velocity),
            "지상" if on_ground else "비행 중",
            heading,
            icon_size,
            category_label,
            icon_shape,
        ]

        rows.append(base_item)

        # 좌우 무한 스크롤 시에도 비행기가 보이도록 -360도, +360도 위치에 마커 복제 추가
        for offset in WRAP_COPIES:
            wrapped_item = base_item.copy()
            wrapped_item[1] = float(lon) + offset
            rows.append(wrapped_item)

    return rows


def add_aircraft_markers(aircraft_map, df, token=""):

    callback = (
        MARKER_CALLBACK
        .replace("__ALT_CONFIG__", build_alt_config())
        .replace("__TOKEN__", json.dumps(token or ""))
    )

    data = build_marker_rows(df)

    # 클러스터는 낮은 줌에서만 쓰고, 확대하면 모든 항공기가 개별 아이콘으로 보이게 한다
    FastMarkerCluster(
        data=data,
        callback=callback,
        options={
            "maxClusterRadius": 35,
            "disableClusteringAtZoom": 6,
        },
    ).add_to(aircraft_map)


# =========================================================
# 8. 이동경로 지도 표시
# =========================================================

def add_aircraft_track(aircraft_map, track_data):

    if not track_data:
        return False

    points = create_track_points(track_data)

    if len(points) < 2:
        return False

    folium.PolyLine(
        locations=points,
        color="blue",
        weight=4,
        opacity=0.85,
        tooltip="항공기 이동경로",
    ).add_to(aircraft_map)

    folium.Marker(
        location=points[0],
        tooltip="경로 시작 지점",
        icon=folium.Icon(
            color="green",
            icon="play",
        ),
    ).add_to(aircraft_map)

    folium.Marker(
        location=points[-1],
        tooltip="마지막 경로 지점",
        icon=folium.Icon(
            color="red",
            icon="flag",
        ),
    ).add_to(aircraft_map)

    return True


# =========================================================
# 9. 실시간 항공기 지도
# =========================================================

def show_live_map():

    st.subheader("🌍 실시간 항공기 위치")

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("🔄 데이터 새로고침"):
            load_aircraft_data.clear()
            load_aircraft_track.clear()
            get_access_token.clear()
            st.rerun()

    try:

        with st.spinner("항공기 데이터를 불러오는 중입니다..."):

            api_timestamp, df = load_aircraft_data()

    except requests.exceptions.HTTPError as exc:

        st.error(f"OpenSky API 오류: {exc}")
        return

    except requests.exceptions.RequestException as exc:

        st.error(f"OpenSky 연결 오류: {exc}")
        return

    except Exception as exc:

        st.error(f"항공기 데이터를 불러오지 못했습니다: {exc}")
        return

    if df.empty:

        st.warning("현재 표시할 항공기 데이터가 없습니다.")
        return

    if api_timestamp:

        data_time = datetime.fromtimestamp(
            api_timestamp
        ).astimezone()

        data_time_text = data_time.strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        )

        st.caption(
            f"OpenSky 데이터 기준 시각: {data_time_text}"
        )

        stale_minutes = (time.time() - api_timestamp) / 60

        if stale_minutes > 20:

            st.warning(
                f"OpenSky 응답이 느려 {stale_minutes:.0f}분 전에 받아 둔 데이터를 보여주고 있습니다. "
                "잠시 뒤 '데이터 새로고침'을 눌러주세요."
            )

    total_aircraft = len(df)

    airborne = int(
        (df["on_ground"] == False).sum()
    )

    on_ground = int(
        (df["on_ground"] == True).sum()
    )

    stat_col1, stat_col2, stat_col3 = st.columns(3)

    stat_col1.metric(
        "위치 확인 항공기",
        f"{total_aircraft:,}대",
    )

    stat_col2.metric(
        "비행 중",
        f"{airborne:,}대",
    )

    stat_col3.metric(
        "지상",
        f"{on_ground:,}대",
    )

    st.divider()

    center_lat = float(
        df["latitude"].median()
    )

    center_lon = float(
        df["longitude"].median()
    )

    # 위아래(위도)는 -85 ~ 85로 고정하고, 좌우(경도)는 사실상 제한 없이 무한 반복되도록 설정합니다.
    # - folium의 max_bounds=True는 min_lon/max_lon 기본값이 -180/180이라 좌우가 막히므로
    #   경도 범위를 매우 크게 넓혀 줍니다.
    # - world_copy_jump=True: 세계 지도를 좌우로 넘기면 원본 지도로 자연스럽게 점프하여
    #   마커/경로가 계속 보이게 합니다.
    aircraft_map = folium.Map(
        location=[
            center_lat,
            center_lon,
        ],
        zoom_start=3,
        min_zoom=3,
        tiles="OpenStreetMap",
        control_scale=False,
        prefer_canvas=True,
        world_copy_jump=True,
        min_lat=-85,
        max_lat=85,
        min_lon=-100000,
        max_lon=100000,
        max_bounds=True,
        max_bounds_viscosity=1.0,  # 위아래 경계 밖으로 벗어나는 것을 강력하게 차단 (회색 화면 방지)
    )

    Fullscreen(
        position="topright",
        title="전체 화면",
        title_cancel="전체 화면 종료",
    ).add_to(aircraft_map)

    legend_rows = [
        (color, label) for _, color, label in ALT_BANDS
    ] + [
        (ALT_ABOVE[0], ALT_ABOVE[1]),
        (ALT_GROUND_COLOR, "지상"),
        (ALT_NONE_COLOR, "고도 정보 없음"),
    ]

    legend_items = "".join(
        f'<span style="color:{color}; text-shadow:0 0 1px #000;">●</span> {label}<br>'
        for color, label in legend_rows
    )

    legend_html = f"""
    <div style="
        position:fixed;
        bottom:25px;
        left:25px;
        z-index:9999;
        background:white;
        padding:12px;
        border:1px solid #ccc;
        border-radius:8px;
        font-size:13px;
        line-height:1.5;
    ">
        <b>항공기 고도</b><br>
        {legend_items}
        <div style="margin-top:6px; color:#666; font-size:12px;">
            아이콘 크기 = 기종 분류, 방향 = 진행 방향
        </div>
    </div>
    """

    aircraft_map.get_root().html.add_child(
        folium.Element(legend_html)
    )

    # 실제 이동 궤적(OpenSky tracks)을 브라우저에서 조회할 때 쓸 토큰.
    # 로컬에서 쓰는 앱이라 지도 HTML에 포함해도 되지만, 외부에 배포한다면 별도 프록시가 필요하다.
    try:
        token = get_opensky_headers()["Authorization"].replace("Bearer ", "")
    except Exception:
        token = ""

    start_track_proxy(
        get_secret("OPENSKY_CLIENT_ID"),
        get_secret("OPENSKY_CLIENT_SECRET"),
    )

    add_aircraft_markers(
        aircraft_map,
        df,
        token=token,
    )

    st_folium(
        aircraft_map,
        width=None,
        height=650,
        key="aircraft_live_map",
        returned_objects=[],  # 지도를 움직일 때마다 Streamlit이 전체 재실행되며 지도가 다시 그려지는 것을 방지
    )

    st.info("💡 지도 위의 비행기 아이콘을 클릭하시면 팝업창에서 상세 정보를 바로 확인하실 수 있습니다.")

    with st.expander("📋 항공기 데이터 전체 보기"):

        display_columns = [
            "icao24",
            "callsign",
            "origin_country",
            "latitude",
            "longitude",
            "baro_altitude",
            "geo_altitude",
            "velocity",
            "true_track",
            "on_ground",
        ]

        st.dataframe(
            df[display_columns],
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# 10. 모델 불러오기
# =========================================================

@st.cache_resource(show_spinner=False, max_entries=8)
def load_model_bundle(model_path, model_mtime=0.0):
    """model_mtime이 인자에 들어 있어서, 모델 파일이 바뀌면 캐시가 자동으로 새로 만들어진다."""

    path = Path(model_path)

    if not path.exists():
        return None

    return joblib.load(path)


def get_model_bundle(model_path):
    """터미널에서 train.py를 돌려 모델 파일이 바뀐 경우에도 화면에 바로 반영되게 한다."""

    path = Path(model_path)
    mtime = path.stat().st_mtime if path.exists() else 0.0

    return load_model_bundle(str(path), mtime)


def load_trainer():
    """models/train.py를 모듈로 불러온다. (재학습 버튼, 실시간 입력 계산에 사용)"""

    spec = importlib.util.spec_from_file_location(
        "aircraft_trainer",
        TRAIN_SCRIPT,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@st.cache_data(ttl=60, show_spinner=False)
def load_live_input(csv_path, csv_mtime, feature_columns):
    """CSV의 가장 최근 시점으로 예측 입력 1행을 만든다. (csv_mtime은 캐시 갱신용)"""

    trainer = load_trainer()

    return trainer.build_latest_features(
        csv_path,
        list(feature_columns),
    )


def render_retrain_controls(dataset_key):
    """실시간(current) 모델을 최신 CSV로 다시 학습하는 버튼."""

    message = st.session_state.pop("retrain_message", None)

    if message:
        st.success(message)

    bundle = get_model_bundle(
        MODEL_DIR / dataset_key / "demand_model.pkl"
    )

    col_button, col_info = st.columns([1, 3])

    with col_info:

        if bundle is not None and bundle.get("trained_at"):

            trained_at = pd.Timestamp(bundle["trained_at"]).tz_convert("Asia/Seoul")

            st.caption(
                f"마지막 학습: {trained_at:%Y-%m-%d %H:%M} (KST) · "
                f"학습 샘플 {bundle.get('n_samples', '?')}개"
            )

        elif bundle is None:

            st.caption("아직 학습된 모델이 없습니다. 버튼을 눌러 학습해주세요.")

        if LIVE_DATA_PATH.exists():

            modified = datetime.fromtimestamp(
                LIVE_DATA_PATH.stat().st_mtime
            )

            st.caption(f"수집 데이터 마지막 갱신: {modified:%Y-%m-%d %H:%M}")

    with col_button:

        clicked = st.button(
            "🔄 최신 데이터로 재학습",
            key=f"retrain_{dataset_key}",
            help="data_collector.py가 모은 최신 CSV로 모델을 다시 학습합니다.",
        )

    if not clicked:
        return

    with st.spinner("재학습 중입니다. 잠시만 기다려주세요..."):

        try:

            trainer = load_trainer()

            result = trainer.train_dataset(
                dataset_key,
                trainer.DATASETS[dataset_key],
            )

        except Exception as exc:

            st.error(f"재학습에 실패했습니다: {exc}")

            return

    # 캐시에 남아 있는 예전 모델/입력값을 비워야 새 결과가 화면에 반영된다
    load_model_bundle.clear()
    load_live_input.clear()

    st.session_state["retrain_message"] = (
        f"재학습이 끝났습니다. 최고 모델: {result['best_model']} "
        f"(학습 샘플 {result['n_samples']:,}개)"
    )

    st.rerun()


# =========================================================
# 11. 항공 교통량 예측
# =========================================================

# ---------------------------------------------------------
# 그래프 / 여러 시점 예측 도우미 (화면에 보이는 시각은 모두 UTC)
# ---------------------------------------------------------

THIN_STEPS = [
    "5min", "10min", "15min", "30min", "1h", "2h", "3h", "6h", "12h", "1D", "7D", "30D",
]

WINDOW_OPTIONS = {
    "6시간": pd.Timedelta(hours=6),
    "1일": pd.Timedelta(days=1),
    "7일": pd.Timedelta(days=7),
    "30일": pd.Timedelta(days=30),
    "1년": pd.Timedelta(days=365),
    "전체": None,
}

# Streamlit 1.37 이상이면 fragment로 감싸서, 구간을 바꿀 때 그래프 부분만 다시 그린다.
# (지도 탭 전체가 다시 실행되지 않는다. 이전 버전이면 그냥 일반 함수로 동작)
st_fragment = getattr(st, "fragment", None) or (lambda func: func)


def fmt_utc(ts):
    """시각을 'YYYY-MM-DD HH:MM' (UTC) 문자열로. tz가 없으면 UTC로 본다."""

    ts = pd.Timestamp(ts)

    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")

    return ts.strftime("%Y-%m-%d %H:%M")


def horizon_label(minutes):
    if minutes % 60 == 0:
        return f"{minutes // 60}시간 뒤"

    return f"{minutes}분 뒤"


@st.cache_resource(show_spinner=False, max_entries=4)
def load_counts_csv(csv_path, csv_mtime):
    """timestamp / aircraft_count 두 컬럼만 읽는다. 시각은 UTC (tz 없음). csv_mtime은 캐시 갱신용"""

    df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        usecols=["timestamp", "aircraft_count"],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    df["aircraft_count"] = pd.to_numeric(df["aircraft_count"], errors="coerce")

    return (
        df.dropna()
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def thin_for_chart(df, time_col, value_cols, start=None, end=None, max_points=1500):
    """보려는 구간[start, end]만 잘라서, 점이 너무 많으면 시간 단위로 평균낸다.

    - 구간이 짧을수록 더 촘촘하게(최대 5분 원본) 보여준다.
    - 데이터가 없는 시간대는 NaN이 되어 선이 이어지지 않고 끊겨 보인다.
    - 반환: (그래프용 DataFrame, 사용한 간격 문자열). 시각은 UTC (tz 없음)
    """

    data = df[[time_col] + list(value_cols)].copy()
    data[time_col] = pd.to_datetime(data[time_col], utc=True, errors="coerce")
    data = data.dropna(subset=[time_col]).set_index(time_col).sort_index()
    data.index = data.index.tz_convert(None)

    if start is not None or end is not None:
        data = data.loc[start:end]

    if data.empty:
        return data.reset_index(), None

    low = start if start is not None else data.index[0]
    high = end if end is not None else data.index[-1]
    span = max(high - low, pd.Timedelta(minutes=5))

    freq = THIN_STEPS[-1]

    for candidate in THIN_STEPS:
        if span / pd.Timedelta(candidate) <= max_points:
            freq = candidate
            break

    return data.resample(freq).mean().reset_index(), freq


def pick_window(data_min, data_max, key, default_label, pad=pd.Timedelta(0)):
    """보기 구간(6시간~전체)과 좌우 이동 슬라이더를 보여주고 (시작, 끝) UTC를 돌려준다.

    pad: 가장 최근 끝에 미래 예측 구간만큼 여유를 더한다.
    """

    labels = list(WINDOW_OPTIONS)

    label = st.radio(
        "보기 구간",
        labels,
        index=labels.index(default_label),
        horizontal=True,
        key=f"{key}_window",
    )

    window = WINDOW_OPTIONS[label]
    bound_end = data_max + pad

    if window is None or bound_end - data_min <= window:
        return data_min, bound_end

    shift = window / 2
    max_offset = int(np.ceil((bound_end - data_min - window) / shift))
    offset = 0

    if max_offset > 0:

        # 범위가 바뀌면(새 데이터가 쌓이면) 슬라이더를 새로 만들어 가장 최근으로 돌아오게 한다.
        value = st.slider(
            "구간 이동 (오른쪽 끝 = 가장 최근, 왼쪽 = 과거)",
            min_value=-max_offset,
            max_value=0,
            value=0,
            key=f"{key}_offset_{label}_{max_offset}",
        )

        offset = -value

    end = bound_end - shift * offset
    start = end - window

    if start < data_min:
        start = data_min
        end = start + window

    return start, end


# ---------------------------------------------------------
# 실시간: 여러 시점 미래 예측
# ---------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def compute_live_forecast(csv_path, csv_mtime, model_path, model_mtime, feature_columns):
    """저장된 5분 예측 모델로 (1) 지난 구간의 5분 뒤 예측, (2) 앞으로 최대 3시간 예측 경로를 만든다.

    미래 예측은 5분 모델을 반복 적용한다. 예측값을 다음 입력(lag, 이동평균 등)으로 넣고,
    날씨·휴일 같은 추가 피처는 마지막 관측값을 그대로 유지한다.
    train.py의 make_features를 그대로 써서 학습 때와 같은 방식으로 피처를 만든다.
    """

    bundle = load_model_bundle(model_path, model_mtime)

    if bundle is None or bundle.get("target_mode") != "delta":
        return {
            "status": "error",
            "message": "이전 방식 모델이라 예측할 수 없습니다. '최신 데이터로 재학습'을 눌러주세요.",
        }

    model = bundle["model"]
    cols = list(feature_columns)

    trainer = load_trainer()
    interval = pd.Timedelta(minutes=trainer.INTERVAL_MINUTES)

    df = trainer.load_and_prepare_data(csv_path, min_rows=1)
    data = trainer.make_features(df)

    missing = [c for c in cols if c not in data.columns]

    if missing:
        return {
            "status": "error",
            "message": (
                f"모델이 쓰는 피처({', '.join(missing)})가 현재 CSV에 없습니다. "
                "재학습해주세요."
            ),
        }

    last_segment = df[df["segment"] == df["segment"].iloc[-1]]

    result = {
        "status": "short",
        "segment_len": int(len(last_segment)),
        "last_ts": df["timestamp"].iloc[-1].tz_convert(None),
        "current": float(df["aircraft_count"].iloc[-1]),
        "past": None,
        "path": None,
    }

    # (1) 지난 구간: 각 시점에서 '5분 뒤'를 예측한 값 (예측값이 놓일 시각 = 시점 + 5분)
    features = data[cols]
    usable = features.notna().all(axis=1)

    if usable.any():

        delta = model.predict(features[usable])

        result["past"] = pd.DataFrame(
            {
                "timestamp": (data.loc[usable, "timestamp"] + interval).dt.tz_convert(None),
                "pred": np.maximum(data.loc[usable, "lag_0"].to_numpy() + delta, 0),
            }
        )

    if not bool(usable.iloc[-1]):
        return result

    # (2) 미래: 5분씩 반복 예측
    max_steps = int(max(HORIZON_MINUTES) // trainer.INTERVAL_MINUTES)
    work = last_segment.tail(40).reset_index(drop=True)
    rows = []

    for step_number in range(1, max_steps + 1):

        feats = trainer.make_features(work)[cols].iloc[[-1]]

        if feats.isna().to_numpy().any():
            break

        current = float(work["aircraft_count"].iloc[-1])
        predicted = max(0.0, current + float(model.predict(feats)[0]))

        new_row = work.iloc[[-1]].copy()
        new_row["timestamp"] = work["timestamp"].iloc[-1] + interval
        new_row["aircraft_count"] = predicted
        work = pd.concat([work, new_row], ignore_index=True)

        rows.append(
            {
                "minutes": step_number * trainer.INTERVAL_MINUTES,
                "timestamp": new_row["timestamp"].iloc[0].tz_convert(None),
                "pred": predicted,
            }
        )

    if rows:
        result["status"] = "ok"
        result["path"] = pd.DataFrame(rows)

    return result


def render_horizon_metrics(forecast, bundle):
    """5분 / 10분 / 30분 / 1시간 / 3시간 뒤 예측값을 카드로 보여준다."""

    path = forecast["path"]
    current = forecast["current"]

    columns = st.columns(len(HORIZON_MINUTES))

    for column, minutes in zip(columns, HORIZON_MINUTES):

        row = path[path["minutes"] == minutes]

        if row.empty:
            column.metric(horizon_label(minutes), "-")
            continue

        value = round(float(row["pred"].iloc[0]))

        column.metric(
            horizon_label(minutes),
            f"{value:,}대",
            delta=f"{value - round(current):+,}대",
        )

    note = (
        f"{fmt_utc(forecast['last_ts'])} UTC 관측값 {round(current):,}대 기준이고, 증감은 이 값 대비입니다. "
        "10분 이후는 5분 예측 모델을 반복 적용한 추정이라 (날씨·휴일은 마지막 값 유지) "
        "멀수록 오차가 커집니다."
    )

    n_samples = bundle.get("n_samples")

    if n_samples is not None and n_samples < RECOMMENDED_SAMPLES:
        note += f" 학습 샘플이 {n_samples:,}개뿐이라 1시간 이상 뒤 예측은 참고용으로만 보세요."

    st.caption(note)


def build_live_figure(history, forecast, start, end):
    """최근 실제값 + 지난 5분 뒤 예측 + 앞으로의 예측 경로"""

    fig = go.Figure()

    actual, freq = thin_for_chart(history, "timestamp", ["aircraft_count"], start, end)

    fig.add_trace(
        go.Scatter(
            x=actual["timestamp"],
            y=actual["aircraft_count"],
            mode="lines",
            name="실제 항공기 수",
            line=dict(color="#1f77b4"),
            connectgaps=False,
        )
    )

    past = forecast.get("past") if forecast else None

    if past is not None and not past.empty:

        past_view, _ = thin_for_chart(past, "timestamp", ["pred"], start, end)

        fig.add_trace(
            go.Scatter(
                x=past_view["timestamp"],
                y=past_view["pred"],
                mode="lines",
                name="5분 뒤 예측 (저장된 모델)",
                line=dict(color="#ff7f0e", dash="dot"),
                connectgaps=False,
            )
        )

    path = forecast.get("path") if forecast else None

    if path is not None and not path.empty:

        origin = pd.DataFrame(
            {"timestamp": [forecast["last_ts"]], "pred": [forecast["current"]]}
        )
        line = pd.concat([origin, path[["timestamp", "pred"]]], ignore_index=True)

        fig.add_trace(
            go.Scatter(
                x=line["timestamp"],
                y=line["pred"],
                mode="lines",
                name="미래 예측 경로",
                line=dict(color="#d62728", dash="dash"),
            )
        )

        marks = path[path["minutes"].isin(HORIZON_MINUTES)]

        fig.add_trace(
            go.Scatter(
                x=marks["timestamp"],
                y=marks["pred"],
                mode="markers+text",
                text=[horizon_label(int(m)) for m in marks["minutes"]],
                textposition="top center",
                name="예측 시점",
                marker=dict(color="#d62728", size=9),
            )
        )

    fig.update_layout(
        title="실시간 항공기 수: 수집 데이터와 미래 예측",
        hovermode="x unified",
        xaxis_title="시각 (UTC)",
        yaxis_title="항공기 수",
        legend_title_text="",
    )
    fig.update_xaxes(range=[start, end])

    return fig, freq


@st_fragment
def render_live_chart_panel(history, forecast):

    has_path = bool(forecast) and forecast.get("path") is not None
    pad = pd.Timedelta(minutes=max(HORIZON_MINUTES)) if has_path else pd.Timedelta(0)

    start, end = pick_window(
        history["timestamp"].iloc[0],
        history["timestamp"].iloc[-1],
        "live",
        "6시간",
        pad,
    )

    fig, freq = build_live_figure(history, forecast, start, end)

    st.plotly_chart(fig, use_container_width=True)

    note = (
        f"수집 {len(history):,}개 · {fmt_utc(history['timestamp'].iloc[0])} ~ "
        f"{fmt_utc(history['timestamp'].iloc[-1])} (UTC) · 선이 끊긴 곳은 수집이 멈춘 시간입니다."
    )

    if freq not in (None, "5min"):
        note += f" 점이 많아 {freq} 평균으로 표시합니다."

    st.caption(note)


def render_live_section(model_path, bundle, feature_columns, is_legacy):
    """실시간 탭: 여러 시점 미래 예측 카드 + 최근 수집 데이터 그래프"""

    if not LIVE_DATA_PATH.exists():
        return

    csv_mtime = LIVE_DATA_PATH.stat().st_mtime

    try:
        history = load_counts_csv(str(LIVE_DATA_PATH), csv_mtime)
    except Exception as exc:
        st.warning(f"수집 데이터를 불러오지 못했습니다: {exc}")
        return

    if history.empty:
        return

    forecast = None

    if not is_legacy:

        path = Path(model_path)

        try:
            forecast = compute_live_forecast(
                str(LIVE_DATA_PATH),
                csv_mtime,
                str(path),
                path.stat().st_mtime if path.exists() else 0.0,
                tuple(feature_columns),
            )
        except Exception as exc:
            st.warning(f"미래 예측을 계산하지 못했습니다: {exc}")

    status = forecast.get("status") if forecast else None

    if status == "error":
        st.warning(forecast["message"])

    elif status == "ok":
        render_horizon_metrics(forecast, bundle)

    elif status == "short":
        st.caption(
            f"현재 연속 수집 {forecast['segment_len']}개 · "
            "미래 예측에는 연속 13개(약 1시간)가 필요합니다."
        )

    render_live_chart_panel(
        history,
        forecast if status in ("ok", "short") else None,
    )


# ---------------------------------------------------------
# 테스트 구간 / 과거 데이터 그래프
# ---------------------------------------------------------

LABELS = {"actual": "실제", "predicted": "예측", "baseline": "직전값(기준선)"}


def render_test_chart(test_predictions):
    """학습에 쓰지 않은 맨 뒤 구간의 실제 vs 예측 (실시간 탭에서는 접어서 보여줌)"""

    value_cols = [c for c in LABELS if c in test_predictions.columns]

    chart_df, freq = thin_for_chart(test_predictions, "timestamp", value_cols)
    plot_df = chart_df.rename(columns=LABELS)

    fig = px.line(
        plot_df,
        x="timestamp",
        y=[LABELS[c] for c in value_cols],
        title="테스트 구간: 실제 vs 예측 항공기 수",
        labels={"timestamp": "시각 (UTC)", "value": "항공기 수", "variable": ""},
    )
    fig.update_layout(hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True)

    note = (
        f"학습 샘플의 맨 뒤 일부(기본 15%)만 보여줍니다 ({len(test_predictions):,}개, "
        f"{fmt_utc(test_predictions['timestamp'].iloc[0])} ~ "
        f"{fmt_utc(test_predictions['timestamp'].iloc[-1])} UTC)."
    )

    if freq not in (None, "5min"):
        note += f" 점이 많아 {freq} 평균으로 표시합니다."

    st.caption(note)


def build_historical_figure(counts, test_predictions, start, end):

    fig = go.Figure()
    freq = None

    if counts is not None:

        actual, freq = thin_for_chart(counts, "timestamp", ["aircraft_count"], start, end)
        actual = actual.rename(columns={"aircraft_count": "actual"})

    else:

        actual, freq = thin_for_chart(test_predictions, "timestamp", ["actual"], start, end)

    fig.add_trace(
        go.Scatter(
            x=actual["timestamp"],
            y=actual["actual"],
            mode="lines",
            name="실제",
            line=dict(color="#1f77b4"),
            connectgaps=False,
        )
    )

    colors = {"predicted": "#ff7f0e", "baseline": "#d62728"}

    for column in ["predicted", "baseline"]:

        if column not in test_predictions.columns:
            continue

        view, _ = thin_for_chart(test_predictions, "timestamp", [column], start, end)

        fig.add_trace(
            go.Scatter(
                x=view["timestamp"],
                y=view[column],
                mode="lines",
                name=LABELS[column],
                line=dict(color=colors[column]),
                connectgaps=False,
            )
        )

    fig.update_layout(
        title="과거 데이터: 실제 항공기 수와 테스트 구간 예측",
        hovermode="x unified",
        xaxis_title="시각 (UTC)",
        yaxis_title="항공기 수",
        legend_title_text="",
    )
    fig.update_xaxes(range=[start, end])

    return fig, freq


@st_fragment
def render_historical_panel(test_predictions):
    """CSV 전체 기간의 실제값 위에, 테스트 구간 예측을 겹쳐서 보여준다."""

    counts = None

    if HISTORICAL_DATA_PATH.exists():

        try:
            counts = load_counts_csv(
                str(HISTORICAL_DATA_PATH),
                HISTORICAL_DATA_PATH.stat().st_mtime,
            )
        except Exception as exc:
            st.warning(f"과거 CSV를 불러오지 못해 테스트 구간만 표시합니다: {exc}")

    if counts is not None and counts.empty:
        counts = None

    test_times = pd.to_datetime(
        test_predictions["timestamp"], utc=True
    ).dt.tz_convert(None)

    if counts is not None:
        data_min, data_max = counts["timestamp"].iloc[0], counts["timestamp"].iloc[-1]
    else:
        data_min, data_max = test_times.min(), test_times.max()

    # 기본은 가장 최근 30일(예측이 있는 구간). '전체'나 '1년'으로 넓히고 슬라이더로 과거까지 옮겨 볼 수 있다.
    start, end = pick_window(data_min, data_max, "hist", "30일")

    fig, freq = build_historical_figure(counts, test_predictions, start, end)

    st.plotly_chart(fig, use_container_width=True)

    note = (
        f"실제값: {fmt_utc(data_min)} ~ {fmt_utc(data_max)} (UTC) 전체 기간. "
        f"예측값은 학습에 쓰지 않은 테스트 구간({fmt_utc(test_times.min())} ~ "
        f"{fmt_utc(test_times.max())} UTC)에만 있습니다."
    )

    if freq not in (None, "5min"):
        note += f" 점이 많아 {freq} 평균으로 표시합니다. 구간을 줄이면 더 촘촘하게 보입니다."

    st.caption(note)


def show_model_prediction(dataset_key):

    dataset = next(
        d for d in DATASETS.values() if d["key"] == dataset_key
    )

    st.subheader("📈 항공 교통량 예측")

    selected_model_name = st.selectbox(
        "예측 모델 선택",
        list(MODEL_SLUGS.keys()),
        key=f"model_select_{dataset_key}",
    )

    model_path = get_model_path(dataset_key, selected_model_name)

    bundle = get_model_bundle(model_path)

    if bundle is None:

        st.warning(
            f"모델 파일이 없습니다.\n\n{model_path}\n\n"
            "먼저 models/train.py를 실행해주세요."
        )

        return

    model = bundle.get("model")

    feature_columns = bundle.get(
        "feature_columns",
        [],
    )

    latest_features = bundle.get(
        "latest_features",
        None,
    )

    interval_minutes = bundle.get(
        "interval_minutes",
        5,
    )

    if model is None or not feature_columns:

        st.error(
            "모델 파일의 데이터 구조를 확인해주세요."
        )

        return

    first_ts = bundle.get("first_data_timestamp")
    last_ts = bundle.get("last_data_timestamp")

    if first_ts is not None and last_ts is not None:
        st.caption(
            f"학습 데이터 기간: {fmt_utc(first_ts)} ~ {fmt_utc(last_ts)} (UTC)"
        )

    n_samples = bundle.get("n_samples")

    if n_samples is not None and n_samples < RECOMMENDED_SAMPLES:

        st.warning(
            f"학습 샘플이 {n_samples:,}개로 아직 적습니다(약 1주일치 미만). "
            "성능 수치가 불안정할 수 있으니 수집을 더 이어간 뒤 재학습해주세요."
        )

    # 실시간 모델에서만 레거시 모델 경고 및 재학습 안내를 제공
    is_legacy_model = bundle.get("target_mode") != "delta"

    if dataset["is_live"] and is_legacy_model:
        st.warning(
            "이전 방식으로 학습된 모델입니다. "
            "'최신 데이터로 재학습'을 눌러 새 모델로 갱신해주세요."
        )

    # 현재 모델만 "다음 5분 예측"을 보여준다.
    # 과거 데이터는 이미 지난 시점이라 미래 예측이 의미가 없다.
    if dataset["is_live"]:

        input_df = None
        last_observed = last_ts

        # 화면을 열 때마다 CSV의 최신 값으로 예측 입력을 새로 만든다.
        # (학습할 때 저장해 둔 값을 그대로 쓰면 학습 시점의 예측만 계속 보이게 된다)
        if not is_legacy_model and LIVE_DATA_PATH.exists():

            try:

                input_df, live_ts = load_live_input(
                    str(LIVE_DATA_PATH),
                    LIVE_DATA_PATH.stat().st_mtime,
                    tuple(feature_columns),
                )

                if live_ts is not None:
                    last_observed = live_ts

            except Exception as exc:

                st.warning(f"최신 수집 데이터를 불러오지 못했습니다: {exc}")

        if input_df is None:

            if isinstance(latest_features, pd.DataFrame):
                input_df = latest_features[feature_columns]
            elif latest_features is not None:
                input_df = pd.DataFrame(
                    [latest_features],
                    columns=feature_columns,
                )

        if input_df is None or input_df[feature_columns].isna().to_numpy().any():

            st.info(
                "최근 연속 수집 데이터가 부족해 지금은 예측할 수 없습니다. "
                "수집이 중간에 끊겼다면 연속으로 약 1시간(관측값 13개) 이상 "
                "쌓여야 예측할 수 있습니다."
            )

        if last_observed is not None:

            age_minutes = (
                pd.Timestamp.now(tz="UTC") - pd.Timestamp(last_observed)
            ).total_seconds() / 60

            st.caption(
                f"마지막 수집: {pd.Timestamp(last_observed):%Y-%m-%d %H:%M} UTC "
                f"({age_minutes:.0f}분 전)"
            )

            if age_minutes > LIVE_STALE_MINUTES:

                st.warning(
                    "수집이 멈춘 것 같습니다. data_collector.py가 "
                    "실행 중인지 확인해주세요."
                )

        render_live_section(model_path, bundle, feature_columns, is_legacy_model)

    else:

        st.info(
            "과거 데이터 모델은 미래 예측 대신, 테스트 구간에서 "
            "모델이 예측한 값과 실제 값을 비교해서 보여줍니다."
        )

    cv_metrics = bundle.get(
        "cv_metrics",
        {},
    )

    test_metrics = bundle.get(
        "test_metrics",
        {},
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "교차검증 MAE",
        f"{cv_metrics.get('MAE', float('nan')):.2f}",
    )

    col2.metric(
        "교차검증 MAPE",
        f"{cv_metrics.get('MAPE (%)', float('nan')):.2f}%",
    )

    col3.metric(
        "테스트 MAE",
        f"{test_metrics.get('MAE', float('nan')):.2f}",
    )

    test_predictions = bundle.get("test_predictions")
    has_test = isinstance(test_predictions, pd.DataFrame) and not test_predictions.empty

    if dataset["is_live"]:

        if has_test:
            with st.expander("테스트 구간 평가 (학습에 쓰지 않은 맨 뒤 일부)"):
                render_test_chart(test_predictions)

    elif has_test:

        render_historical_panel(test_predictions)

    weather_columns = bundle.get("weather_columns") or []

    if weather_columns:

        WEATHER_LABELS = {
            "temp_mean": "평균 기온",
            "precip_mean": "평균 강수량",
            "cloud_mean": "평균 구름량",
            "wind_mean": "평균 풍속",
            "gust_mean": "평균 돌풍",
            "precip_max": "최대 강수량",
            "gust_max": "최대 돌풍",
            "rain_share": "비 오는 비율",
            "severe_weather_idx": "악천후 지수",
        }

        st.caption(
            "사용한 날씨 피처: "
            + ", ".join(WEATHER_LABELS.get(c, c) for c in weather_columns)
        )

    importance = bundle.get("feature_importance")

    if importance:

        importance_df = (
            pd.DataFrame(
                {
                    "피처": list(importance.keys()),
                    "중요도": list(importance.values()),
                }
            )
            .sort_values("중요도", ascending=False)
            .head(10)
            .sort_values("중요도")
        )

        importance_df["구분"] = importance_df["피처"].map(
            lambda c: "날씨" if c in weather_columns else "시간/과거값"
        )

        fig_importance = px.bar(
            importance_df,
            x="중요도",
            y="피처",
            color="구분",
            orientation="h",
            title="피처 중요도 (상위 10개)",
        )

        with st.expander("피처 중요도 보기"):
            st.plotly_chart(
                fig_importance,
                use_container_width=True,
            )

    st.caption(dataset["caption"])


# =========================================================
# 12. 모델 성능 비교
# =========================================================

def show_model_comparison(dataset_key):

    st.subheader("🏆 모델 성능 비교")

    comparison_file = get_comparison_file(dataset_key)

    if not comparison_file.exists():

        st.warning(
            "모델 비교 파일이 없습니다. "
            "먼저 models/train.py를 실행해주세요."
        )

        return

    try:

        comparison_df = pd.read_csv(
            comparison_file
        )

    except Exception as exc:

        st.error(
            f"모델 비교 파일을 읽을 수 없습니다: {exc}"
        )

        return

    if comparison_df.empty:

        st.info(
            "비교할 모델 성능 데이터가 없습니다."
        )

        return

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Baseline (직전값)은 '5분 뒤에도 지금과 같다'고 가정한 기준선입니다. "
        "모델의 오차(MAE)가 이 기준선보다 낮아야 모델을 쓰는 의미가 있습니다."
    )

    # train.py가 저장하는 컬럼명: Model, CV MAE, CV MAPE (%)
    model_column = "Model" if "Model" in comparison_df.columns else None
    mae_column = "CV MAE" if "CV MAE" in comparison_df.columns else None
    mape_column = (
        "CV MAPE (%)" if "CV MAPE (%)" in comparison_df.columns else None
    )

    if model_column and mae_column:

        fig_mae = px.bar(
            comparison_df,
            x=model_column,
            y=mae_column,
            title="모델별 교차검증 MAE",
            labels={
                model_column: "모델",
                mae_column: "MAE",
            },
        )

        st.plotly_chart(
            fig_mae,
            use_container_width=True,
        )

    if model_column and mape_column:

        fig_mape = px.bar(
            comparison_df,
            x=model_column,
            y=mape_column,
            title="모델별 교차검증 MAPE",
            labels={
                model_column: "모델",
                mape_column: "MAPE (%)",
            },
        )

        st.plotly_chart(
            fig_mape,
            use_container_width=True,
        )

    # 날씨 포함/제외 비교 (날씨 데이터가 있는 경우에만)
    if (
        "Test MAE" in comparison_df.columns
        and "Test MAE (날씨 제외)" in comparison_df.columns
    ):

        weather_compare = comparison_df.dropna(
            subset=["Test MAE (날씨 제외)"]
        ).melt(
            id_vars="Model",
            value_vars=["Test MAE (날씨 제외)", "Test MAE"],
            var_name="구분",
            value_name="테스트 MAE",
        )

        weather_compare["구분"] = weather_compare["구분"].map(
            {
                "Test MAE (날씨 제외)": "날씨 제외",
                "Test MAE": "날씨 포함",
            }
        )

        fig_weather = px.bar(
            weather_compare,
            x="Model",
            y="테스트 MAE",
            color="구분",
            barmode="group",
            title="날씨 포함 vs 제외 (테스트 MAE, 낮을수록 좋음)",
            labels={"Model": "모델"},
        )

        st.plotly_chart(
            fig_weather,
            use_container_width=True,
        )

        st.caption(
            "날씨를 넣었을 때 오차가 줄어드는지 확인하는 비교입니다. "
            "차이가 거의 없다면 이 데이터에서는 날씨의 영향이 작다는 뜻입니다."
        )

    folds_file = (
        MODEL_DIR / dataset_key / "cross_validation_folds.csv"
    )

    if folds_file.exists():

        with st.expander("교차검증 Fold별 결과"):

            try:

                folds_df = pd.read_csv(
                    folds_file
                )

                st.dataframe(
                    folds_df,
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as exc:

                st.error(
                    f"Fold 결과를 읽을 수 없습니다: {exc}"
                )


# =========================================================
# 13. 메인 실행
# =========================================================

def main():

    st.title(
        "✈️ 항공 교통 모니터링 및 수요 예측"
    )

    st.markdown(
        """
        OpenSky Network의 실제 항공기 관측 데이터를 활용합니다.

        실시간 항공기 위치를 확인하고,
        선택한 항공기의 비행경로와 미래 항공 교통량을 확인할 수 있습니다.
        """
    )

    tab_map, tab_prediction = st.tabs(
        [
            "🌍 실시간 항공기 지도",
            "📊 항공 교통량 예측 및 모델 비교",
        ]
    )

    with tab_map:

        show_live_map()

    with tab_prediction:

        dataset_label = st.radio(
            "데이터 기준",
            list(DATASETS.keys()),
            horizontal=True,
            key="dataset_radio",
        )

        dataset_key = DATASETS[dataset_label]["key"]

        if DATASETS[dataset_label]["is_live"]:

            render_retrain_controls(dataset_key)

        prediction_tab, comparison_tab = st.tabs(
            [
                "🔮 항공 교통량 예측",
                "🏆 모델 성능 비교",
            ]
        )

        with prediction_tab:

            show_model_prediction(dataset_key)

        with comparison_tab:

            show_model_comparison(dataset_key)


if __name__ == "__main__":

    main()