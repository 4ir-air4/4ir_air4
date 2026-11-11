from pathlib import Path
from datetime import datetime
from html import escape
import json
import math
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import folium
import joblib
import pandas as pd
import requests
import streamlit as st

from folium.plugins import Fullscreen, FastMarkerCluster
from streamlit_folium import st_folium

import sys

# 팀 app.py 에서 페이지로 불려도 ojh/ 안의 탭 파일을 찾도록 이 폴더를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _2_traffic_tab import show_traffic_prediction_tab
from _3_airport_tab import show_airport_demand


# =========================================================
# 1. 기본 설정
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent
STATES_CACHE_PATH = ROOT_DIR / "data" / "cache" / "opensky_states_last.json"

OPENSKY_BASE_URL = "https://opensky-network.org/api"

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
# 4. 궤적 조회용 로컬 프록시 (브라우저 CORS 차단 회피)
# =========================================================

TRACK_PROXY_PORT = 8765
TRACK_PROXY_CACHE_SEC = 120

# OpenSky는 한 비행을 여러 조각(track)으로 나눠 저장하고, time=0 으로는 마지막 조각만 준다.
# 그래서 마지막 조각이 공중에서 시작하면 바로 앞 조각을 더 받아 이륙 지점까지 이어 붙인다.
TRACK_MAX_EXTRA_SEGMENTS = 3   # 앞 조각을 최대 몇 개까지 더 받을지
TRACK_DEPARTURE_ALT_M = 1500    # 이 고도보다 낮거나 지상이면 이륙(출발) 지점으로 본다
# 조각 안에는 몇 시간 전의 '다른 비행'이 섞여 있기도 하다 (사이에 지상 대기).
# 20분 넘게 끊겼는데 그 사이 평균 속도가 300km/h 미만이면 지상 대기 = 비행 경계로 본다.
# (바다 위 수신 공백은 끊겨도 평균 속도가 800km/h 안팎이라 같은 비행으로 이어진다)
TRACK_BOUNDARY_MIN_GAP_SEC = 20 * 60
TRACK_BOUNDARY_MAX_SPEED_KMH = 300


def _km_between(lat1, lon1, lat2, lon2):
    r = math.pi / 180
    a = (
        math.sin((lat2 - lat1) * r / 2) ** 2
        + math.cos(lat1 * r) * math.cos(lat2 * r) * math.sin((lon2 - lon1) * r / 2) ** 2
    )
    return 12742 * math.asin(math.sqrt(a))


def _is_departure_point(point):
    """궤적 점 [time, lat, lon, baro_altitude, true_track, on_ground] 이 지상/저고도인지."""
    altitude, on_ground = point[3], point[5]
    return bool(on_ground) or (altitude is not None and altitude < TRACK_DEPARTURE_ALT_M)


def _is_flight_boundary(a, b):
    """두 점 사이가 지상 대기(다른 비행으로 넘어감)인지."""
    gap = b[0] - a[0]
    if gap < TRACK_BOUNDARY_MIN_GAP_SEC:
        return False
    speed = _km_between(a[1], a[2], b[1], b[2]) / (gap / 3600)
    return speed < TRACK_BOUNDARY_MAX_SPEED_KMH


def _trim_to_current_flight(path):
    """뒤에서부터 보다가 비행 경계가 나오면 그 뒤(=지금 비행)만 남긴다."""
    for i in range(len(path) - 1, 0, -1):
        if _is_flight_boundary(path[i - 1], path[i]):
            return path[i:], True
    return path, False


def merge_track_segments(fetch_segment, live):
    """마지막 조각(live) 앞에 같은 비행의 이전 조각들을 이어 붙인다.

    fetch_segment(time) -> 그 시각의 조각(dict) 또는 None.
    이전 비행이 섞이지 않도록 비행 경계에서 자르고, 이륙 지점을 찾으면 거기서 멈춘다.
    """
    path = [p for p in (live.get("path") or []) if p[1] is not None and p[2] is not None]
    path, cut = _trim_to_current_flight(path)

    for _ in range(TRACK_MAX_EXTRA_SEGMENTS):
        if cut or not path or _is_departure_point(path[0]):
            break  # 비행 경계에 닿았거나, 이미 이륙 지점부터 있다

        prev = fetch_segment(int(path[0][0]) - 60)
        prev_path = [
            p for p in ((prev or {}).get("path") or [])
            if p[1] is not None and p[2] is not None and p[0] < path[0][0]
        ]
        if not prev_path:
            break  # 앞 조각이 없다 (수신 기록 없음)

        combined, cut = _trim_to_current_flight(prev_path + path)
        added = len(combined) - len(path)
        if added <= 0:
            break  # 앞 조각은 전부 다른 비행이었다

        # 새로 붙은 부분에서 가장 마지막 지상/저고도 점 = 이번 비행의 이륙 지점
        takeoff = next(
            (i for i in range(added - 1, -1, -1) if _is_departure_point(combined[i])),
            None,
        )
        if takeoff is not None:
            path = combined[takeoff:]
            break
        path = combined

    merged = dict(live)
    merged["path"] = path
    if path:
        merged["startTime"] = path[0][0]
    return merged


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

            def fetch(at):
                return requests.get(
                    f"{OPENSKY_BASE_URL}/tracks/all",
                    headers={"Authorization": f"Bearer {get_token()}"},
                    params={"icao24": icao24, "time": at},
                    timeout=30,
                )

            def fetch_segment(at):
                try:
                    r = fetch(at)
                    return r.json() if r.status_code == 200 else None
                except Exception:
                    return None

            try:
                r = fetch(0)
                status, body = r.status_code, r.content
                if status == 200:
                    merged = merge_track_segments(fetch_segment, r.json())
                    body = json.dumps(merged).encode("utf-8")
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


# =========================================================
# 6. 항공기 마커 (기종 크기/방향/고도 색상 + 클릭 시 사진·경로·실제 궤적)
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
    //      8 속도, 9 상태, 10 진행방향(도), 11 아이콘 크기, 12 분류 이름, 13 아이콘 모양,
    //      14 진행방향 값이 있는지(1/0)
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
                       "<div class='ac-route-note' style='font-size:11px; color:#b26a00;'></div>" +
                       "<b>기압고도:</b> " + row[6] + "<br>" +
                       "<b>기하고도:</b> " + row[7] + "<br>" +
                       "<b>속도:</b> " + row[8] + "<br>" +
                       "<b>상태:</b> " + row[9] +
                       "<div class='ac-track' style='margin-top:4px; font-size:11px; color:#666;'></div></div>"; }
    marker.bindPopup(buildPopup);
    marker.bindTooltip(row[2] + " (" + row[3] + ")");

    // 고도 범례(id=alt-legend)는 지도 위에 고정되어 팝업을 가리므로, 팝업이 열려 있는 동안 숨긴다
    function setLegendVisible(show) {
        var legend = document.getElementById('alt-legend');
        if (legend) { legend.style.display = show ? '' : 'none'; }
    }
    marker.on('popupclose', function () { setLegendVisible(true); });

    // 팝업을 열 때만 브라우저에서 사진/경로/궤적을 조회한다 (수천 대를 한꺼번에 조회하지 않기 위해)
    marker.on('popupopen', function (e) {
        setLegendVisible(false);
        var root = e.popup.getElement();
        if (!root) { return; }
        var photoEl = root.querySelector('.ac-photo');
        var originEl = root.querySelector('.ac-origin');
        var destEl = root.querySelector('.ac-dest');
        var trackEl = root.querySelector('.ac-track');
        var typeEl = root.querySelector('.ac-type');
        var noteEl = root.querySelector('.ac-route-note');
        var hex = String(row[3]).toLowerCase();
        var callsign = String(row[2]).trim();
        var map = e.popup._map || e.target._map;
        var cur = e.target.getLatLng();
        var onGround = row[9] === '지상';
        var heading = row[14] ? row[10] : null;
        var ALT = __ALT_CONFIG__;

        window.__acCache = window.__acCache || {photo: {}, route: {}, track: {}};
        var id = window.__acPopupId = (window.__acPopupId || 0) + 1;
        var state = {rawRoute: undefined, route: undefined, track: undefined, trackErr: null};

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
            var GAP_STYLE = {color: '#555', weight: 3, opacity: 0.85, dashArray: '4 7'};
            var hasGap = false;

            if (tr && tr.length > 1) {
                // 실제로 날아온 궤적: 고도 색상으로 구간별 표시.
                // OpenSky 궤적은 직선 비행 구간의 점을 생략해서 주기 때문에, 점 사이가 멀어도
                // 수신 공백이라고 단정할 수 없다. 그래서 궤적 안의 구간은 모두 실선으로 잇는다.
                for (var i = 1; i < tr.length; i++) {
                    addLine(layer, [tr[i - 1], tr[i]], {
                        color: altColor(tr[i][2], tr[i][3]), weight: 4, opacity: 0.95
                    });
                }
                if (hasXY(o)) {
                    var oLon = o.longitude + 360 * Math.round((tr[0][1] - o.longitude) / 360);
                    // 출발 공항과 궤적 시작점이 멀면 OpenSky가 수신하지 못한 구간 -> 회색 점선으로 이어서 표시
                    if (kmBetween(o.latitude, oLon, tr[0][0], tr[0][1]) > 50) {
                        hasGap = true;
                        addLine(layer, gcPath(o.latitude, oLon, tr[0][0], tr[0][1], 32), GAP_STYLE);
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

            // 선 설명 (궤적·경로 응답이 모두 와야 정확하므로 여기서 한 번에 정한다)
            if (tr && tr.length > 1) {
                var since = '';
                if (tr.since) {
                    var mins = Math.max(0, Math.round((Date.now() / 1000 - tr.since) / 60));
                    var span = mins >= 60 ? Math.floor(mins / 60) + '시간 ' + (mins % 60) + '분' : mins + '분';
                    since = ' · 수신 기록: 최근 ' + span + '치';
                }
                trackEl.textContent = '실선: 실제 이동 궤적 (고도 색상)' + since +
                    (hasGap ? ' · 회색 점선: 출발 공항~수신 시작 지점 (수신 기록 없음, 추정)' : '');
            } else if (state.track !== undefined) {
                var why = state.trackErr ? '궤적 조회 실패 (' + state.trackErr + ')' : '궤적 좌표가 부족합니다';
                trackEl.textContent = why + (fr ? ' - 점선은 추정 경로입니다' : '');
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
        // adsbdb는 '이 편명은 보통 이 노선'이라는 정적 정보라서, 같은 편명을 다른 노선에
        // 다시 쓰는 항공사(특히 저비용·전세기)는 출발·도착이 틀리게 나온다 (실측 20~30%).
        // 확실한 근거가 있을 때만 바꾸거나 숨긴다.
        //  - 근거 1: 실제 궤적이 이륙 지점부터 있으면, 그 위치가 어느 공항인지로 판단
        //  - 근거 2: 현재 위치가 노선에서 아주 멀리(1000km 이상, 노선 길이의 절반 이상) 벗어남
        // 진행방향은 항로 우회·선회 때문에 틀릴 수 있어서 안내 문구에만 쓴다.
        function bearingTo(lat1, lon1, lat2, lon2) {
            var r = Math.PI / 180;
            var y = Math.sin((lon2 - lon1) * r) * Math.cos(lat2 * r);
            var x = Math.cos(lat1 * r) * Math.sin(lat2 * r) -
                    Math.sin(lat1 * r) * Math.cos(lat2 * r) * Math.cos((lon2 - lon1) * r);
            return (Math.atan2(y, x) / r + 360) % 360;
        }

        var APT_NEAR_KM = 60;  // 궤적 시작점이 공항에서 이 거리 안이면 그 공항에서 이륙한 것으로 본다

        // 결과: {route: 그릴 경로 또는 null, note: 팝업 안내 문구}
        function checkRoute(fr, tr) {
            if (!fr || !hasXY(fr.origin) || !hasXY(fr.destination)) { return {route: fr, note: ''}; }
            var o = fr.origin, d = fr.destination;
            var total = kmBetween(o.latitude, o.longitude, d.latitude, d.longitude);
            var toO = kmBetween(cur.lat, cur.lng, o.latitude, o.longitude);
            var toD = kmBetween(cur.lat, cur.lng, d.latitude, d.longitude);

            // 근거 2: 노선에서 아주 멀리 벗어나 있음 (지상이면 두 공항 어디에도 없음)
            var farOff = onGround
                ? (toO > 30 && toD > 30)
                : (toO + toD - total > Math.max(1000, total * 0.5));
            if (farOff) {
                return {route: null, note: '현재 위치가 이 노선과 동떨어져 있어 경로 DB 정보가 틀린 것 같습니다. 경로선은 그리지 않았습니다'};
            }

            // 근거 1: 실제 궤적이 이륙 지점부터 있으면 그 공항으로 판단
            var swapped = false;
            if (tr && tr.depart && tr.start) {
                var sO = kmBetween(tr.start[0], tr.start[1], o.latitude, o.longitude);
                var sD = kmBetween(tr.start[0], tr.start[1], d.latitude, d.longitude);
                if (sO > APT_NEAR_KM && sD <= APT_NEAR_KM) {
                    swapped = true;
                } else if (sO > APT_NEAR_KM) {
                    return {route: null, note: '실제 이륙 지점이 경로 DB의 출발 공항과 달라 경로 DB 정보가 틀린 것 같습니다. 경로선은 그리지 않았습니다 (실선 궤적은 실제 데이터입니다)'};
                }
            }
            var route = swapped ? {origin: d, destination: o} : fr;
            var note = swapped ? '실제 궤적이 ' + fmtAirport(d) + '에서 이륙해서, 경로 DB와 반대로 출발·도착을 바꿔 표시했습니다' : '';

            // 진행방향은 참고용 안내만 (경로선은 그대로 그린다)
            var dst = route.destination;
            var toDst = kmBetween(cur.lat, cur.lng, dst.latitude, dst.longitude);
            if (!onGround && heading !== null && toDst > 150 && kmBetween(cur.lat, cur.lng, route.origin.latitude, route.origin.longitude) > 150) {
                var diff = Math.abs((heading - bearingTo(cur.lat, cur.lng, dst.latitude, dst.longitude) + 540) % 360 - 180);
                if (diff > 135) {
                    note = (note ? note + ' · ' : '') + '진행방향이 도착지와 반대입니다. 도착지 정보가 틀렸을 수 있습니다';
                }
            }
            return {route: route, note: note};
        }

        // 경로 DB 응답과 궤적 응답이 각각 도착할 때마다 다시 판정한다
        function applyRoute() {
            if (state.rawRoute === undefined) { redraw(); return; }  // 경로 DB 응답 대기 중
            var fr = state.rawRoute;
            var checked = checkRoute(fr, state.track);
            state.route = checked.route;
            var shown = checked.route || fr;
            var mark = (fr && !checked.route) ? ' (확인 안 됨)' : '';
            originEl.textContent = shown ? fmtAirport(shown.origin) + mark : '정보 없음';
            destEl.textContent = shown ? fmtAirport(shown.destination) + mark : '정보 없음';
            noteEl.textContent = checked.note;
            redraw();
        }

        function showRoute(fr) {
            state.rawRoute = fr || null;
            applyRoute();
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
            var times = [];
            var prev = null;
            var first = null;
            path.forEach(function (p) {
                if (p[1] == null || p[2] == null) { return; }
                if (first === null) { first = p; }
                var lon = p[2];
                if (prev !== null) { lon += 360 * Math.round((prev - lon) / 360); }
                prev = lon;
                pts.push([p[1], lon, p[3], p[5]]);
                times.push(p[0]);
            });
            if (pts.length < 2) { return null; }
            var shift = 360 * Math.round((cur.lng - pts[pts.length - 1][1]) / 360);
            pts.forEach(function (p) { p[1] += shift; });

            // 항공기 위치는 최대 10분 전 스냅샷이라 궤적이 아이콘보다 앞서 있을 수 있다.
            // 아이콘과 가장 가까운 점까지만 사용해서 선이 아이콘 앞으로 튀어나오지 않게 한다.
            // 단, 최근 30분 안의 점에서만 찾는다 (예전에 같은 곳을 지난 점에서 잘리지 않도록).
            var recent = times[times.length - 1] - 1800;
            var nearest = pts.length - 1, best = Infinity;
            pts.forEach(function (p, i) {
                if (times[i] < recent) { return; }
                var dd = kmBetween(cur.lat, cur.lng, p[0], p[1]);
                if (dd <= best) { best = dd; nearest = i; }
            });
            pts = pts.slice(0, nearest + 1);

            var last = pts[pts.length - 1];
            pts.push([cur.lat, cur.lng, last[2], last[3]]);
            // 수신 기록이 몇 분치인지, 궤적이 이륙 지점(지상/저고도)부터 시작하는지
            pts.since = first[0];
            pts.start = [first[1], first[2]];
            pts.depart = !!first[5] || (first[3] != null && first[3] < 1500);
            return pts;
        }

        function showTrack(pts, err) {
            state.track = pts || null;
            state.trackErr = err || null;
            applyRoute();  // 궤적 시작점으로 출발 공항을 다시 판정
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
AIRCRAFT_DB_PATH = ROOT_DIR / "data" / "aircraft_db" / "aircraft_database.csv"
AIRCRAFT_DB_CACHE = ROOT_DIR / "data" / "aircraft_db" / "aircraft_category_cache.pkl"

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

    # git으로 받으면 두 파일의 수정 시각이 몇 초씩 어긋날 수 있어 60초 여유를 둔다
    if (
        AIRCRAFT_DB_CACHE.exists()
        and AIRCRAFT_DB_CACHE.stat().st_mtime >= AIRCRAFT_DB_PATH.stat().st_mtime - 60
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

        heading_known = 0 if pd.isna(heading) else 1
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
            heading_known,
        ]

        rows.append(base_item)

        # 좌우 무한 스크롤 시에도 비행기가 보이도록 -360도, +360도 위치에 마커 복제 추가
        for offset in WRAP_COPIES:
            wrapped_item = base_item.copy()
            wrapped_item[1] = float(lon) + offset
            rows.append(wrapped_item)

    return rows


def add_aircraft_markers(aircraft_map, df):

    callback = MARKER_CALLBACK.replace("__ALT_CONFIG__", build_alt_config())

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
# 7. 실시간 항공기 지도
# =========================================================

def show_live_map():

    st.subheader("🌍 실시간 항공기 위치")

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("🔄 데이터 새로고침"):
            load_aircraft_data.clear()
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
    <div id="alt-legend" style="
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

    # 실제 이동 궤적(OpenSky tracks)은 브라우저가 이 로컬 프록시(127.0.0.1)를 거쳐 조회한다.
    # 인증 토큰은 프록시(서버)에만 있고 지도 HTML에는 넣지 않는다.
    # 프록시가 127.0.0.1이라 앱을 실행한 PC의 브라우저에서만 궤적이 보인다.
    start_track_proxy(
        get_secret("OPENSKY_CLIENT_ID"),
        get_secret("OPENSKY_CLIENT_SECRET"),
    )

    add_aircraft_markers(
        aircraft_map,
        df,
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
# 8. 메인 실행
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

    tab_map, tab_prediction, tab_airport = st.tabs(
        [
            "🌍 실시간 항공기 지도",
            "📊 항공 교통량 예측 및 모델 비교",
            "🛫 전 세계 공항 이용객 수 수요 예측",
        ]
    )

    with tab_map:

        show_live_map()

    with tab_prediction:

        show_traffic_prediction_tab()

    with tab_airport:

        show_airport_demand()


if __name__ == "__main__":

    main()