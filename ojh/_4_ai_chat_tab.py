"""AI 항공 도우미 '에어로': 모든 페이지 오른쪽 아래에 떠 있는 채팅 버튼 (app.py 에서 floating_chat() 호출).

AI 는 secrets.toml 에 있는 키에 따라 고른다 (둘 다 있으면 Gemini):
  - GEMINI_API_KEY    : Google Gemini (gemini-3.1-flash-lite, 무료 등급 사용 가능)
                        + TAVILY_API_KEY 가 있으면 Tavily 무료 검색(월 1,000회)으로 최신 소식을 찾는다
  - SERPAPI_API_KEY   : (두 AI 공통) 날짜·노선을 넣어 Google 항공권 검색 결과를 조회한다 (무료 월 250회)
  - ANTHROPIC_API_KEY : Claude (유료, 웹 검색으로 항공권 가격·시간표 조회 가능)

대시보드 데이터는 아래 '도구'로만 본다 (임의 코드 실행 없음, 읽기 전용).
항공권 가격·시간표 같은 최신 정보는 Anthropic 서버의 웹 검색 도구로 찾고, 사진은 Claude 가 직접 본다.
  - list_tables      : 조회할 수 있는 표와 칸 목록
  - query_table      : 표 하나를 조건·그룹·정렬해서 최대 50행 돌려줌
  - get_ldh_summary  : 팀원(ldh) 카테고리별 결과 요약(summary.json)

API 키: 저장소 최상위 .streamlit/secrets.toml 에 ANTHROPIC_API_KEY = "sk-ant-..." 를 넣는다 (또는 같은 이름의 환경변수).
"""
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent            # ojh/
REPO_DIR = ROOT_DIR.parent
AIRPORT_CSV = ROOT_DIR / "data" / "airport_demand" / "airport_demand_passengers_monthly.csv"
GDP_CSV = ROOT_DIR / "data" / "airport_demand" / "airport_demand_gdp_growth.csv"
LIVE_CSV = ROOT_DIR / "data" / "raw" / "aircraftlive_counts.csv"
AIRPORT_MODEL_DIR = ROOT_DIR / "models" / "saved" / "airport"
LDH_RESULTS = REPO_DIR / "ldh" / "results" / "full"

MODEL = "claude-opus-5-5"
GEMINI_MODEL = "gemini-3.1-flash-lite"
MAX_ROWS = 50          # 도구 결과 한 번에 돌려줄 최대 행 수 (토큰 절약)
MAX_TOOL_ROUNDS = 8    # 한 질문에서 도구를 부를 수 있는 최대 횟수



# ---------------------------------------------------------
# 1. 조회할 표 만들기
# ---------------------------------------------------------
def _mtime(path):
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _airport_monthly(mtime):
    df = pd.read_csv(AIRPORT_CSV, parse_dates=["date"], encoding="utf-8-sig")
    df["year"], df["month"] = df["date"].dt.year, df["date"].dt.month
    df["date"] = df["date"].dt.strftime("%Y-%m")
    return df


@st.cache_data(show_spinner="공항별 12개월 예측을 계산하는 중...")
def _airport_forecast(mtime):
    """공항 수요 탭과 같은 방법(선정 모델, 공항별 마지막 실측 달 다음부터)으로 만든 예측값만 모은 표"""
    from _3_airport_tab import CONTEXT_FILE, MODEL_FILES, load_pickle, load_trainer

    ctx = load_pickle(CONTEXT_FILE)
    if ctx is None:
        return pd.DataFrame()
    bundle = load_pickle(AIRPORT_MODEL_DIR / MODEL_FILES[ctx["best_model"]])
    filled, last_obs = load_trainer().forecast_all_filled(bundle, ctx, n_months=12)
    meta = ctx["meta"]
    rows = []
    for code, last in last_obs.items():
        future = filled.loc[code, filled.columns > last].dropna()
        for d, v in future.items():
            rows.append({"airport": code, "airport_name": meta.loc[code, "airport_name"], "country": meta.loc[code, "country"],
                         "region": meta.loc[code, "region"], "date": f"{d:%Y-%m}", "year": d.year, "month": d.month,
                         "predicted_passengers": round(float(v)), "last_actual_month": f"{last:%Y-%m}"})
    out = pd.DataFrame(rows)
    out.attrs["model"] = ctx["best_model"]
    return out


@st.cache_data(show_spinner=False)
def _csv(path, mtime):
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def _live(mtime):
    df = pd.read_csv(LIVE_CSV, encoding="utf-8-sig")
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp_kst"] = ts.dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d %H:%M")
    df["date_kst"] = ts.dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    df["hour_kst"] = ts.dt.tz_convert("Asia/Seoul").dt.hour
    return df.drop(columns=["timestamp"])


TABLES = {
    "airport_monthly": {
        "desc": "공항별 월별 실제 이용객 수 (미국 BTS T-100, 유럽 Eurostat, 영국 CAA, 한국 에어포탈). region: US/Europe/Korea/Other. "
                "airport 코드: 미국·한국은 IATA 3자리(예: ICN=인천, GMP=김포, ATL=애틀랜타), 유럽·영국은 ICAO 4자리(예: EGLL=런던 히드로, LFPG=파리 샤를드골). "
                "airport_name 은 한글 이름(예: 인천국제공항).",
        "load": lambda: _airport_monthly(_mtime(AIRPORT_CSV)),
    },
    "airport_forecast": {
        "desc": "공항 수요 예측 탭의 선정 모델이 만든 향후 12개월 예측 이용객 수 (공항마다 자기 마지막 실측 달 다음부터 시작). "
                "최근 12개월 넘게 데이터가 끊긴 공항은 빠져 있다.",
        "load": lambda: _airport_forecast(_mtime(AIRPORT_MODEL_DIR / "airport_context.pkl")),
    },
    "airport_model_comparison": {
        "desc": "공항 수요 예측 모델 성능 비교. 선정 기준은 'Val 12개월 재귀 WAPE (%)'(낮을수록 좋음). Baseline 은 작년 같은 달 값을 그대로 쓰는 기준선.",
        "load": lambda: _csv(AIRPORT_MODEL_DIR / "model_comparison.csv", _mtime(AIRPORT_MODEL_DIR / "model_comparison.csv")),
    },
    "gdp_growth": {
        "desc": "나라별 연도별 실질 GDP 성장률(%). country 는 ISO 2자리 코드.",
        "load": lambda: _csv(GDP_CSV, _mtime(GDP_CSV)),
    },
    "live_aircraft": {
        "desc": "OpenSky 실시간 수집기가 5분마다 저장한 전 세계 비행 중 항공기 수(aircraft_count)와 항공기 위치 중심점의 날씨. 시각은 한국 시간(KST).",
        "load": lambda: _live(_mtime(LIVE_CSV)),
    },
}


def _get_table(name):
    if name not in TABLES:
        raise ValueError(f"'{name}' 표는 없습니다. 가능한 표: {', '.join(TABLES)}")
    df = TABLES[name]["load"]()
    if df is None or df.empty:
        raise ValueError(f"'{name}' 표의 파일이 없거나 비어 있습니다.")
    return df


def _ldh_categories():
    try:
        from ldh.core.registry import discover
        return {c["slug"]: c["title"] for c in discover()[0]}
    except Exception:
        return {p.name: p.name for p in LDH_RESULTS.iterdir() if p.is_dir()} if LDH_RESULTS.exists() else {}


# ---------------------------------------------------------
# 2. 도구 실행
# ---------------------------------------------------------
OPS = {"==", "!=", ">", ">=", "<", "<=", "contains", "in"}
AGGS = {"sum", "mean", "median", "min", "max", "count", "nunique"}


def _to_csv(df, total):
    note = f"(전체 {total:,}행 중 {len(df):,}행)\n" if total > len(df) else f"({total:,}행)\n"
    return note + df.to_csv(index=False, lineterminator="\n", float_format=lambda x: f"{x:.4g}" if abs(x) < 1 else f"{x:,.2f}".replace(",", ""))


def tool_list_tables(_):
    lines = []
    for name, info in TABLES.items():
        try:
            df = _get_table(name)
            cols = ", ".join(f"{c}({'숫자' if pd.api.types.is_numeric_dtype(df[c]) else '문자'})" for c in df.columns)
            extra = ""
            if "date" in df.columns:
                extra = f" 기간 {df['date'].min()}~{df['date'].max()}."
            elif "date_kst" in df.columns:
                extra = f" 기간 {df['date_kst'].min()}~{df['date_kst'].max()}."
            lines.append(f"- {name}: {info['desc']} {len(df):,}행.{extra}\n  칸: {cols}")
        except Exception as e:
            lines.append(f"- {name}: 지금 사용할 수 없음 ({e})")
    cats = _ldh_categories()
    lines.append("- (get_ldh_summary 로 조회) 팀원 결과 요약 카테고리: " + ", ".join(f"{k}={v}" for k, v in cats.items()))
    return "\n".join(lines)


def _apply_filter(df, f):
    col, op, val = f.get("column"), f.get("op"), f.get("value")
    if col not in df.columns:
        raise ValueError(f"'{col}' 칸이 없습니다. 칸: {', '.join(df.columns)}")
    if op not in OPS:
        raise ValueError(f"op 는 {sorted(OPS)} 중 하나여야 합니다.")
    s = df[col]
    if op == "contains":
        return df[s.astype(str).str.contains(str(val), case=False, na=False, regex=False)]
    if op == "in":
        vals = val if isinstance(val, list) else [v.strip() for v in str(val).split(",")]
        if pd.api.types.is_numeric_dtype(s):
            vals = [float(v) for v in vals]
        return df[s.isin(vals) | s.astype(str).isin([str(v) for v in vals])]
    if pd.api.types.is_numeric_dtype(s) and not isinstance(val, (int, float)):
        val = float(val)
    return df[{"==": s == val, "!=": s != val, ">": s > val, ">=": s >= val, "<": s < val, "<=": s <= val}[op]]


def tool_query_table(args):
    df = _get_table(args.get("table", ""))
    for f in args.get("filters") or []:
        df = _apply_filter(df, f)
    group_by = args.get("group_by") or []
    metrics = args.get("metrics") or []
    for c in group_by:
        if c not in df.columns:
            raise ValueError(f"group_by 의 '{c}' 칸이 없습니다.")
    if group_by or metrics:
        if not metrics:
            metrics = [{"column": group_by[0], "agg": "count"}]
        named = {}
        for m in metrics:
            col, agg = m.get("column"), m.get("agg")
            if col not in df.columns or agg not in AGGS:
                raise ValueError(f"metrics 항목이 잘못됐습니다: {m}. agg 는 {sorted(AGGS)} 중 하나.")
            named[f"{col}_{agg}"] = (col, agg)
        df = df.groupby(group_by, dropna=False).agg(**named).reset_index() if group_by else df.agg(**named).T.reset_index(drop=True)
    elif args.get("columns"):
        missing = [c for c in args["columns"] if c not in df.columns]
        if missing:
            raise ValueError(f"없는 칸: {missing}")
        df = df[args["columns"]]
    sort_by = args.get("sort_by")
    if sort_by:
        if sort_by not in df.columns:
            raise ValueError(f"sort_by '{sort_by}' 칸이 없습니다. 지금 표의 칸: {', '.join(map(str, df.columns))}")
        df = df.sort_values(sort_by, ascending=not args.get("descending", True))
    limit = max(1, min(int(args.get("limit") or 20), MAX_ROWS))
    return _to_csv(df.head(limit), len(df))


def tool_get_ldh_summary(args):
    slug = args.get("category", "")
    path = LDH_RESULTS / slug / "summary.json"
    if not path.exists():
        return f"'{slug}' 요약이 없습니다. 가능한 카테고리: {', '.join(_ldh_categories())}"
    data = json.loads(path.read_text(encoding="utf-8"))
    keep = ("title", "kind", "train_period", "test_period", "note", "roc_auc", "pr_auc", "baseline_pr_auc", "mae", "rmse",
            "baseline_mae", "improvement", "r2", "accuracy", "f1", "model")
    models = [{k: v for k, v in m.items() if k in keep} for m in data.get("models", [])[:15]]
    out = {"headline": data.get("headline"), "kpis": data.get("kpis"), "notes": data.get("notes"),
           "updated": data.get("updated"), "models(최대 15개)": models}
    return json.dumps(out, ensure_ascii=False, default=str)


TAVILY_URL = "https://api.tavily.com/search"


def tool_web_search(args):
    """Tavily 검색: 요약 한 줄 + 결과마다 '- 제목 | 주소' 와 본문 일부"""
    import requests

    query = (args.get("query") or "").strip()
    if not query:
        raise ValueError("query 가 비어 있습니다.")
    r = requests.post(TAVILY_URL, headers={"Authorization": f"Bearer {_secret('TAVILY_API_KEY')}"},
                      json={"query": query, "max_results": 5, "search_depth": "basic", "include_answer": True}, timeout=25)
    if r.status_code in (429, 432, 433):
        raise RuntimeError("이번 달 무료 검색 한도를 다 썼습니다. 다음 달 1일에 다시 쓸 수 있습니다.")
    if r.status_code == 401:
        raise RuntimeError("Tavily API 키가 올바르지 않습니다.")
    r.raise_for_status()
    data = r.json()
    lines = [f"검색 요약: {data['answer']}"] if data.get("answer") else []
    for x in data.get("results", []):
        lines.append(f"- {x.get('title', '')} | {x.get('url', '')}\n  {(x.get('content') or '')[:600]}")
    return "\n".join(lines) or "검색 결과가 없습니다."


def _parse_sources(text):
    """tool_web_search 결과에서 (제목, 주소) 목록을 뽑는다"""
    out = []
    for line in text.splitlines():
        if line.startswith("- ") and " | http" in line:
            title, url = line[2:].rsplit(" | ", 1)
            out.append((title, url))
    return out


SERPAPI_URL = "https://serpapi.com/search"
TRAVEL_CLASS = {"economy": 1, "premium_economy": 2, "business": 3, "first": 4}
PRICE_LEVEL = {"low": "싼 편", "typical": "보통", "high": "비싼 편"}


def _fmt_minutes(m):
    return f"{int(m) // 60}시간 {int(m) % 60}분" if m else ""


def tool_search_flights(args):
    """Google 항공권 검색 결과(SerpApi). 같은 검색은 최대 1시간 저장본을 쓴다."""
    import datetime

    import requests

    dep, arr = (args.get("departure") or "").upper().replace(" ", ""), (args.get("arrival") or "").upper().replace(" ", "")
    out_date, ret_date = args.get("outbound_date") or "", args.get("return_date") or ""
    if not dep or not arr or not out_date:
        raise ValueError("departure, arrival, outbound_date 가 필요합니다.")
    today = datetime.date.today()
    for d in filter(None, [out_date, ret_date]):
        try:
            if datetime.date.fromisoformat(d) < today:
                raise ValueError(f"{d} 는 지난 날짜입니다. 오늘({today}) 이후 날짜로 조회해주세요.")
        except ValueError as e:
            raise ValueError(str(e) if "지난 날짜" in str(e) else f"날짜 형식은 YYYY-MM-DD 입니다: {d}")
    params = {"engine": "google_flights", "api_key": _secret("SERPAPI_API_KEY"), "departure_id": dep, "arrival_id": arr,
              "outbound_date": out_date, "type": 1 if ret_date else 2, "currency": "KRW", "hl": "ko", "gl": "kr",
              "adults": int(args.get("adults") or 1), "travel_class": TRAVEL_CLASS.get(args.get("travel_class") or "economy", 1)}
    if ret_date:
        params["return_date"] = ret_date
    r = requests.get(SERPAPI_URL, params=params, timeout=40)
    data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    if r.status_code == 401:
        raise RuntimeError("SerpApi 키가 올바르지 않습니다.")
    if r.status_code == 429 or "run out of searches" in str(data.get("error", "")).lower():
        raise RuntimeError("이번 달 무료 항공권 조회 한도(250회)를 다 썼습니다.")
    if data.get("error"):
        if "hasn't returned any results" in str(data["error"]):
            return (f"결과 없음: Google 항공권에 {dep}→{arr} {out_date} 항공편이 없습니다. "
                    "작은 공항이면 가까운 큰 공항(같은 나라 수도 공항, 국경 너머 가까운 공항)을 쉼표로 함께 넣어 다시 조회하세요. "
                    "예: 시우다드델에스테 AGT → AGT,IGU,ASU")
        raise RuntimeError(f"항공권 조회 오류: {data['error']}")
    r.raise_for_status()

    flights = (data.get("best_flights") or []) + (data.get("other_flights") or [])
    if not flights:
        return "조건에 맞는 항공편이 없습니다."
    kind = "왕복(가는 편 기준, 가격은 왕복 합계)" if ret_date else "편도"
    lines = [f"Google 항공권 검색 결과 · {dep}→{arr} · {out_date}{' ~ ' + ret_date if ret_date else ''} · {kind} · 원화(KRW) · 성인 {params['adults']}명"]
    ins = data.get("price_insights") or {}
    if ins:
        rng = ins.get("typical_price_range") or []
        lines.append(f"가격 분석: 최저가 {ins.get('lowest_price', '?'):,}원 · 지금 가격 수준 '{PRICE_LEVEL.get(ins.get('price_level'), ins.get('price_level', '?'))}'"
                     + (f" · 보통 가격대 {rng[0]:,}~{rng[1]:,}원" if len(rng) == 2 else "") if isinstance(ins.get("lowest_price"), (int, float)) else f"가격 분석: {ins}")
    if ret_date:
        lines.append(f"주의: 아래 시각·편명·소요 시간은 모두 가는 편({out_date})의 것이다. 오는 편({ret_date}) 시각은 이 결과에 없다. 가격만 왕복 합계다.")
        lines.append("왕복 합계 가격(원),가는 편 항공사,가는 편 편명,가는 편 출발,가는 편 도착,가는 편 소요,가는 편 경유")
    else:
        lines.append("가격(원),항공사,편명,출발,도착,총 소요,경유")
    sort = args.get("sort_by") or "price"
    keys = {"price": lambda f: f.get("price") or 10**12,
            "duration": lambda f: f.get("total_duration") or 10**9,
            "departure": lambda f: ((f.get("flights") or [{}])[0].get("departure_airport") or {}).get("time") or "9999"}
    # 정렬과 상관없이 '가장 싼 편'과 '가장 일찍 출발하는 편'을 한 줄씩 요약해 준다
    def _brief(f):
        legs = f.get("flights") or [{}]
        dep_t = (legs[0].get("departure_airport") or {}).get("time", "")
        price = f"{f['price']:,}원" if isinstance(f.get("price"), (int, float)) else "가격 없음"
        return f"{price}, {'/'.join(dict.fromkeys(l.get('airline', '') for l in legs))}, 출발 {dep_t}, 소요 {_fmt_minutes(f.get('total_duration'))}, {'직항' if len(legs) == 1 else str(len(legs) - 1) + '회 경유'}"
    summary = [f"요약: 가장 싼 편 = {_brief(min(flights, key=keys['price']))}",
               f"요약: 가장 빨리 도착(소요 최단) = {_brief(min(flights, key=keys['duration']))}",
               f"요약: 가장 일찍 출발 = {_brief(min(flights, key=keys['departure']))}"]
    header_at = len(lines) - (2 if ret_date else 1)   # 요약은 표 머리글(왕복이면 주의 문구) 바로 앞에 넣는다
    lines[header_at:header_at] = summary
    flights = sorted(flights, key=keys.get(sort, keys["price"]))[: max(1, min(int(args.get("max_results") or 8), 15))]
    lines.insert(1, "정렬: " + {"price": "가격 싼 순", "duration": "총 소요 시간 짧은 순", "departure": "출발 시각 빠른 순"}.get(sort, "가격 싼 순"))
    for f in flights:
        legs = f.get("flights") or []
        if not legs:
            continue
        first, last = legs[0], legs[-1]
        stops = len(legs) - 1
        via = "직항" if stops == 0 else f"{stops}회 경유(" + "·".join(l.get("id", "") for l in f.get("layovers") or []) + ")"
        lines.append(",".join(str(x) for x in [
            f"{f['price']:,}".replace(",", "") if isinstance(f.get("price"), (int, float)) else "가격 없음",
            "/".join(dict.fromkeys(l.get("airline", "") for l in legs)),
            "/".join(l.get("flight_number", "") for l in legs),
            f"{first.get('departure_airport', {}).get('id', '')} {first.get('departure_airport', {}).get('time', '')}",
            f"{last.get('arrival_airport', {}).get('id', '')} {last.get('arrival_airport', {}).get('time', '')}",
            _fmt_minutes(f.get("total_duration")), via]))
    return "\n".join(lines)


SEARCH_FLIGHTS_FUNC = {
    "name": "search_flights",
    "description": "날짜와 노선을 넣어 Google 항공권 검색 결과(항공사, 편명, 출발·도착 시각, 소요 시간, 경유, 원화 가격, 보통 가격대)를 조회한다. "
                   "특정 날짜의 항공권 가격·시간표 질문에는 web_search 보다 이 도구를 먼저 쓴다. 결과는 최대 1시간 전 데이터일 수 있다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "departure": {"type": "string", "description": "출발 공항 IATA 코드. 여러 공항은 쉼표로 (예: ICN 또는 ICN,GMP). '한국에서'는 ICN"},
            "arrival": {"type": "string", "description": "도착 공항 IATA 코드. 도시는 공항들을 쉼표로 (예: 도쿄=NRT,HND, 오사카=KIX, 파리=CDG,ORY, 뉴욕=JFK,EWR,LGA). 나라만 말하면 대표 국제공항 (예: 아르헨티나=EZE, 브라질=GRU, 호주=SYD, 베트남=SGN,HAN)"},
            "outbound_date": {"type": "string", "description": "가는 날 YYYY-MM-DD"},
            "return_date": {"type": "string", "description": "오는 날 YYYY-MM-DD (왕복일 때만, 편도면 생략)"},
            "adults": {"type": "integer", "description": "성인 인원 (기본 1)"},
            "travel_class": {"type": "string", "enum": list(TRAVEL_CLASS), "description": "좌석 등급 (기본 economy)"},
            "max_results": {"type": "integer", "description": "보여줄 항공편 수 (기본 8, 최대 15)"},
            "sort_by": {"type": "string", "enum": ["price", "duration", "departure"],
                        "description": "정렬: price=가장 싼 편(기본), duration=가장 빨리 도착하는(소요 시간 짧은) 편, departure=가장 일찍 출발하는 편"},
        },
        "required": ["departure", "arrival", "outbound_date"],
    },
}


WEB_SEARCH_FUNC = {
    "name": "web_search",
    "description": "인터넷 검색. 항공권 가격, 운항 시간표, 노선별 소요 시간, 공항·항공사 최신 소식처럼 대시보드 데이터에 없는 최신 정보를 찾을 때 쓴다. "
                   "한국 노선은 한국어로 검색하면 국내 예약 사이트 결과가 잘 나온다. 결과에는 출처 주소가 함께 온다.",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string", "description": "검색어 (예: 인천 도쿄 항공권 가격)"}},
                     "required": ["query"]},
}

TOOL_FUNCS = {"list_tables": tool_list_tables, "query_table": tool_query_table, "get_ldh_summary": tool_get_ldh_summary,
              "web_search": tool_web_search, "search_flights": tool_search_flights}

FLIGHT_RULE = """
항공권 조회(search_flights) - 이 규칙이 위의 '3. 최신 정보' 보다 우선한다:
- 항공권 가격·출발/도착 시각·소요 시간·항공편 질문에는 날짜가 없어도 반드시 search_flights 를 쓴다. 이런 질문에 web_search 를 쓰지 않는다.
- 작은 도시·작은 공항은 처음부터 가까운 큰 공항을 쉼표로 함께 넣어 조회한다 (예: 시우다드델에스테=AGT,IGU,ASU, 제주 근처가 아니라 해외 소도시도 마찬가지). '결과 없음'이 오면 포기하지 말고 가까운 큰 공항으로 넓혀 다시 조회한 뒤, 그 공항에서 목적지까지 가는 방법(국내선·버스·국경 이동)을 한 줄로 덧붙인다.
- '지금', '당장', '오늘', '가장 빠른 비행기'처럼 날짜가 없거나 바로 떠나는 표를 찾으면 오늘 날짜로 조회한다. 오늘 결과가 없으면 내일 날짜로 한 번 더 조회한다.
- 날짜를 전혀 말하지 않은 가격 질문은 7일 뒤 날짜로 조회하고 그 날짜를 밝힌다.
- '가장 빠른'은 보통 '가장 빨리 도착하는(소요 시간이 짧은) 편'이므로 sort_by=duration 으로 조회하고, 가장 일찍 출발하는 편도 함께 알려준다. '가장 싼'은 sort_by=price.
- 답 끝에는 결과의 '요약:' 줄을 이용해 '가장 싼 편'과 '가장 일찍 출발하는 편'을 한 줄씩 덧붙인다 (표에 이미 있으면 생략).
- '다음 주 금요일'처럼 상대 날짜는 아래 '오늘 날짜'를 기준으로 YYYY-MM-DD 로 바꾼다.
- 왕복 결과에는 가는 편 시각만 있다. 오는 편 시각을 지어내지 말고 '오는 편 시간은 예약 사이트에서 선택'이라고 안내한다.
- 항공편 전체 표(가격·항공사·편명·출발·도착·소요·경유)는 화면이 답 아래에 자동으로 붙인다. 답에서 표를 다시 만들지 말고,
  가장 빨리 도착하는 편·가장 싼 편·가장 일찍 출발하는 편을 출발→도착 시각, 소요 시간, 가격과 함께 2~4줄로 요약한다.
- 끝에 'Google 항공권 검색 결과이며 예약 시 가격이 바뀔 수 있다'고 밝힌다. 예약·결제는 할 수 없다."""


def _extra_tools_and_rule():
    """키가 있는 선택 도구(항공권 조회)와 오늘 날짜 안내"""
    import datetime

    tools = [SEARCH_FLIGHTS_FUNC] if _secret("SERPAPI_API_KEY") else []
    rule = (FLIGHT_RULE if tools else "") + f"\n\n오늘 날짜: {datetime.date.today():%Y-%m-%d} ({'월화수목금토일'[datetime.date.today().weekday()]}요일)"
    return tools, rule

TOOLS = [
    {
        "name": "list_tables",
        "description": "조회할 수 있는 표 이름, 설명, 칸 이름과 형식, 행 수, 기간, 그리고 get_ldh_summary 에 쓸 카테고리 목록을 돌려준다. "
                       "어떤 칸이 있는지 모르면 먼저 부른다.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "query_table",
        "description": "표 하나를 조건으로 거르고(filters), 묶어서 집계하고(group_by + metrics), 정렬해(sort_by) 최대 50행을 CSV 로 돌려준다. "
                       "집계하면 결과 칸 이름은 '<칸>_<agg>' (예: passengers_sum) 이므로 sort_by 에 그 이름을 쓴다. "
                       "공항 이름을 모르면 airport_name 에 contains 필터로 찾는다. 결과 첫 줄에 전체 행 수가 나온다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "enum": list(TABLES)},
                "filters": {
                    "type": "array",
                    "description": "모두 만족하는 행만 남긴다 (AND).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "op": {"type": "string", "enum": sorted(OPS)},
                            "value": {"type": "string", "description": "비교 값 (숫자도 문자열로, 예: \"2025\"). op 가 in 이면 쉼표로 구분 (예: \"ICN,GMP\")."},
                        },
                        "required": ["column", "op", "value"],
                    },
                },
                "group_by": {"type": "array", "items": {"type": "string"}},
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"column": {"type": "string"}, "agg": {"type": "string", "enum": sorted(AGGS)}},
                        "required": ["column", "agg"],
                    },
                },
                "columns": {"type": "array", "items": {"type": "string"}, "description": "집계하지 않을 때 보여줄 칸 (생략하면 전체)."},
                "sort_by": {"type": "string"},
                "descending": {"type": "boolean", "description": "기본 true (큰 값부터)."},
                "limit": {"type": "integer", "description": "돌려줄 행 수, 1~50 (기본 20)."},
            },
            "required": ["table"],
        },
    },
    {
        "name": "get_ldh_summary",
        "description": "팀원(ldh) 분석 카테고리의 결과 요약: 한 줄 요약, 주요 지표, 주의 사항, 모델별 성능(최대 15개). "
                       "인천 지연(korea_ops), 한국 수요(korea_demand), 항공료(korea_airfare), 미국 운항(us_flights), 미국 기상(us_weather), "
                       "유럽(europe), 전 세계(world_traffic), 항공 안전(safety) 질문에 쓴다.",
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string", "description": "카테고리 slug (예: korea_ops)"}},
            "required": ["category"],
        },
    },
]

# 서버에서 Anthropic 이 실행하는 웹 검색 도구 (항공권 가격·운항 시간표처럼 우리 데이터에 없는 최신 정보용)
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 3}

SYSTEM = """너는 'AERO INSIGHT' 항공 대시보드의 AI 도우미 '에어로'다. 사용자는 한국어로 묻고, 너는 친절한 한국어로 답한다.

답할 수 있는 범위:
1. 이 대시보드의 데이터: 공항 이용객·예측, 실시간 수집, 팀원 분석 결과 -> query_table / get_ldh_summary 로 조회한다.
2. 항공 일반 지식: 기종, 공항, 항공사, 항공 용어, 여행 상식 -> 알고 있는 지식으로 답한다.
3. 최신 정보: 항공권 가격, 운항 시간표, 노선별 소요 시간, 공항 운영 소식 -> web_search 로 찾아본다.
4. 사진: 사용자가 올린 비행기·공항 사진의 기종, 항공사 도장, 장소 등을 설명한다. 확실하지 않으면 그렇게 말한다.

규칙:
- 대시보드 데이터의 숫자는 반드시 도구로 조회한 값만 쓴다. 칸 이름이나 공항 코드를 모르면 list_tables 나 contains 필터로 먼저 확인한다.
- 항공권 가격은 날짜·좌석·예약 시점에 따라 계속 바뀐다. 검색으로 찾은 가격은 '검색 시점의 참고 가격'이라고 밝히고, 실제 예약은 항공사나 예약 사이트에서 확인하라고 안내한다. 출처 사이트 이름을 함께 쓴다.
- 너는 예약이나 결제를 할 수 없다.
- 답은 짧게: 핵심 결론 한두 문장, 필요하면 마크다운 표(10행 이내). 대시보드 데이터를 썼다면 끝에 어떤 표·조건으로 계산했는지 한 줄.
- 큰 숫자는 읽기 쉽게 쓴다 (예: 7,123만 명).
- 예측값(airport_forecast)은 모델 추정치라고 밝힌다. 공항마다 마지막 실측 달이 달라 기간 비교 시 주의한다.
- 데이터 출처 차이: 미국은 BTS T-100, 유럽은 Eurostat, 영국은 CAA, 한국은 에어포탈이라 집계 기준이 조금씩 다르다."""


SYSTEM_GEMINI = SYSTEM.replace(
    "3. 최신 정보: 항공권 가격, 운항 시간표, 노선별 소요 시간, 공항 운영 소식 -> web_search 로 찾아본다.",
    "3. 최신 정보: 너는 인터넷 검색을 할 수 없다. 노선별 평균 비행시간처럼 잘 알려진 사실은 일반 지식으로 '대략'이라고 밝히고 답한다. "
    "항공권 가격·오늘의 시간표처럼 계속 바뀌는 정보는 알 수 없다고 말하고, 보통 가격대가 알려져 있으면 '참고용 대략적 범위'라고만 말한 뒤 "
    "항공사 홈페이지나 네이버 항공권·스카이스캐너 같은 예약 사이트에서 확인하라고 안내한다.",
).replace(
    "- 항공권 가격은 날짜·좌석·예약 시점에 따라 계속 바뀐다. 검색으로 찾은 가격은 '검색 시점의 참고 가격'이라고 밝히고, 실제 예약은 항공사나 예약 사이트에서 확인하라고 안내한다. 출처 사이트 이름을 함께 쓴다.\n",
    "",
)


def run_tool(name, args):
    """(결과 문자열, 오류 여부)"""
    try:
        return TOOL_FUNCS[name](args or {}), False
    except Exception as e:
        return f"오류: {e}", True


# ---------------------------------------------------------
# 3. Claude 호출
# ---------------------------------------------------------
SECRETS_FILE = REPO_DIR / ".streamlit" / "secrets.toml"


@st.cache_data(show_spinner=False)
def _secrets_file(mtime):
    import tomllib
    try:
        return tomllib.loads(SECRETS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _secret(name):
    """키 찾기: st.secrets -> secrets.toml 파일을 직접 읽기 -> 환경변수.
    앱을 켠 뒤에 secrets.toml 에 키를 추가해도 재시작 없이 읽히도록 파일도 직접 확인한다."""
    import os
    try:
        key = st.secrets.get(name)
    except Exception:
        key = None
    if not key and SECRETS_FILE.exists():
        key = _secrets_file(SECRETS_FILE.stat().st_mtime).get(name)
    return key or os.environ.get(name)


def _api_key():
    return _secret("ANTHROPIC_API_KEY")


def _provider():
    """('gemini' | 'claude', 키) 또는 (None, None). 둘 다 있으면 무료로 쓸 수 있는 Gemini 를 쓴다."""
    gemini = _secret("GEMINI_API_KEY") or _secret("GOOGLE_API_KEY")
    if gemini:
        return "gemini", gemini
    claude = _api_key()
    return ("claude", claude) if claude else (None, None)


@st.cache_resource(show_spinner=False)
def _client(key):
    import anthropic
    return anthropic.Anthropic(api_key=key)


def _sources(content):
    """웹 검색 결과 블록에서 (제목, 주소) 목록을 뽑는다"""
    out = []
    for b in content:
        if b.type == "web_search_tool_result" and isinstance(b.content, list):
            out += [(r.title, r.url) for r in b.content if getattr(r, "url", None)]
    return out


def ask_claude(api_messages, on_tool):
    """도구를 부르며 답이 끝날 때까지 반복한다. api_messages 에 이번 턴의 assistant/tool 결과가 덧붙는다.
    반환: (최종 답 텍스트, 웹 출처 목록, 오류 메시지 또는 None)"""
    import anthropic

    client = _client(_api_key())
    sources = []
    extra_tools, extra_rule = _extra_tools_and_rule()
    for _ in range(MAX_TOOL_ROUNDS + 1):
        try:
            resp = client.beta.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=SYSTEM + extra_rule,
                tools=TOOLS + extra_tools + [WEB_SEARCH_TOOL],
                messages=api_messages,
                output_config={"effort": "medium"},
                cache_control={"type": "ephemeral"},          # 같은 system·tools 를 매번 다시 읽지 않게 캐시
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",                          # 안전 필터에 막히면 서버가 다른 모델로 다시 시도
            )
        except anthropic.AuthenticationError:
            return "", [], "API 키가 올바르지 않습니다. secrets.toml 의 ANTHROPIC_API_KEY 를 확인해주세요."
        except anthropic.RateLimitError:
            return "", [], "요청이 너무 많습니다. 잠시 뒤 다시 시도해주세요."
        except anthropic.BadRequestError as e:
            return "", [], f"요청 오류: {e.message}"
        except anthropic.APIStatusError as e:
            return "", [], f"API 오류 ({e.status_code}): {e.message}"
        except anthropic.APIConnectionError:
            return "", [], "Claude API 에 연결할 수 없습니다. 인터넷 연결을 확인해주세요."

        # thinking·검색 결과 블록도 다음 요청에 그대로 돌려줘야 하므로 응답 전체를 저장한다
        api_messages.append({"role": "assistant", "content": [b.to_dict() for b in resp.content]})
        sources += _sources(resp.content)
        for b in resp.content:
            if b.type == "server_tool_use" and b.name == "web_search":
                on_tool("web_search", b.input or {}, "", False)
        text = "\n".join(b.text for b in resp.content if b.type == "text").strip()

        if resp.stop_reason == "refusal":
            return text, sources, "이 질문에는 답할 수 없습니다. 질문을 바꿔서 다시 시도해주세요."
        if resp.stop_reason == "pause_turn":       # 서버 도구(웹 검색)가 길어져 잠시 멈춘 경우: 그대로 이어서 요청
            continue
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason != "tool_use" or not tool_uses:
            if resp.stop_reason == "max_tokens":
                text += "\n\n_(답이 길어 중간에 잘렸습니다.)_"
            return text, sources, None

        results = []
        for tu in tool_uses:
            out, is_err = run_tool(tu.name, tu.input)
            on_tool(tu.name, tu.input, out, is_err)
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": out, "is_error": is_err})
        api_messages.append({"role": "user", "content": results})
    return "", sources, f"도구 호출이 {MAX_TOOL_ROUNDS}번을 넘어 멈췄습니다. 질문을 더 구체적으로 해주세요."


@st.cache_resource(show_spinner=False)
def _gemini_client(key):
    from google import genai
    return genai.Client(api_key=key)


FLIGHT_WORDS = re.compile(r"항공권|비행기\s*표|비행기\s*티켓|항공편|비행편|티켓|최저가|편도|왕복|직항|항공\s*요금|flight|ticket", re.I)


def is_flight_question(text):
    """항공권 가격·시간표를 묻는 질문인지 (작은 모델이 웹 검색으로 빠지지 않도록 코드가 먼저 판단한다)"""
    return bool(FLIGHT_WORDS.search(text or "")) or ("비행기" in (text or "") and bool(re.search(r"가는|가격|얼마|빠른|싼|시간", text)))


def ask_gemini(history, parts, on_tool, force_flights=False):
    """Gemini 로 도구를 부르며 답이 끝날 때까지 반복한다.
    history: 지금까지의 대화 (google.genai Content 목록). 반환: (답, 새 history, 웹 출처 목록, 오류 메시지 또는 None)
    force_flights: 항공권 질문이면 첫 단계에서 search_flights 를 반드시 부르게 한다 (결과 없음이면 한 번 더)."""
    from google.genai import errors, types

    searchable = bool(_secret("TAVILY_API_KEY"))
    extra_tools, extra_rule = _extra_tools_and_rule()
    web = dict(WEB_SEARCH_FUNC)
    if extra_tools:  # 항공권 조회가 되면 가격·시간표는 그쪽으로 보낸다
        web["description"] = ("인터넷 검색. 공항·항공사 최신 소식, 입국 규정, 수하물 규정처럼 대시보드 데이터에 없는 최신 정보를 찾을 때 쓴다. "
                              "항공권 가격·출발/도착 시각·소요 시간은 이 도구가 아니라 search_flights 로 조회한다.")
    tools = TOOLS + extra_tools + ([web] if searchable else [])
    decls = [types.FunctionDeclaration(name=t["name"], description=t["description"], parameters_json_schema=t["input_schema"])
             for t in tools]
    sources = []
    config = types.GenerateContentConfig(
        system_instruction=(SYSTEM if searchable else SYSTEM_GEMINI) + extra_rule,   # 검색이 되면 Claude 와 같은 안내문(web_search 사용)
        tools=[types.Tool(function_declarations=decls)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),   # 도구 실행은 직접 해서 화면에 보여준다
    )
    # chat 이 모델 응답(생각 서명 포함)을 그대로 history 에 쌓아 다음 요청에 돌려준다
    chat = _gemini_client(_provider()[1]).chats.create(model=GEMINI_MODEL, config=config, history=history)
    message = parts
    force = force_flights and any(t["name"] == "search_flights" for t in tools)
    forced_config = config.model_copy(update={"tool_config": types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode="ANY", allowed_function_names=["search_flights"]))})
    try:
        for round_no in range(MAX_TOOL_ROUNDS + 1):
            resp = chat.send_message(message, config=forced_config if force else None)
            calls = resp.function_calls or []
            if not calls:
                return (resp.text or "").strip(), chat.get_history(), sources, None
            message, flight_found = [], False
            for fc in calls:
                args = dict(fc.args or {})
                out, is_err = run_tool(fc.name, args)
                on_tool(fc.name, args, out, is_err)
                if fc.name == "web_search" and not is_err:
                    sources += _parse_sources(out)
                if fc.name == "search_flights" and not is_err and out.startswith("Google 항공권 검색 결과"):
                    flight_found = True
                message.append(types.Part.from_function_response(name=fc.name, response={"error": out} if is_err else {"result": out}))
            # 첫 조회에서 항공편을 못 찾았으면(결과 없음·오류) 가까운 공항으로 넓혀 한 번 더 조회하게 한다. 그 뒤에는 모델이 자유롭게 답한다
            force = force and round_no == 0 and not flight_found
    except errors.ClientError as e:
        if e.code == 429:
            return "", history, [], "무료 사용 한도를 넘었습니다. 잠시(1분) 뒤 다시 시도하거나 내일 다시 써주세요."
        if e.code in (400, 401, 403) and "key" in str(e).lower():
            return "", history, [], "Gemini API 키가 올바르지 않습니다. secrets.toml 의 GEMINI_API_KEY 를 확인해주세요."
        return "", history, [], f"Gemini 요청 오류 ({e.code}): {e.message}"
    except errors.ServerError as e:
        return "", history, [], f"Gemini 서버 오류 ({e.code}). 잠시 뒤 다시 시도해주세요."
    except errors.APIError as e:
        return "", history, [], f"Gemini 오류: {e}"
    return "", history, [], f"도구 호출이 {MAX_TOOL_ROUNDS}번을 넘어 멈췄습니다. 질문을 더 구체적으로 해주세요."


def _image_block(file):
    """올린 사진을 긴 변 1568px 이하 JPEG 로 줄여 Claude 이미지 블록으로 만든다 (토큰 절약)"""
    import base64
    import io

    from PIL import Image

    img = Image.open(file)
    img = img.convert("RGB")
    img.thumbnail((1568, 1568))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}, buf.getvalue()


# ---------------------------------------------------------
# 4. 화면: 모든 페이지 오른쪽 아래에 떠 있는 채팅 버튼
# ---------------------------------------------------------
TOOL_LABELS = {"list_tables": "표 목록 확인", "query_table": "표 조회", "get_ldh_summary": "팀원 결과 요약 조회", "web_search": "웹 검색", "search_flights": "항공권 조회"}

GREETING = "안녕하세요, 항공 도우미 에어로예요 ✈️\n\n공항 이용객·예측, 항공권 가격, 소요 시간, 비행기 사진까지 궁금한 건 뭐든 물어보세요!"
QUICK = ["2025년 이용객 TOP 5 공항은?", "다음 주 금요일 인천→도쿄 편도 최저가는?", "인천→파리 비행시간은?", "인천 도착편 지연 예측 요약"]

CSS = """
<style>
/* 오른쪽 아래 고정 버튼 */
.st-key-ai_fab { position: fixed !important; right: 26px; bottom: 26px; z-index: 999990; width: auto !important; }
.st-key-ai_fab button {
  width: 64px; height: 64px; border-radius: 50% !important; border: none !important; padding: 0 !important;
  background: linear-gradient(135deg, #2563eb, #7c3aed) !important; color: #fff !important;
  box-shadow: 0 10px 28px rgba(37, 99, 235, .45); transition: transform .15s ease;
}
.st-key-ai_fab button:hover { transform: scale(1.07); }
.st-key-ai_fab button p { font-size: 1.75rem !important; line-height: 1; }
.st-key-ai_fab button [data-testid="stIconMaterial"] { display: none; }   /* 펼침 화살표 숨김 */
/* 말풍선: 페이지를 열면 잠깐 나타났다가 사라진다 */
.ai-hello {
  position: fixed; right: 100px; bottom: 40px; z-index: 999989; pointer-events: none;
  background: #ffffff; color: #1e293b; border: 1px solid #e2e8f0; border-radius: 16px 16px 4px 16px;
  padding: .6rem .9rem; font-size: .88rem; font-weight: 600; box-shadow: 0 8px 24px rgba(15, 23, 42, .14);
  opacity: 0; animation: ai-hello 9s ease 1.2s 1 forwards;
}
@keyframes ai-hello { 0% {opacity:0; transform: translateY(8px);} 8% {opacity:1; transform:none;} 85% {opacity:1;} 100% {opacity:0;} }
/* 펼친 채팅 창 크기 */
[data-testid="stPopoverBody"]:has(.ai-chat-head) { width: min(600px, 94vw) !important; max-width: 94vw; }
/* 답 안의 표: 글자를 조금 줄이고, 그래도 넓으면 표만 가로로 스크롤 (잘리지 않게) */
[data-testid="stPopoverBody"]:has(.ai-chat-head) [data-testid="stMarkdownContainer"] table {
  display: block; max-width: 100%; overflow-x: auto; font-size: .8rem; border-collapse: collapse;
}
/* 답 안의 제목(#, ##)은 채팅 창에 맞게 작게 */
[data-testid="stPopoverBody"]:has(.ai-chat-head) [data-testid="stChatMessage"] :is(h1, h2, h3, h4) {
  font-size: 1rem !important; padding: .4rem 0 .2rem; margin: 0;
}
[data-testid="stPopoverBody"]:has(.ai-chat-head) [data-testid="stMarkdownContainer"] th,
[data-testid="stPopoverBody"]:has(.ai-chat-head) [data-testid="stMarkdownContainer"] td { white-space: nowrap; padding: .3rem .5rem; }
.ai-chat-head { font-weight: 800; font-size: 1.02rem; margin-bottom: .1rem; }
.ai-chat-head span { font-size: .75rem; font-weight: 600; color: #64748b; margin-left: .35rem; }
</style>
"""


def _flight_table(text):
    """search_flights 결과 글에서 (제목, 표) 를 만든다. 표 부분이 없으면 None"""
    import io

    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith(("가격(원),", "왕복 합계 가격(원),"))), None)
    if start is None:
        return None
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), dtype=str).fillna("")
    price_col = df.columns[0]
    df[price_col] = df[price_col].map(lambda v: f"{int(float(v)):,}원" if v.replace(".", "").isdigit() else v)
    for col in df.columns:
        if "출발" in col or "도착" in col:   # 'ICN 2026-11-16 23:40' -> 'ICN 11-16 23:40'
            df[col] = df[col].str.replace(r"(\b[A-Z]{3}) \d{4}-(\d{2}-\d{2})", r"\1 \2", regex=True)
    df.columns = [c.replace("가는 편 ", "").replace("(원)", "").replace("왕복 합계 가격", "왕복 가격") for c in df.columns]
    if "경유" in df:
        df["경유"] = df["경유"].str.replace("회 경유", "회", regex=False)
    if "항공사" in df:   # '에미레이트항공/GOL 브라질항공' -> '에미레이트/GOL 브라질'
        df["항공사"] = df["항공사"].str.replace("항공", "", regex=False)
    # 좁은 채팅 창에서도 중요한 칸이 먼저 보이게: 출발·도착·소요·가격·경유, 그다음 항공사·편명
    first = [c for c in ("출발", "도착", "총 소요", "소요") if c in df.columns]
    rest = [c for c in df.columns if c not in first and c not in ("항공사", "편명")]
    df = df[first + rest + [c for c in ("항공사", "편명") if c in df.columns]]
    head = lines[0].replace("Google 항공권 검색 결과 · ", "")
    sort = next((l.replace("정렬: ", "") for l in lines if l.startswith("정렬: ")), "")
    note = next((l for l in lines if l.startswith("주의: ")), "")
    return f"✈️ {head}" + (f" · {sort}" if sort else "") + (f"  \n{note.replace('주의: ', '※ ')}" if note else ""), df


def _render_flight_tables(steps):
    """항공권 조회 결과는 AI 가 고른 칸이 아니라 항상 같은 칸으로 코드가 직접 표를 그린다"""
    for s in steps:
        if s["name"] != "search_flights" or s["error"]:
            continue
        made = _flight_table(s["result"])
        if made:
            title, df = made
            st.caption(title)
            st.dataframe(df, hide_index=True, use_container_width=True, height=min(38 + 35 * len(df), 320))


def _render_steps(steps, sources=()):
    if not steps and not sources:
        return
    with st.expander(f"조회한 내용 {len(steps)}건", icon=":material/database:"):
        for s in steps:
            mark = "⚠️ " if s["error"] else ""
            st.markdown(f"**{mark}{TOOL_LABELS.get(s['name'], s['name'])}** `{s['caption']}`")
            if s["result"]:
                st.code(s["result"][:2000], language="text")
        for title, url in list(dict.fromkeys(sources))[:6]:
            st.markdown(f"- [{title or url}]({url})")


def _tool_caption(name, args):
    if name == "query_table":
        parts = [args.get("table", "")]
        for f in args.get("filters") or []:
            parts.append(f"{f.get('column')} {f.get('op')} {f.get('value')}")
        if args.get("group_by"):
            parts.append("묶기: " + ", ".join(args["group_by"]))
        if args.get("sort_by"):
            parts.append(f"정렬: {args['sort_by']}")
        return " · ".join(map(str, parts))
    if name == "get_ldh_summary":
        return args.get("category", "")
    if name == "web_search":
        return args.get("query", "")
    if name == "search_flights":
        return f"{args.get('departure')}→{args.get('arrival')} {args.get('outbound_date')}" + (f"~{args['return_date']}" if args.get("return_date") else "")
    return ""


@st.fragment
def _chat_panel():
    """채팅 창 내용. fragment 라서 질문을 보내도 뒤의 페이지 전체가 다시 그려지지 않는다."""
    ss = st.session_state
    ss.setdefault("ai_api_messages", [])   # Claude 에 보내는 전체 대화 (도구 결과 포함)
    ss.setdefault("ai_gemini_history", [])  # Gemini 에 보내는 전체 대화
    ss.setdefault("ai_display", [])        # 화면에 보여줄 대화

    head = st.columns([5, 1], vertical_alignment="center")
    head[0].markdown('<div class="ai-chat-head">✈️ 에어로 <span>AI 항공 도우미</span></div>', unsafe_allow_html=True)
    if head[1].button("", icon=":material/delete:", key="ai_clear", help="대화 지우기", disabled=not ss.ai_display):
        ss.ai_api_messages, ss.ai_gemini_history, ss.ai_display = [], [], []
        st.rerun(scope="fragment")

    provider, _ = _provider()
    if provider is None:
        st.warning("AI 키가 없습니다. https://aistudio.google.com/apikey 에서 무료 Gemini 키를 만들고, "
                   "저장소 최상위 `.streamlit/secrets.toml` 에 `GEMINI_API_KEY = \"...\"` 를 추가한 뒤 새로고침하세요. "
                   "(secrets.toml 은 git 에 올라가지 않습니다)", icon=":material/key:")
        return
    st.caption(f"{'Gemini · ' + GEMINI_MODEL if provider == 'gemini' else 'Claude · ' + MODEL}"
               + (" · 무료 등급은 입력 내용이 Google 제품 개선에 쓰일 수 있어요" if provider == "gemini" else ""))

    box = st.container(height=400, border=False)
    with box:
        with st.chat_message("assistant", avatar="✈️"):
            st.markdown(GREETING)
        for m in ss.ai_display:
            with st.chat_message(m["role"], avatar="✈️" if m["role"] == "assistant" else None):
                for img in m.get("images", []):
                    st.image(img, width=220)
                st.markdown(m["text"])
                _render_flight_tables(m.get("steps", []))
                _render_steps(m.get("steps", []), m.get("sources", []))

    question, files = None, []
    if not ss.ai_display:
        cols = st.columns(2)
        for i, q in enumerate(QUICK):
            if cols[i % 2].button(q, key=f"ai_quick_{i}", use_container_width=True):
                question = q
    sent = st.chat_input("질문하거나 사진을 올려보세요", accept_file=True,
                         file_type=["jpg", "jpeg", "png", "webp"], key="ai_input")
    if sent:
        question, files = (sent.text or "").strip(), list(sent.files or [])
        if not question and files:
            question = "이 사진을 설명해줘."
    if not question:
        return

    content, shown = [], []
    for f in files[:3]:
        try:
            block, jpeg = _image_block(f)
            content.append(block)
            shown.append(jpeg)
        except Exception:
            st.toast(f"{f.name} 사진을 읽지 못했습니다.", icon="⚠️")
    content.append({"type": "text", "text": question})
    if provider == "gemini":
        from google.genai import types
        gemini_parts = [types.Part.from_bytes(data=j, mime_type="image/jpeg") for j in shown] + [types.Part.from_text(text=question)]

    ss.ai_display.append({"role": "user", "text": question, "images": shown})
    with box:
        with st.chat_message("user"):
            for img in shown:
                st.image(img, width=220)
            st.markdown(question)

    start = len(ss.ai_api_messages)   # 실패하면 이번 질문을 대화에서 뺀다
    if provider == "claude":
        ss.ai_api_messages.append({"role": "user", "content": content})
    steps = []
    with box:
        with st.chat_message("assistant", avatar="✈️"):
            with st.status("생각하는 중...", expanded=False) as status:
                def on_tool(name, args, out, is_err):
                    steps.append({"name": name, "caption": _tool_caption(name, args), "result": out, "error": is_err})
                    status.update(label=f"{TOOL_LABELS.get(name, name)}: {_tool_caption(name, args)}"[:80])
                if provider == "gemini":
                    answer, new_history, sources, error = ask_gemini(ss.ai_gemini_history, gemini_parts, on_tool,
                                                                    force_flights=is_flight_question(question))
                    if not error:
                        ss.ai_gemini_history = new_history
                else:
                    answer, sources, error = ask_claude(ss.ai_api_messages, on_tool)
                status.update(label="완료" if not error else "멈춤", state="complete" if not error else "error")
            if error:
                del ss.ai_api_messages[start:]
                ss.ai_display.pop()
                st.error(error)
                return
    ss.ai_display.append({"role": "assistant", "text": answer or "_(빈 답)_", "steps": steps, "sources": sources})
    st.rerun(scope="fragment")    # 답을 대화 목록에 넣은 상태로 다시 그린다


def floating_chat():
    """app.py 에서 모든 페이지 마지막에 부른다: 오른쪽 아래 채팅 버튼 + 인사 말풍선"""
    st.markdown(CSS, unsafe_allow_html=True)
    if not st.session_state.get("ai_display"):
        st.markdown('<div class="ai-hello">저 여기 있어요~ 궁금한 거 물어보세요! 👋</div>', unsafe_allow_html=True)
    with st.container(key="ai_fab"):
        with st.popover("💬", help="AI 항공 도우미에게 물어보기"):
            _chat_panel()
