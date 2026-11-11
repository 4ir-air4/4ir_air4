"""인천공항 운항 자료 수집기. 운항 API 는 당일 자료만 주므로 정기 실행해 쌓는다.

키는 환경변수 DATA_GO_KR_API_KEY 또는 .env 에 둔다. .env 는 git 제외 대상이다.

    python collect_icn.py --check      연결과 응답 항목 확인 (저장하지 않음)
    python collect_icn.py              도착, 출발 현황 수집 (10분마다 권장)
    python collect_icn.py --closing    운항마감 수집 (당일 자료뿐이므로 1시간마다 권장)
    python collect_icn.py --backup     수집 자료를 구글 드라이브로 복사 (1시간마다 권장)
    python collect_icn.py --status     최근 수집 상태와 공백 확인

결과는 data/icn_live/<날짜>/<자료>.jsonl 에 받은 항목 그대로 쌓는다. 값이 바뀐 때만 줄을 추가하므로
각 시점에 무엇을 알 수 있었는지 fetched_at 으로 재현할 수 있다. 하나라도 실패하면 종료 코드 1 을 돌려준다.
현황 자료의 시각은 날짜 없는 HHMM 이고 공동운항(codeshare=Slave)도 함께 온다. 원본은 그대로 저장하고 변환할 때 처리한다.
"""
import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote

import requests

BASE = "https://apis.data.go.kr/B551177/"
PROJECT = Path(__file__).resolve().parent  # 예약 작업은 프로젝트 밖에서 시작하므로 파일 위치를 기준으로 삼는다
ROOT = PROJECT / "data" / "icn_live"
DRIVE = Path(r"G:\내 드라이브\4ir")
KST = timezone(timedelta(hours=9))
PAGE_ROWS = 1000
MAX_PAGES = 50
RETRY_WAITS = (5, 15)
DATASETS = {"arrivals": {"path": "StatusOfPassengerFlightsOdp/getPassengerArrivalsOdp", "params": {"lang": "K"}, "key": ("flightId", "scheduleDateTime", "airportCode")},
            "departures": {"path": "StatusOfPassengerFlightsOdp/getPassengerDeparturesOdp", "params": {"lang": "K"}, "key": ("flightId", "scheduleDateTime", "airportCode")},
            "closing": {"path": "FlightClosingInfoSpot/getFlightClosingInfoSpot", "params": {}, "key": ("msDate", "msIo", "flightNum", "msStime")}}


class CollectError(Exception):
    """메시지에 인증키가 들어가지 않게 만든 오류."""


def load_key(env_path=PROJECT / ".env"):
    key = os.environ.get("DATA_GO_KR_API_KEY", "").strip()
    if not key and env_path.exists():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "DATA_GO_KR_API_KEY":
                key = value.strip().strip("'\"")
    if not key:
        sys.exit("DATA_GO_KR_API_KEY 가 없습니다. 환경변수나 .env 에 공공데이터포털 인증키를 설정하세요.")
    return unquote(key)  # 포털의 인코딩 키를 넣어도 requests 가 다시 인코딩하므로 원래 값으로 되돌린다


def parse_page(text, key):
    """(항목 목록, 전체 건수)를 돌려준다. 포털은 인증 오류를 JSON 요청에도 XML 로 돌려줄 수 있다."""
    try:
        data = json.loads(text)
    except ValueError:
        raise CollectError("JSON 이 아닌 응답: " + text[:300].replace(key, "***").replace("\n", " ")) from None
    response = data.get("response", data) if isinstance(data, dict) else {}
    header = response.get("header") or {}
    code = str(header.get("resultCode", response.get("resultCode", "00")))
    if code not in ("00", "0"):
        raise CollectError(f"API 오류 {code}: {header.get('resultMsg', response.get('resultMsg', ''))}".replace(key, "***"))
    body = response.get("body") or {}
    items = body.get("items") if isinstance(body, dict) else None
    if isinstance(items, dict):
        items = items.get("item", [])
    if isinstance(items, dict):
        items = [items]
    return (items if isinstance(items, list) else []), int(body.get("totalCount") or 0) if isinstance(body, dict) else 0


def get_page(spec, key, page):
    params = {**spec["params"], "serviceKey": key, "type": "json", "numOfRows": PAGE_ROWS, "pageNo": page}
    for attempt, wait in enumerate((*RETRY_WAITS, None)):
        try:
            response = requests.get(BASE + spec["path"], params=params, timeout=60)
            if response.status_code == 404:
                raise CollectError(f"경로를 찾지 못했습니다 ({spec['path']}). 포털 상세 페이지의 호출 경로와 비교해 DATASETS 를 고치세요.")
            if 400 <= response.status_code < 500 and response.status_code != 429:
                raise CollectError(f"HTTP {response.status_code}. 활용 신청 승인과 인증키를 확인하세요.")
            if response.status_code < 400:
                return parse_page(response.text, key)
            problem = f"HTTP {response.status_code}"
        except requests.RequestException as e:
            # 예외 문구의 URL 에 serviceKey 가 들어 있으므로 종류와 상태 코드만 남긴다
            problem = f"{type(e).__name__} ({e.response.status_code if e.response is not None else '응답 없음'})"
        if wait is None:
            raise CollectError(f"요청 실패: {problem}")
        time.sleep(wait)


def fetch(spec, key):
    rows, page = [], 1
    while page <= MAX_PAGES:
        items, total = get_page(spec, key, page)
        rows += items
        # 현황 API 는 totalCount 없이 당일 전체를 한 번에 준다
        if not items or not total or len(rows) >= total:
            return rows
        page += 1
    raise CollectError(f"{MAX_PAGES}쪽을 넘었습니다. 전체 {total}건 중 {len(rows)}건만 읽었습니다.")


def row_key(row, fields):
    return json.dumps([row.get(f) for f in fields], ensure_ascii=False)


def latest_state(path, fields):
    """파일의 편별 마지막 값. 저장 중 끊겨 잘린 줄은 건너뛴다."""
    state = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            state[row_key(row["data"], fields)] = json.dumps(row["data"], ensure_ascii=False, sort_keys=True)
    return state


def append_changed(path, rows, fields, fetched_at):
    state = latest_state(path, fields)
    lines = []
    for row in rows:
        text = json.dumps(row, ensure_ascii=False, sort_keys=True)
        key = row_key(row, fields)
        if state.get(key) != text:
            state[key] = text
            lines.append(json.dumps({"fetched_at": fetched_at, "data": row}, ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    if lines:
        broken = path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n")
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(("\n" if broken else "") + "\n".join(lines) + "\n")
    return len(lines)


def log_run(root, record):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "runs.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect(names, key, root=ROOT, now=None):
    now = now or datetime.now(KST)
    failed = 0
    for name in names:
        spec = DATASETS[name]
        record = {"at": now.isoformat(timespec="seconds"), "dataset": name}
        try:
            rows = fetch(spec, key)
            added = append_changed(root / now.strftime("%Y-%m-%d") / f"{name}.jsonl", rows, spec["key"], record["at"])
            record |= {"ok": True, "rows": len(rows), "added": added}
            print(f"{record['at']} {name}: {len(rows)}건 수신, {added}건 저장", flush=True)
        except (CollectError, OSError) as e:
            failed += 1
            record |= {"ok": False, "error": str(e).replace(key, "***")}
            print(f"{record['at']} {name} 실패: {record['error']}", file=sys.stderr, flush=True)
        log_run(root, record)
    return failed


def backup(root=ROOT, drive=DRIVE, now=None):
    """원본이 더 큰 파일만 복사한다. 수집 파일은 뒤에 덧붙기만 하므로, 이 PC의 파일이 사라지거나 짧아져도 백업의 이력은 남는다."""
    record = {"at": (now or datetime.now(KST)).isoformat(timespec="seconds"), "dataset": "backup"}
    try:
        if not drive.exists():  # 드라이브 프로그램이 꺼져 있으면 폴더를 새로 만들지 않고 실패로 남긴다
            raise OSError(f"백업 폴더가 없습니다: {drive}")
        copied = 0
        for source in root.rglob("*.jsonl"):
            target = drive / "data" / "icn_live" / source.relative_to(root)
            if not target.exists() or target.stat().st_size < source.stat().st_size:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                copied += 1
        record |= {"ok": True, "added": copied}
        print(f"{record['at']} backup: {copied}개 파일 복사", flush=True)
    except OSError as e:
        record |= {"ok": False, "error": str(e)}
        print(f"{record['at']} backup 실패: {e}", file=sys.stderr, flush=True)
    log_run(root, record)
    return not record["ok"]


def check(names, key):
    failed = 0
    for name in names:
        try:
            items, total = get_page(DATASETS[name], key, 1)
            print(f"{name}: 첫 쪽 {len(items)}건, 전체 {total or '건수 미제공'}")
            if items:
                print("  항목:", sorted(items[0]))
                missing = [f for f in DATASETS[name]["key"] if f not in items[0]]
                if missing:
                    print("  구분 항목 없음, DATASETS 를 고치세요:", missing)
                    failed += 1
        except CollectError as e:
            print(f"{name} 실패: {e}", file=sys.stderr)
            failed += 1
    return failed


def status(root=ROOT, days=7, now=None):
    """날짜와 자료별 성공, 실패, 가장 긴 수집 공백(분)을 돌려준다."""
    now = now or datetime.now(KST)
    path = root / "runs.jsonl"
    runs = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                runs.append(json.loads(line))
            except ValueError:
                continue
    result = []
    for offset in range(days - 1, -1, -1):
        day = (now - timedelta(days=offset)).strftime("%Y-%m-%d")
        for name in (*DATASETS, "backup"):
            today = [r for r in runs if r["dataset"] == name and r["at"].startswith(day)]
            if not today and name in ("closing", "backup"):
                continue
            good = sorted(datetime.fromisoformat(r["at"]) for r in today if r.get("ok"))
            edges = [datetime.fromisoformat(f"{day}T00:00:00+09:00"), *good, min(now, datetime.fromisoformat(f"{day}T23:59:59+09:00"))]
            gap = max((b - a).total_seconds() / 60 for a, b in zip(edges, edges[1:]))
            result.append({"날짜": day, "자료": name, "성공": len(good), "실패": len(today) - len(good), "최대 공백(분)": round(gap),
                           "저장": sum(r.get("added", 0) for r in today), "마지막 오류": next((r["error"] for r in reversed(today) if not r.get("ok")), "")})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--closing", action="store_true")
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if stream:  # 예약 작업의 pythonw 에는 콘솔이 없다
            stream.reconfigure(encoding="utf-8")
    if args.status:
        for row in status():
            print(row)
        sys.exit(0)
    if args.backup:
        sys.exit(1 if backup() else 0)
    selected = list(DATASETS) if args.check else ["closing"] if args.closing else ["arrivals", "departures"]
    sys.exit(1 if (check if args.check else collect)(selected, load_key()) else 0)
