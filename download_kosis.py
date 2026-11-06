"""KOSIS Open API 통계표 검색, 코드 확인, 수집.

키는 환경변수 KOSIS_API_KEY 또는 .env 의 KOSIS_API_KEY=... 에 둔다. .env 는 git 제외 대상이다.

순서:
    python download_kosis.py search 소비자물가지수
    python download_kosis.py meta --org 101 --tbl <표ID> --grep 항공
    python download_kosis.py fetch --org 101 --tbl <표ID> --item <항목> --obj <품목> --recent 3 --name 시험
    python download_kosis.py fetch --org 101 --tbl <표ID> --item <항목> --obj <품목> --start 200001 --name 국내항공료

결과는 reference_data/kosis/<name>.csv 에 저장하고, 같은 이름이 있으면 덮어쓴다.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests

BASE = "https://kosis.kr/openapi/"
OUT_DIR = Path("reference_data/kosis")


def load_key(env_path=Path(".env")):
    key = os.environ.get("KOSIS_API_KEY", "").strip()
    if not key and env_path.exists():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "KOSIS_API_KEY":
                key = value.strip().strip("'\"")
    if not key:
        sys.exit("KOSIS_API_KEY 가 없습니다. 환경변수나 .env 에 설정하세요.")
    return key


def call(path, **params):
    # jsonVD=Y 가 없으면 키에 따옴표가 없는 형식으로 올 수 있다
    params.update(apiKey=load_key(), format="json", jsonVD="Y")
    try:
        response = requests.get(BASE + path, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        # 예외 문구의 URL에 apiKey가 들어 있으므로 종류와 상태 코드만 출력한다
        status = e.response.status_code if e.response is not None else "응답 없음"
        sys.exit(f"KOSIS 요청 실패: {type(e).__name__} ({status})")
    if isinstance(data, dict) and "err" in data:
        sys.exit(f"KOSIS 오류 {data['err']}: {data.get('errMsg', '')}")
    if not isinstance(data, list):
        sys.exit(f"KOSIS 응답 형식이 예상과 다릅니다: {type(data).__name__}")
    return data


def search(args):
    rows = call("statisticsSearch.do", method="getList", searchNm=args.keyword, resultCount=args.limit)
    for r in rows:
        period = f"{r.get('STRT_PRD_DE', '?')}~{r.get('END_PRD_DE', '?')}"
        print(r.get("ORG_ID"), r.get("TBL_ID"), period, r.get("TBL_NM"), sep="\t")
    print(f"{len(rows)}건", file=sys.stderr)


def meta(args):
    # shortcut: type=ITM 은 공식 가이드 표에 없어 응답 필드를 가정하지 않고 행 전체를 출력한다
    rows = call("statisticsData.do", method="getMeta", type="ITM", orgId=args.org, tblId=args.tbl)
    matched = [r for r in rows if not args.grep or args.grep in json.dumps(r, ensure_ascii=False)]
    for r in matched:
        print(json.dumps(r, ensure_ascii=False))
    print(f"전체 {len(rows)}행 중 {len(matched)}행 출력", file=sys.stderr)


def missing_months(periods):
    """YYYYMM 목록에서 처음과 끝 사이에 빠진 달을 돌려준다."""
    months = pd.PeriodIndex(sorted(set(periods)), freq="M")
    full = pd.period_range(months.min(), months.max(), freq="M")
    return [str(p) for p in full.difference(months)]


def fetch(args):
    if not args.start and not args.recent:
        sys.exit("--start 또는 --recent 를 지정하세요. 둘 다 없으면 최근 1개 시점만 옵니다.")
    params = {f"objL{i}": code for i, code in enumerate(args.obj, 1)}
    if args.recent:
        params["newEstPrdCnt"] = args.recent
    else:
        params["startPrdDe"] = args.start
        params["endPrdDe"] = args.end or pd.Timestamp.now().strftime("%Y%m")
    rows = call("Param/statisticsParameterData.do", method="getList", orgId=args.org, tblId=args.tbl,
                itmId=args.item, prdSe=args.prd, **params)
    if not rows:
        sys.exit("받은 행이 없어 저장하지 않았습니다. 코드와 기간을 확인하세요.")

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")

    blank = pd.to_numeric(df["DT"], errors="coerce").isna().sum()
    print(f"{path}: {len(df)}행, {df['PRD_DE'].min()}~{df['PRD_DE'].max()}, 숫자가 아닌 값 {blank}개")
    if args.prd == "M":
        gaps = missing_months(df["PRD_DE"])
        print(f"빠진 달 {len(gaps)}개: {gaps[:12]}" if gaps else "빠진 달 없음")


def main():
    sys.stdout.reconfigure(errors="replace")  # cp949 콘솔에 없는 문자로 중단되지 않게 한다
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("search", help="통계표 검색")
    p.add_argument("keyword")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=search)

    p = sub.add_parser("meta", help="통계표의 항목, 분류 코드 확인")
    p.add_argument("--org", required=True)
    p.add_argument("--tbl", required=True)
    p.add_argument("--grep", help="이 글자가 들어간 행만 출력")
    p.set_defaults(func=meta)

    p = sub.add_parser("fetch", help="통계 자료를 받아 CSV로 저장")
    p.add_argument("--org", required=True)
    p.add_argument("--tbl", required=True)
    p.add_argument("--item", required=True, help="항목 코드 itmId")
    p.add_argument("--obj", action="append", required=True, help="분류 코드. 분류1, 분류2 순서로 반복 지정")
    p.add_argument("--prd", default="M", help="수록주기 (M=월, Q=분기, Y=연)")
    p.add_argument("--start", help="시작 시점, 예: 200001")
    p.add_argument("--end", help="종료 시점. 생략하면 이번 달")
    p.add_argument("--recent", type=int, help="최근 N개 시점만 받기. 지정하면 --start 무시")
    p.add_argument("--name", required=True, help="저장할 파일 이름 (확장자 제외)")
    p.set_defaults(func=fetch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
