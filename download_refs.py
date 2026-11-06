"""BTS 외 항공 참고 데이터 일괄 다운로드. 출처와 이용 조건은 DATA_SOURCES.md 참고.

사용법:
    python download_refs.py              # ./reference_data 에 저장
    python download_refs.py -o D:/refs

이미 받은 파일은 건너뛰므로 중단 후 다시 실행하면 이어서 받는다.
최신본을 받으려면 해당 파일을 지우고 다시 실행한다.
"""
import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

OURAIRPORTS = "https://davidmegginson.github.io/ourairports-data/{}"
OPENFLIGHTS = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/{}"
NTSB = "https://data.ntsb.gov/avdata/FileDirectory/DownloadFile?fileID={}"
WORLDBANK = "https://api.worldbank.org/v2/en/indicator/{}?downloadformat=csv"
EUROCONTROL = "https://www.eurocontrol.int/performance/data/download/xls/{}"

FILES = {
    **{f"ourairports/{n}": OURAIRPORTS.format(n) for n in
       ["airports.csv", "runways.csv", "navaids.csv", "countries.csv", "regions.csv"]},
    **{f"openflights/{n}": OPENFLIGHTS.format(n) for n in ["airports.dat", "airlines.dat", "planes.dat", "routes.dat"]},
    **{f"ntsb/{n}": NTSB.format(n) for n in
       ["avall.zip", "Pre2008.zip", "PRE1982.zip", "eadmspub.pdf", "codman.pdf"]},
    **{f"worldbank/{i}.zip": WORLDBANK.format(i) for i in ["IS.AIR.PSGR", "IS.AIR.DPRT", "IS.AIR.GOOD.MT.K1"]},
    **{f"eurocontrol/{n}": EUROCONTROL.format(n) for n in
       ["Airport_Traffic.xlsx", "Airport_Arrival_ATFM_Delay.xlsx", "All_Pre-Departure_Delay.xlsx",
        "ATC_Pre-Departure_Delay.xlsx", "Taxi-Out_Additional_Time.xlsx", "Taxi-In_Additional_Time.xlsx",
        "ASMA_Additional_Time.xlsx", "En-Route_ATFM_Delay_AUA.xlsx"]},
    "opensky/aircraftDatabase.csv": "https://opensky-network.org/datasets/metadata/aircraftDatabase.csv",
}

# 공공데이터포털 파일데이터 ID
KOREA = {
    15134005: "한국공항공사_공항별_지연현황",
    15002615: "한국공항공사_공항별_결항현황",
    15002610: "한국공항공사_공항별_여객실적",
    15002612: "한국공항공사_공항별_화물실적",
    15132851: "한국공항공사_기종별_운송실적",
    15002707: "한국공항공사_공항별_항공노선",
    15043890: "한국공항공사_국내선_스케줄",
    15002853: "한국공항공사_공항코드",
    15152006: "한국공항공사_지연시간_학습데이터",
    15152055: "한국공항공사_제주공항_여객_학습데이터",
    15062056: "인천공항_항공사별_노선별_운송실적",
    15048966: "인천공항_환승승객현황",
    15002638: "한국공항공사_수송실적_통계",
    15002628: "한국공항공사_항공사별_운항실적",
    15052302: "국토교통부_국내국제_항공수송실적",
    15061960: "국토교통부_연도별_국내국제_항공수송실적",
    15047533: "제주도_제주공항_항공수송실적",
    15047534: "제주도_제주공항_국제선_운항현황",
    15049016: "국토교통부_사고유형별_국내항공사고",
    15049017: "국토교통부_연도별_국내주요항공사고",
    15052305: "국토교통부_연도별_항공사고_통계",
    15049018: "국토교통부_연도별_전세계항공사고",
    15049019: "국토교통부_항공사고_용어",
    15049020: "국토교통부_항공안전데이터_분류체계",
    15049021: "국토교통부_국내외_주요항공일지",
}

# shortcut: 2024~2025년 14개 공항뿐, 전체 BTS와 조인하려면 ASOS_STATIONS와 기간을 늘린다
ASOS_STATIONS = ["ATL", "DFW", "DEN", "ORD", "LAX", "CLT", "MCO", "LAS", "PHX", "MIA",
                 "RKSI", "RKSS", "RKPC", "RKPK"]
ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" + urllib.parse.urlencode(
    {"station": ",".join(ASOS_STATIONS),
     "data": "tmpf,dwpf,relh,drct,sknt,gust,p01i,alti,vsby,skyc1,skyl1,wxcodes",
     "year1": 2024, "month1": 1, "day1": 1, "year2": 2026, "month2": 1, "day2": 1,
     "tz": "Etc/UTC", "format": "onlycomma", "latlon": "yes", "missing": "empty", "trace": "empty",
     "report_type": "3,4"})

UA = {"User-Agent": "air_csv-research-download/1.0 (python-urllib)"}
PORTAL = "https://www.data.go.kr"
OPENSKY_RECORD = 7923702  # 이용 조건: 받은 폴더의 LICENSE.txt
HISTORY_RE = re.compile(r'data-public-pk="(uddi:[^"]+)"\s+data-public-detail-sn="\d+">[^<]*?_(\d{8})\s*<')


def looks_valid(path, suffix):
    """오류 시 HTML 페이지가 200으로 오는 사이트가 있어 내용으로 검증한다."""
    if suffix in (".zip", ".xlsx"):
        return zipfile.is_zipfile(path)
    with open(path, "rb") as f:
        head = f.read(512)
    if suffix == ".pdf":
        return head.startswith(b"%PDF")
    return bool(head) and not head.lstrip().lower().startswith((b"<!doctype", b"<html"))


def fetch(url, dest, retries=3):
    if dest.exists():
        print(f"{dest} 이미 있음, 건너뜀")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if not looks_valid(tmp, dest.suffix):
                raise ValueError("받은 내용이 기대한 파일 형식이 아님")
            tmp.replace(dest)
            print(f"{dest} 완료 ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)
            return True
        except (OSError, ValueError) as e:  # URLError, TimeoutError 모두 OSError
            print(f"{dest} 실패 {attempt}/{retries}: {e}", flush=True)
            time.sleep(5 * attempt)
    tmp.unlink(missing_ok=True)
    return False


def post_portal(path, fields):
    body = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(f"{PORTAL}{path}", data=body, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def fetch_portal_file(pk, uddi, dest):
    """파일 1개. csv로 표기된 zip이 있어 받은 뒤 확장자를 실제 형식으로 바로잡는다."""
    if not any(p.suffix != ".part" for p in dest.parent.glob(f"{dest.stem}.*")):
        info = json.loads(post_portal("/tcs/dss/selectFileDataDownload.do", {
            "publicDataDetailPk": uddi, "publicDataPk": pk, "atchFileId": "",
            "fileDetailSn": 1, "publicDataTyCode": "PR0051"}))
        query = urllib.parse.urlencode({"atchFileId": info["atchFileId"], "fileDetailSn": info["fileDetailSn"]})
        if not fetch(f"{PORTAL}/cmm/cmm/fileDownload.do?{query}", dest):
            return False
        time.sleep(1)
    if dest.suffix == ".csv" and dest.exists() and zipfile.is_zipfile(dest):
        with zipfile.ZipFile(dest) as z:
            real = ".xlsx" if "xl/workbook.xml" in z.namelist() else ".zip"
        dest.replace(dest.with_suffix(real))
    return True


def fetch_korea(pk, name, out_dir):
    """포털의 다운로드 버튼과 같은 순서로 현재 파일과 과거 버전을 받는다."""
    try:
        req = urllib.request.Request(f"{PORTAL}/data/{pk}/fileData.do", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            page = r.read().decode("utf-8", "replace")
        uddi = re.search(r"fn_fileDataDown\('\d+',\s*'(uddi:[^']+)'", page).group(1)
        ext = re.search(r'"encodingFormat":"(\w+)"', page).group(1).lower()
        history = post_portal("/tcs/dss/selectHistAndCsvData.do", {"publicDataPk": pk, "publicDataDetailPk": uddi})
        ok = fetch_portal_file(pk, uddi, out_dir / "korea" / f"{name}.{ext}")
        for old, date in HISTORY_RE.findall(history):
            ok = fetch_portal_file(pk, old, out_dir / "korea" / f"{name}_{date}.{ext}") and ok
        return ok
    except (OSError, ValueError, KeyError, AttributeError) as e:
        print(f"{name} 실패: {e!r}", flush=True)
        return False


def fetch_opensky(out_dir):
    """Zenodo 월별 항공편 목록. 전송이 끊겨 잘린 gz는 형식 검사로 못 잡으므로 md5로 확인한다."""
    try:
        req = urllib.request.Request(f"https://zenodo.org/api/records/{OPENSKY_RECORD}", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            files = json.load(r)["files"]
    except (OSError, ValueError, KeyError) as e:
        print(f"opensky 목록 조회 실패: {e!r}", flush=True)
        return False
    ok = True
    for f in files:
        dest = out_dir / "opensky_flightlist" / f["key"]
        url = f"https://zenodo.org/records/{OPENSKY_RECORD}/files/{f['key']}?download=1"
        if not fetch(url, dest):
            ok = False
            continue
        with open(dest, "rb") as fh:
            digest = hashlib.file_digest(fh, "md5").hexdigest()
        if f"md5:{digest}" != f["checksum"]:
            print(f"{dest} 체크섬 불일치, 삭제. 다시 실행하면 새로 받는다", flush=True)
            dest.unlink()
            ok = False
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--out", default="reference_data")
    out_dir = Path(p.parse_args().out)

    failed = [n for n, url in FILES.items() if not fetch(url, out_dir / n)]
    failed += [name for pk, name in KOREA.items() if not fetch_korea(pk, name, out_dir)]
    # IEM 안내에 따라 공항별로 나누지 않고 한 번에 요청한다
    if not fetch(ASOS_URL, out_dir / "weather_asos" / "asos_14airports_2024_2025.csv"):
        failed.append("weather_asos")
    if not fetch_opensky(out_dir):
        failed.append("opensky_flightlist")
    if failed:
        print("실패:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        t = Path(d) / "t"
        t.write_bytes(b"<html>error</html>")
        assert not any(looks_valid(t, s) for s in (".csv", ".zip", ".xlsx", ".pdf"))
        t.write_bytes(b"a,b\n1,2\n")
        assert looks_valid(t, ".csv")
    assert HISTORY_RE.findall('data-public-pk="uddi:a-1"\n  data-public-detail-sn="2">\n 공항별 지연 현황_20250917\n</a>') \
        == [("uddi:a-1", "20250917")]
    main()
