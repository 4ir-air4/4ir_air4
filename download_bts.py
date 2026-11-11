"""BTS On-Time Performance 월별 zip 일괄 다운로드.

사용법:
    python download_bts.py 2015-01 2024-12          # ./bts_data 에 저장
    python download_bts.py 2023-01 2023-03 -o D:/bts

이미 받은 파일은 건너뛰므로 중단 후 다시 실행하면 이어서 받는다.
읽기 (압축 해제 불필요):
    from download_bts import load
    df = load("bts_data/ontime_2024_01.zip")
"""
import argparse
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

URL = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip"


def months(start, end):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def download(y, m, out_dir, retries=3):
    dest = out_dir / f"ontime_{y}_{m:02d}.zip"
    if dest.exists() and zipfile.is_zipfile(dest):
        print(f"{y}-{m:02d} 이미 있음, 건너뜀")
        return True
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(URL.format(y=y, m=m), headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            # 없는 달은 404 대신 HTML 페이지가 올 수 있어 zip 여부로 검증
            if not zipfile.is_zipfile(tmp):
                raise ValueError("zip 파일이 아님 (아직 공개되지 않은 달일 수 있음)")
            tmp.replace(dest)
            print(f"{y}-{m:02d} 완료 ({dest.stat().st_size / 1e6:.1f} MB)")
            return True
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            print(f"{y}-{m:02d} 실패 {attempt}/{retries}: {e}")
            time.sleep(5 * attempt)
    tmp.unlink(missing_ok=True)
    return False


def load(path):
    """월별 zip을 DataFrame으로 읽는다. zip 안에 readme.html도 있어 CSV만 골라 연다."""
    import pandas as pd
    with zipfile.ZipFile(path) as z:
        csv = next(n for n in z.namelist() if n.endswith(".csv"))
        return pd.read_csv(z.open(csv), low_memory=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("start", help="YYYY-MM (최소 1987-10)")
    p.add_argument("end", help="YYYY-MM")
    p.add_argument("-o", "--out", default="bts_data")
    a = p.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = [f"{y}-{m:02d}" for y, m in months(a.start, a.end) if not download(y, m, out_dir)]
    if failed:
        print("실패한 달:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    assert list(months("2023-11", "2024-02")) == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]
    main()
