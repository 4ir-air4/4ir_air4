"""Streamlit Cloud 배포용 OpenSky 저장본을 만든다.

배포 서버(클라우드 IP)는 OpenSky가 막아서, 배포 앱의 실시간 지도는 이 저장본을 보여 준다.
이 PC에서 앱을 열어 실시간 지도가 한 번 뜨면 ojh/data/cache/opensky_states_last.json 이 생기고,
이 스크립트가 그 파일을 압축해 ojh/data/snapshot/opensky_states.json.gz 로 복사한다.

    python -m ldh.scripts.save_opensky_snapshot

새 저장본은 `git add -f ojh/data/snapshot/opensky_states.json.gz` 로 올린다(data/ 폴더는 git 제외 규칙에 걸림).
"""
import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "ojh" / "data" / "cache" / "opensky_states_last.json"
DST = ROOT / "ojh" / "data" / "snapshot" / "opensky_states.json.gz"


def main():
    if not SRC.exists():
        raise SystemExit(f"{SRC} 가 없습니다. 먼저 이 PC에서 앱을 열어 실시간 지도를 한 번 띄우세요.")

    data = json.loads(SRC.read_text(encoding="utf-8"))
    states = data.get("states") or []
    assert data.get("time") and states, "받아 둔 응답에 시각이나 항공기 목록이 없습니다."

    DST.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0: 내용이 같으면 압축 파일도 같게 (git에 불필요한 변경이 생기지 않도록)
    DST.write_bytes(gzip.compress(json.dumps(data, separators=(",", ":")).encode("utf-8"), mtime=0))

    kst = datetime.fromtimestamp(data["time"], timezone(timedelta(hours=9)))
    print(f"저장본: {DST.relative_to(ROOT)} ({DST.stat().st_size / 1024:,.0f}KB)")
    print(f"기준 시각 {kst:%Y-%m-%d %H:%M} KST, 항공기 {len(states):,}대")


if __name__ == "__main__":
    main()
