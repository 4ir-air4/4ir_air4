"""네트워크 없이 도는 download_kosis.py 점검. 실행: python test_download_kosis.py"""
import os
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest import mock

import pandas as pd
import requests

import download_kosis as dk

KEY = "SECRET-TEST-KEY"
os.environ["KOSIS_API_KEY"] = KEY


def response(payload):
    r = mock.Mock()
    r.json.return_value = payload
    return r


def exit_message(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except SystemExit as e:
        return str(e.code)
    raise AssertionError("SystemExit 가 발생하지 않음")


# 환경변수가 없으면 .env 에서 읽는다
with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ):
    del os.environ["KOSIS_API_KEY"]
    env = Path(tmp) / ".env"
    env.write_text('OTHER=1\nKOSIS_API_KEY="from-file"\n', encoding="utf-8")
    assert dk.load_key(env) == "from-file"
    assert "없습니다" in exit_message(dk.load_key, Path(tmp) / "none")

# 오류 응답은 중단하고, 요청에는 jsonVD 가 들어간다
with mock.patch.object(dk.requests, "get", return_value=response({"err": "20", "errMsg": "필수 누락"})) as get:
    assert exit_message(dk.call, "x") == "KOSIS 오류 20: 필수 누락"
    assert get.call_args.kwargs["params"]["jsonVD"] == "Y"

# 요청 실패 메시지에 키가 새지 않는다
leak = requests.ConnectionError(f"failed: https://kosis.kr/openapi/x?apiKey={KEY}")
with mock.patch.object(dk.requests, "get", side_effect=leak):
    message = exit_message(dk.call, "x")
    assert KEY not in message and "ConnectionError" in message

# 목록이 아닌 응답은 받아들이지 않는다
with mock.patch.object(dk.requests, "get", return_value=response({"a": 1})):
    assert "형식" in exit_message(dk.call, "x")

# 빠진 달 찾기
assert dk.missing_months(["202001", "202003", "202003"]) == ["2020-02"]
assert dk.missing_months(["202012", "202101"]) == []

# fetch: 기간 없으면 중단, 빈 응답은 저장 안 함, 정상 응답은 CSV 저장
base = dict(org="101", tbl="T", item="I", obj=["A", "B"], prd="M", start=None, end=None, recent=None, name="t")
with tempfile.TemporaryDirectory() as tmp, mock.patch.object(dk, "OUT_DIR", Path(tmp)):
    assert "--start" in exit_message(dk.fetch, Namespace(**base))

    with mock.patch.object(dk.requests, "get", return_value=response([])):
        assert "저장하지" in exit_message(dk.fetch, Namespace(**{**base, "recent": 3}))
    assert not (Path(tmp) / "t.csv").exists()

    rows = [{"PRD_DE": "202001", "DT": "100.5"}, {"PRD_DE": "202003", "DT": "-"}]
    with mock.patch.object(dk.requests, "get", return_value=response(rows)) as get:
        dk.fetch(Namespace(**{**base, "start": "202001", "end": "202003"}))
    params = get.call_args.kwargs["params"]
    assert (params["objL1"], params["objL2"], params["startPrdDe"]) == ("A", "B", "202001")
    assert len(pd.read_csv(Path(tmp) / "t.csv", encoding="utf-8-sig")) == 2

print("OK")
