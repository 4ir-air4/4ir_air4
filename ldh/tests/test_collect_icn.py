"""네트워크 없이 도는 collect_icn.py 점검. 실행: python -m ldh.tests.test_collect_icn"""
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest import mock

import requests

from ldh.collect import collect_icn as ci

KEY = "SECRET-TEST-KEY"


def body(items, total=None, code="00"):
    return json.dumps({"response": {"header": {"resultCode": code, "resultMsg": "NORMAL" if code == "00" else "ERROR"},
                                    "body": {"items": items, "totalCount": len(items) if total is None else total}}}, ensure_ascii=False)


def reply(text, status=200):
    return mock.Mock(status_code=status, text=text)


def flight(name, estimated="1210", remark="도착"):
    return {"flightId": name, "scheduleDateTime": "1200", "airportCode": "NRT", "estimatedDateTime": estimated, "remark": remark}


def readable_lines(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def check_parse():
    assert ci.parse_page(body([flight("A")]), KEY) == ([flight("A")], 1), "목록"
    assert ci.parse_page(body({"item": [flight("A")]}, 1), KEY)[0] == [flight("A")], "item 목록"
    assert ci.parse_page(body({"item": flight("A")}, 1), KEY)[0] == [flight("A")], "item 한 건"
    assert ci.parse_page(body("", 0), KEY) == ([], 0), "빈 결과"
    for text in (body([], code="30"), f"<OpenAPI_ServiceResponse><returnAuthMsg>{KEY} NOT_REGISTERED</returnAuthMsg></OpenAPI_ServiceResponse>"):
        try:
            ci.parse_page(text, KEY)
        except ci.CollectError as e:
            assert KEY not in str(e), "오류 문구에 인증키가 없어야 함"
        else:
            raise AssertionError("오류 응답을 정상으로 처리함")


def check_paging_and_retry():
    with mock.patch.object(ci, "PAGE_ROWS", 2), mock.patch.object(ci.time, "sleep") as sleep, mock.patch.object(ci.requests, "get") as get:
        get.side_effect = [reply("", 503), reply(body([flight("A"), flight("B")], 3)), reply(body([flight("C")], 3))]
        assert [r["flightId"] for r in ci.fetch(ci.DATASETS["arrivals"], KEY)] == ["A", "B", "C"], "재시도와 여러 쪽"
        assert sleep.call_count == 1 and get.call_args_list[2].kwargs["params"]["pageNo"] == 2, "두 번째 쪽 요청"
        get.side_effect = requests.ConnectionError(f"failed ?serviceKey={KEY}")
        try:
            ci.fetch(ci.DATASETS["arrivals"], KEY)
        except ci.CollectError as e:
            assert KEY not in str(e), "연결 오류에 인증키가 없어야 함"
        else:
            raise AssertionError("연결 실패를 정상으로 처리함")
        get.reset_mock(side_effect=True)
        sleep.reset_mock()
        whole_day = json.dumps({"response": {"header": {"resultCode": "00"}, "body": {"items": [flight("A"), flight("B"), flight("C")]}}})
        get.return_value = reply(whole_day)  # 실제 현황 API 형태: totalCount 없이 numOfRows 보다 많이 온다
        assert len(ci.fetch(ci.DATASETS["arrivals"], KEY)) == 3 and get.call_count == 1, "건수가 없으면 한 번만 요청"
        get.reset_mock()
        for status, hint in ((404, "경로"), (401, "인증키")):
            get.return_value = reply("", status)
            try:
                ci.fetch(ci.DATASETS["arrivals"], KEY)
            except ci.CollectError as e:
                assert hint in str(e), f"{status} 안내"
            else:
                raise AssertionError(f"{status} 를 정상으로 처리함")
        assert sleep.call_count == 0, "400번대 오류는 재시도하지 않음"


def check_changes_only(root):
    path = root / "a.jsonl"
    fields = ci.DATASETS["arrivals"]["key"]
    assert ci.append_changed(path, [flight("A"), flight("B")], fields, "t1") == 2, "처음에는 모두 저장"
    assert ci.append_changed(path, [flight("A"), flight("B")], fields, "t2") == 0, "그대로면 저장 안 함"
    assert ci.append_changed(path, [flight("A", "1230"), flight("B")], fields, "t3") == 1, "바뀐 편만 저장"
    assert ci.append_changed(path, [flight("A", "1210")], fields, "t4") == 1, "이전 값으로 돌아가도 저장"
    saved = readable_lines(path)
    assert [(r["fetched_at"], r["data"]["estimatedDateTime"]) for r in saved if r["data"]["flightId"] == "A"] == [("t1", "1210"), ("t3", "1230"), ("t4", "1210")], "시점별 값 재현"
    with path.open("ab") as f:
        f.write(b'{"fetched_at": "t5", "data": {"flight')  # 저장 중 끊긴 줄
    assert ci.append_changed(path, [flight("C")], fields, "t6") == 1, "잘린 줄 뒤에도 저장"
    assert readable_lines(path)[-1]["data"]["flightId"] == "C", "새 줄이 잘린 줄과 섞이지 않음"
    closing = root / "c.jsonl"
    row = {"msDate": "20261106", "msIo": "A", "flightNum": "KE1", "msStime": "1200", "msDelay": "0"}
    assert ci.append_changed(closing, [row, dict(row, flightNum="KE2")], ci.DATASETS["closing"]["key"], "t1") == 2, "마감 자료는 편별로 구분"
    assert ci.append_changed(closing, [dict(row, msDelay="1")], ci.DATASETS["closing"]["key"], "t2") == 1, "지연 표시가 바뀌면 저장"
    assert ci.append_changed(closing, [row], ci.DATASETS["closing"]["key"], "t3") == 1, "지연 표시가 되돌아가도 저장"


def check_collect_and_status(root):
    now = datetime(2026, 11, 7, 1, 0, tzinfo=ci.KST)
    with mock.patch.object(ci.requests, "get") as get, mock.patch.object(ci.time, "sleep"):
        get.side_effect = [reply(body([flight("A")])), reply("", 401)]
        assert ci.collect(["arrivals", "departures"], KEY, root, now) == 1, "실패 수를 돌려줌"
    assert (root / "2026-11-07" / "arrivals.jsonl").exists() and not (root / "2026-11-07" / "departures.jsonl").exists(), "한 자료가 실패해도 다른 자료는 저장"
    assert KEY not in (root / "runs.jsonl").read_text(encoding="utf-8"), "실행 기록에 인증키가 없어야 함"
    rows = {r["자료"]: r for r in ci.status(root, days=1, now=datetime(2026, 11, 7, 3, 0, tzinfo=ci.KST))}
    assert rows["arrivals"]["성공"] == 1 and rows["arrivals"]["최대 공백(분)"] == 120, "성공 수와 공백"
    assert rows["departures"]["실패"] == 1 and rows["departures"]["최대 공백(분)"] == 180 and "인증키" in rows["departures"]["마지막 오류"], "실패와 오류 표시"


def check_backup(root):
    source, drive = root / "live", root / "drive"
    day = source / "2026-11-07" / "arrivals.jsonl"
    day.parent.mkdir(parents=True)
    day.write_text("a\n", encoding="utf-8")
    copy = drive / "ldh" / "datasets" / "raw" / "icn_live" / "2026-11-07" / "arrivals.jsonl"
    assert ci.backup(source, drive) == 1 and not drive.exists(), "드라이브가 없으면 실패로 남기고 폴더를 만들지 않음"
    assert '"ok": false' in (source / "runs.jsonl").read_text(encoding="utf-8"), "실패 기록"
    drive.mkdir()
    assert ci.backup(source, drive) == 0 and copy.read_text(encoding="utf-8") == "a\n", "처음에는 복사"
    day.write_text("a\nb\n", encoding="utf-8")
    assert ci.backup(source, drive) == 0 and copy.read_text(encoding="utf-8") == "a\nb\n", "덧붙은 파일을 다시 복사"
    day.write_text("a\n", encoding="utf-8")
    assert ci.backup(source, drive) == 0 and copy.read_text(encoding="utf-8") == "a\nb\n", "원본이 짧아져도 백업을 덮어쓰지 않음"
    day.unlink()
    assert ci.backup(source, drive) == 0 and copy.exists(), "원본이 없어져도 백업은 남음"
    rows = {r["자료"]: r for r in ci.status(source, days=1)}
    assert rows["backup"]["성공"] == 4 and rows["backup"]["실패"] == 1, "상태에 백업 성공과 실패가 나옴"


def check_live_flights(root):
    from ldh.categories.korea_ops.pipeline import live_flights
    day = root / "2026-11-10"
    day.mkdir(parents=True)
    line = lambda d: json.dumps({"fetched_at": "t", "data": {"airline": "X항공", "airportCode": "NRT", "codeshare": None, **d}}, ensure_ascii=False)  # noqa: E731
    (day / "arrivals.jsonl").write_text("\n".join([
        line({"flightId": "A1", "scheduleDateTime": "1200", "estimatedDateTime": "1215", "remark": "착륙"}),
        line({"flightId": "A1", "scheduleDateTime": "1200", "estimatedDateTime": "1216", "remark": "도착"}),
        line({"flightId": "A2", "scheduleDateTime": "1200", "estimatedDateTime": "1216", "remark": "도착", "codeshare": "Slave"}),
        line({"flightId": "A3", "scheduleDateTime": "2350", "estimatedDateTime": "0010", "remark": "착륙"})]), encoding="utf-8")
    (day / "departures.jsonl").write_text("\n".join([
        line({"flightId": "D1", "scheduleDateTime": "0005", "estimatedDateTime": "2350", "remark": "출발"}),
        line({"flightId": "D2", "scheduleDateTime": "0900", "estimatedDateTime": "", "remark": "결항"})]), encoding="utf-8")
    d = live_flights(root).set_index("flight")
    assert sorted(d.index) == ["A1", "D1"], "끝난 편만, 공동운항·결항·미완료 제외"
    assert d.loc["A1", "api_delay"] == 16 and d.loc["A1", "delay"] == 3, "도착은 마지막 상태 값에서 13분 뺌"
    assert d.loc["D1", "api_delay"] == -15 and d.loc["D1", "delay"] == 0, "자정 전에 일찍 떠난 00시대 편은 음수, 출발은 15분 더함"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        check_live_flights(Path(tmp))
    check_parse()
    check_paging_and_retry()
    with tempfile.TemporaryDirectory() as tmp:
        check_changes_only(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        check_collect_and_status(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        check_backup(Path(tmp))
    print("collect_icn 점검 통과")
