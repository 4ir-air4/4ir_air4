"""팀원이 저장소를 새로 받았을 때 모든 화면이 나오는지 점검한다. 팀원 PR을 합친 뒤에 돌린다.

git 에 올라간 파일만 임시 폴더에 복사해 "새로 받은 상태"를 만들고, app.py 의 모든 화면을 AppTest 로 연다.
화면마다 선택 상자·라디오·버튼형 선택(앞 6개, 선택지 12개까지)을 하나씩 바꿔 다른 보기도 연다. 화면이 많아 몇 분 걸린다.
- 실패: 화면 오류, "파일·모델이 없다"는 안내, 100MB 넘는 파일(GitHub 한도)
- 알림: 50MB 넘는 파일, 키가 필요한 안내(실시간 지도 OpenSky 키는 각자 .streamlit/secrets.toml 에 넣는다)
실행: python -m ldh.tests.test_fresh_clone [--include-new]   (--include-new: 아직 커밋 안 한 새 파일도 포함, 커밋 전 점검용)
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MISSING = re.compile(r"파일이 없|모델이 없|불러오지 못|찾을 수 없|결과가 없")
NEEDS_KEY = re.compile(r"인증 정보|secrets\.toml")

# 임시 폴더를 작업 폴더로 한 별도 프로세스에서 돈다. 그래야 import 가 원래 프로젝트가 아니라 복사본의 ldh·ojh 를 쓴다
PAGES_CODE = r"""
import glob, json, re
from streamlit.runtime.state.common import TESTING_KEY
from streamlit.testing.v1 import AppTest
core = re.findall(r'st\.Page\("([^"]+)"', open("app.py", encoding="utf-8").read())
pages = core + [p.replace("\\", "/") for p in sorted(glob.glob("ldh/categories/*/page.py")) if "_template" not in p]
KINDS = ("selectbox", "radio", "segmented_control", "pills")


def choose(kind, w, label):
    # AppTest 는 고른 값을 format_func 로 한 번 더 바꿔 보내서("1월" -> "1월월") 표시 이름으로는 못 고른다.
    # 실제 브라우저처럼 표시 이름을 그대로 보내도록 이 위젯의 변환을 잠시 그대로 두기로 바꾼다
    w.root.session_state[TESTING_KEY][w.id] = lambda x: x
    if kind == "pills" and not w._is_single_select:
        return w.set_value([label])
    return w.set_value(label)


def seen(at, where):
    # 선택지를 바꾼 경우 어느 선택에서 나온 안내인지 앞에 붙인다
    tag = f"[{where}] " if where else ""
    return ([tag + str(e.value)[:300] for e in at.exception], [tag + str(m.value) for kind in (at.error, at.warning, at.info) for m in kind])


def widgets(at):
    return [(kind, w) for kind in KINDS for w in getattr(at, kind)]


out = []
for p in dict.fromkeys(pages):
    at = AppTest.from_file("app.py", default_timeout=300).run()
    at.switch_page(p).run()
    errors, msgs = seen(at, "")
    tables, metrics, tried = len(at.dataframe), len(at.metric), 0
    # 화면 안의 선택 상자·라디오 선택지를 하나씩 바꿔 연다(첫 보기만 열면 다른 보기의 파일 누락을 놓친다)
    for i in range(min(len(widgets(at)), 6)):
        count = len(widgets(at)[i][1].options) if i < len(widgets(at)) else 0
        for k in range(min(count, 12)):
            ws = widgets(at)
            if i >= len(ws) or k >= len(ws[i][1].options):
                break
            kind, w = ws[i]
            label = f"{w.label}={w.options[k]}"
            try:
                choose(kind, w, w.options[k]).run()
            except Exception as exc:
                errors.append(f"[{label}] 선택 실패: {str(exc)[:150]}")
                continue
            tried += 1
            e, m = seen(at, label)
            errors += e
            msgs += m
    out.append({"page": p, "errors": list(dict.fromkeys(errors)), "messages": list(dict.fromkeys(msgs)), "tables": tables, "metrics": metrics, "tried": tried})
print("@@" + json.dumps(out, ensure_ascii=False))
"""


def classify(page):
    """화면 하나의 결과 -> (실패 사유 목록, 알림 목록)."""
    fails = [f"오류: {e[:150]}" for e in page["errors"]]
    notes = []
    for m in page["messages"]:
        if NEEDS_KEY.search(m):
            notes.append(f"키 필요: {m[:80]}")
        elif MISSING.search(m):
            fails.append(f"파일 없음 안내: {m[:120]}")
    return fails, notes


def tracked_files(include_new):
    args = ["git", "-c", "core.quotepath=false", "ls-files", "-z"] + (["-co", "--exclude-standard"] if include_new else [])
    return [f for f in subprocess.run(args, cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8").split("\0") if f]


def main(include_new):
    files = tracked_files(include_new)
    sizes = {f: (ROOT / f).stat().st_size for f in files if (ROOT / f).is_file()}
    big = {f: s for f, s in sizes.items() if s > 50 * 2**20}
    fails = [f"100MB 넘음: {f} ({s / 2**20:.0f}MB)" for f, s in big.items() if s > 100 * 2**20]
    print(f"파일 {len(sizes):,}개, {sum(sizes.values()) / 2**20:.0f}MB" + (" (커밋 전 새 파일 포함)" if include_new else ""))
    for f, s in big.items():
        print(f"  알림: 50MB 넘는 파일 {f} ({s / 2**20:.0f}MB)")
    with tempfile.TemporaryDirectory() as tmp:
        for f in sizes:
            (Path(tmp) / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, Path(tmp) / f)
        run = subprocess.run([sys.executable, "-c", PAGES_CODE], cwd=tmp, capture_output=True, text=True, encoding="utf-8", errors="replace",
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        line = next((x for x in run.stdout.splitlines() if x.startswith("@@")), None)
        if line is None:
            raise SystemExit(f"화면 점검을 돌리지 못했습니다:\n{run.stderr[-2000:]}")
        pages = json.loads(line[2:])
    for page in pages:
        page_fails, notes = classify(page)
        fails += [f"{page['page']}: {x}" for x in page_fails]
        print(f"{'실패' if page_fails else '정상'} {page['page']} (표 {page['tables']}, 지표 {page['metrics']}, 선택지 {page.get('tried', 0)}개 확인)"
              + "".join(f"\n  {x}" for x in page_fails + list(dict.fromkeys(notes))))
    if fails:
        raise SystemExit(f"\n새로 받은 상태에서 문제 {len(fails)}건:\n" + "\n".join(fails))
    print(f"\n화면 {len(pages)}개 모두 정상입니다.")


def check():
    ok = classify({"errors": [], "messages": ["항공기 데이터를 불러오지 못했습니다: OpenSky 인증 정보가 없습니다.", "수집이 멈춘 것 같습니다."]})
    assert ok == ([], ["키 필요: 항공기 데이터를 불러오지 못했습니다: OpenSky 인증 정보가 없습니다."]), "키 안내는 실패가 아님"
    bad = classify({"errors": ["boom"], "messages": ["모델 파일이 없습니다."]})
    assert len(bad[0]) == 2, "오류와 파일 없음 안내는 실패"


if __name__ == "__main__":
    check()
    main("--include-new" in sys.argv)
