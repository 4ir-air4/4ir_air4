"""카테고리별 통계·학습 실행.  사용법: python -m ldh.scripts.run_all [slug ...]   인자 없으면 전부

하나가 실패해도 나머지는 계속 실행하고, 실패한 것이 있으면 종료 코드 1로 끝난다.
"""
import importlib
import sys
import time
import traceback

from ldh.core.registry import discover
from ldh.core.store import Store

if __name__ == "__main__":
    categories, problems = discover()
    for p in problems:
        print("카테고리 읽기 실패:", p, flush=True)
    slugs = [c["slug"] for c in categories]
    names = sys.argv[1:] or slugs
    unknown = set(names) - set(slugs)
    if unknown:
        raise SystemExit(f"알 수 없는 카테고리: {unknown}. 가능: {slugs}")
    failed = list(problems)
    for slug in names:
        start = time.time()
        print(f"[{slug}] 시작", flush=True)
        try:
            importlib.import_module(f"ldh.categories.{slug}.pipeline").run(Store(slug))
            print(f"[{slug}] 완료 {round(time.time() - start)}s", flush=True)
        except Exception:
            traceback.print_exc()
            failed.append(slug)
    if failed:
        raise SystemExit(f"실패: {failed}")
