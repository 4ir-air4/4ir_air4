"""모든 카테고리에서 LightGBM, XGBoost, RandomForest, LSTM 학습.  사용법: python -m ldh.scripts.run_models [slug 또는 slug:task ...] [--force | --resave] [--models=lstm,lightgbm] [--year=2026]

각 카테고리의 models.py 가 tasks() 로 학습 대상을 알려 준다. 끝난 모델은 건너뛰므로 끊겨도 다시 실행하면 이어서 진행한다.
--year: 그 해의 행으로만 학습해 outputs_<해>/ 에 따로 저장한다. 전체 학습 결과와 섞이지 않는다. 자료가 없는 과제는 건너뛴다.
"""
import gc
import importlib
import sys
import threading
import time
import traceback

from ldh.core.models import MODEL_NAMES, MODELS, merge_summary, only_year, run_model, self_check
from ldh.core.registry import CATEGORY_DIR, discover
from ldh.core.store import OUTPUT_DIR, Store


def log(message):
    print(f"{time.strftime('%m-%d %H:%M:%S')} {message}", flush=True)


current = "준비 중"


def heartbeat(seconds=300):
    """오래 걸리는 모델도 셀 출력과 로그에 작업 중 신호를 남긴다. 코랩의 최대 사용 시간을 늘리지는 못한다."""
    started = time.time()
    while True:
        time.sleep(seconds)
        log(f"작업 중 ({(time.time() - started) / 60:.0f}분 경과): {current}")


if __name__ == "__main__":
    self_check()
    threading.Thread(target=heartbeat, daemon=True).start()
    force = "--force" in sys.argv
    resave = "--resave" in sys.argv
    year = next((int(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--year=")), None)
    chosen = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--models=")), MODELS)
    unknown = set(chosen) - set(MODELS)
    if unknown:
        raise SystemExit(f"알 수 없는 모델: {unknown}. 가능: {MODELS}")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    picks = {}  # slug -> 고른 과제 (비면 전부). 'slug:task' 로 과제 하나만 고를 수 있다
    for a in args:
        slug, _, task = a.partition(":")
        picks.setdefault(slug, set()).update([task] if task else [])
    categories, problems = discover()
    for p in problems:
        log(f"카테고리 읽기 실패: {p}")
    specs = []
    for c in categories:
        if (args and c["slug"] not in picks) or not (CATEGORY_DIR / c["slug"] / "models.py").exists():
            continue
        module = importlib.import_module(f"ldh.categories.{c['slug']}.models")
        specs.extend((c["slug"], t) for t in module.tasks() if not picks.get(c["slug"]) or t["task"] in picks[c["slug"]])
    specs.sort(key=lambda s: s[1].get("rows", 0))  # 작은 데이터부터 끝내 결과를 빨리 확인한다
    total = len(specs) * len(chosen)
    log(f"학습 대상 {len(specs)}개, 모델 {total}개 (작은 데이터부터 진행)")
    done, failed = 0, []
    for slug, t in specs:
        store = Store(slug)
        if year:
            store.dir = OUTPUT_DIR.parent / str(year) / slug
        names = chosen
        # --resave: 결과가 있어도 모델 파일이 없으면 다시 학습한다. --force 와 달리 끊긴 뒤 이어 갈 수 있다
        marker = (lambda m: store.dir / "saved_models" / f"{t['task']}__{m}" / "meta.json") if resave else (lambda m: store.dir / "model_results" / f"{t['task']}__{m}.json")
        todo = [m for m in names if force or not marker(m).exists()]
        done += len(names) - len(todo)
        if not todo:
            continue
        log(f"=== [{slug}] {t['title']} ({t['task']}), 약 {t.get('rows', 0):,}행")
        current = f"{t['task']} 데이터 읽는 중"
        try:
            data = t["load"]()
            if year:
                data = only_year(data, year)
        except Exception:
            traceback.print_exc()
            failed.extend(f"{t['task']}/{m}" for m in todo)
            done += len(todo)
            continue
        if data is None:
            log(f"건너뜀: {t['task']} 에 {year}년 자료가 부족합니다")
            done += len(todo)
            continue
        for m in todo:
            done += 1
            log(f"--- [{done}/{total}] {MODEL_NAMES[m]} 학습 시작: {t['task']}")
            current = f"[{done}/{total}] {MODEL_NAMES[m]} 학습: {t['task']}"
            try:
                row = run_model(store, t["task"], t["title"], t["kind"], data, m)
                shown = {k: round(row[k], 4) for k in ("roc_auc", "pr_auc", "baseline_pr_auc", "mae", "baseline_mae", "improve", "r2", "accuracy", "macro_f1", "majority_rate") if k in row}
                log(f"완료 {MODEL_NAMES[m]} {t['task']}: {row['seconds']}초 {shown}")
                merge_summary(store)
            except Exception:
                traceback.print_exc()
                failed.append(f"{t['task']}/{m}")
            gc.collect()
        del data
        gc.collect()
    log(f"전체 완료. 실패 {len(failed)}개: {failed}")
    if failed:
        raise SystemExit(1)
