"""한 과제를 시험 학습해 속도와 지표를 저장된 결과와 비교한다. 결과는 임시 폴더에 쓰므로 ldh/results/full/ 는 바뀌지 않는다.

사용법: python -m ldh.scripts.trial_model <카테고리> <과제> [모델]   예: python -m ldh.scripts.trial_model korea_demand type_passengers lstm
"""
import importlib
import json
import sys
import tempfile
from pathlib import Path

from ldh.core.models import MODELS, run_model, self_check
from ldh.core.store import Store

SHOWN = ("seconds", "fit_seconds", "roc_auc", "pr_auc", "mae", "baseline_mae", "r2", "accuracy", "macro_f1")

if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    slug, task = sys.argv[1:3]
    model = sys.argv[3] if len(sys.argv) == 4 else "lstm"
    if model not in MODELS:
        raise SystemExit(f"알 수 없는 모델: {model}. 가능: {MODELS}")
    self_check()
    t = next((t for t in importlib.import_module(f"ldh.categories.{slug}.models").tasks() if t["task"] == task), None)
    if t is None:
        raise SystemExit(f"{slug} 에 과제 {task} 가 없습니다.")
    store = Store(slug)
    saved = store.dir / "model_results" / f"{task}__{model}.json"
    before = json.loads(saved.read_text(encoding="utf-8")) if saved.exists() else {}
    with tempfile.TemporaryDirectory() as tmp:
        store.dir = Path(tmp)
        row = run_model(store, task, t["title"], t["kind"], t["load"](), model)
    for k in SHOWN:
        if k in row:
            print(f"{k:14} 이번 {row[k]:>12.4f}   저장된 결과 {before.get(k, float('nan')):>12.4f}")
