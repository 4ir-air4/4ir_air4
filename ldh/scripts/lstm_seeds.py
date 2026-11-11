"""작은 카테고리 LSTM 을 시드 1, 2 로 더 학습해 시드에 따른 흔들림을 잰다. 결과는 임시 폴더에 쓰고 ldh/results/full 은 바꾸지 않는다.
시드 0 은 run_models 로 학습한 결과를 쓴다. 결과: ldh/results/lstm_seeds.json. 실행: python -m ldh.scripts.lstm_seeds europe korea_demand korea_ops safety"""
import importlib
import json
import sys
import tempfile
from pathlib import Path


from ldh.core.models import run_model
from ldh.core.store import Store

OUT = Path(__file__).resolve().parents[1] / "results" / "lstm_seeds.json"
results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
for slug in sys.argv[1:]:
    for t in importlib.import_module(f"ldh.categories.{slug}.models").tasks():
        d = t["load"]()
        for seed in (1, 2):
            key = f"{t['task']}/{seed}"
            if key in results:
                continue
            store = Store(slug)
            with tempfile.TemporaryDirectory() as tmp:
                store.dir = Path(tmp)
                row = run_model(store, t["task"], t["title"], t["kind"], d, "lstm", seed=seed)
            results[key] = {k: row[k] for k in ("pr_auc", "mae", "macro_f1") if k in row}
            OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            print(key, results[key], flush=True)
