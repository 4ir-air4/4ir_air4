"""저장한 트리 모델의 변수 중요도와 SHAP 값을 계산해 ldh/results/full/<slug>/ 에 저장한다. 모델 비교 화면이 읽는다.
사용법: python -m ldh.scripts.explain_models [slug ...] [--force] [--rows=2000]
결과: importance_<과제>__<모델>.parquet (특성, 중요도), shap_<과제>__<모델>.parquet (특성, SHAP, 값, 순위)
SHAP 은 LightGBM·XGBoost 가 내장한 TreeSHAP(pred_contrib)으로 평가 구간 표본 행에서 계산한다. LSTM 은 계산하지 않는다."""
import importlib
import json
import sys

import numpy as np
import pandas as pd

from ldh.core.models import TEST, THREADS, load_tree
from ldh.core.store import Store

TREES = ["lightgbm", "xgboost", "random_forest"]


def contributions(model, name, X, cat_idx, n_classes):
    """행×특성 SHAP 값. 다중 분류는 각 행의 예측 클래스 쪽 기여만 남긴다."""
    if name == "xgboost":
        import xgboost as xgb
        types = ["c" if i in cat_idx else "q" for i in range(X.shape[1])]
        dm = xgb.DMatrix(X, feature_types=types, enable_categorical=True, nthread=THREADS)
        best = (0, model.best_iteration + 1)
        raw = model.predict(dm, pred_contribs=True, iteration_range=best)
        proba = model.predict(dm, iteration_range=best) if n_classes else None
    else:
        raw = model.predict(X, pred_contrib=True)
        proba = model.predict(X) if n_classes else None
    F = X.shape[1]
    if n_classes:
        raw = np.asarray(raw).reshape(len(X), n_classes, F + 1)
        raw = raw[np.arange(len(X)), proba.argmax(axis=1)]
    return np.asarray(raw)[:, :F]  # 마지막 열은 기댓값(bias)


def importance(model, name, F):
    if name == "xgboost":
        score = model.get_score(importance_type="total_gain")
        gain = np.array([score.get(f"f{i}", 0.0) for i in range(F)])
    else:
        gain = model.feature_importance("gain").astype(float)
    return gain / gain.sum() if gain.sum() > 0 else gain


def explain(store, task, data, name, n_rows, rng):
    folder = store.dir / "saved_models" / f"{task}__{name}"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    model = load_tree(folder, name)
    cols = meta["columns"]
    store.save_table(f"importance_{task}__{name}", pd.DataFrame({"특성": cols, "중요도": importance(model, name, len(cols))}))
    test = np.flatnonzero(data.split == TEST)
    rows = np.sort(rng.choice(test, size=min(n_rows, len(test)), replace=False))
    X = np.asarray(data.X[rows], dtype=np.float32)
    shap = contributions(model, name, X, meta["cat_idx"], len(meta["classes"]) if meta["kind"] == "multiclass" else 0)
    # 순위: 특성마다 값의 백분위(0~1). 색으로 값이 큰 행과 작은 행을 구분한다
    rank = pd.DataFrame(X).rank(pct=True).to_numpy()
    store.save_table(f"shap_{task}__{name}", pd.DataFrame({"특성": np.repeat([cols], len(rows), axis=0).ravel(), "SHAP": shap.ravel(),
                                                          "값": X.ravel(), "순위": rank.ravel()}))
    return len(rows)


def check():
    """다중 분류 기여 선택과 중요도 정규화를 작은 예로 확인한다."""
    class Fake:
        def predict(self, X, pred_contrib=False):
            if pred_contrib:  # 2행, 2클래스, 특성 1개 + bias
                return np.array([[1.0, 0.5, -2.0, 0.5], [3.0, 0.5, 4.0, 0.5]])
            return np.array([[0.9, 0.1], [0.2, 0.8]])

        def feature_importance(self, kind):
            return np.array([3, 1])
    out = contributions(Fake(), "lightgbm", np.zeros((2, 1)), [], 2)
    assert out.ravel().tolist() == [1.0, 4.0], out
    assert importance(Fake(), "lightgbm", 2).tolist() == [0.75, 0.25]


if __name__ == "__main__":
    check()
    force = "--force" in sys.argv
    n_rows = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--rows=")), 2000)
    slugs = [a for a in sys.argv[1:] if not a.startswith("--")]
    for slug in slugs or ["safety"]:
        store = Store(slug)
        for t in importlib.import_module(f"ldh.categories.{slug}.models").tasks():
            names = [m for m in TREES if (store.dir / "saved_models" / f"{t['task']}__{m}" / "meta.json").exists()
                     and (force or not store.table_path(f"shap_{t['task']}__{m}").exists())]
            if not names:
                continue
            data = t["load"]()
            for name in names:
                n = explain(store, t["task"], data, name, n_rows, np.random.default_rng(0))
                print(f"완료 {slug}/{t['task']}/{name}: 표본 {n:,}행", flush=True)
            del data
    print("끝")
