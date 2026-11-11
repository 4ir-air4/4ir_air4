"""작은 과제의 LightGBM 하이퍼파라미터를 검증 구간만 보고 무작위 탐색한다. core/models.py 와 outputs/ 는 바꾸지 않는다.

사용법: python tune_small.py [slug 또는 slug:task ...] [--trials=30] [--threads=2]   (기본: europe korea_demand korea_ops safety)
결과: tuned_params.json. 평가 구간은 설정을 고른 뒤 기본 설정과 고른 설정으로 한 번씩만 계산한다.
기본 설정은 core/models.py fit_lightgbm 과 같아서 test_default 가 저장된 결과(test_saved)와 같아야 한다.
"""
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from core.models import TEST, TRAIN, VAL, balanced_weight

DEFAULT_TASKS = ["europe", "korea_demand", "korea_ops", "safety"]
OUT = Path("tuned_params.json")
CORE = {"learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 20}  # core/models.py fit_lightgbm 의 값


def sample(rng, n_train):
    return {"learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.1)))),
            "num_leaves": int(rng.choice([7, 15, 31, 63])),
            "min_data_in_leaf": int(min(rng.choice([5, 10, 20, 50, 100, 200]), max(5, n_train // 20))),
            "feature_fraction": float(rng.uniform(0.5, 1.0)),
            "bagging_fraction": float(rng.uniform(0.6, 1.0)), "bagging_freq": 1,
            "lambda_l2": float(rng.choice([0.0, 0.1, 1.0, 10.0]))}


def fit(d, kind, params, rounds, stop, threads):
    import lightgbm as lgb
    tr, val = np.flatnonzero(d.split == TRAIN), np.flatnonzero(d.split == VAL)
    objective, metric = {"binary": ("binary", "binary_logloss"), "regression": ("regression", "l2")}[kind]
    y_tr, y_val = np.asarray(d.y[tr]), np.asarray(d.y[val])
    weight = balanced_weight if d.balanced else (lambda y: None)  # fit_lightgbm 과 같이 검증에도 같은 가중치
    train_set = lgb.Dataset(np.asarray(d.X[tr]), label=y_tr, weight=weight(y_tr), categorical_feature=d.cat_idx or "auto")
    valid_set = lgb.Dataset(np.asarray(d.X[val]), label=y_val, weight=weight(y_val), reference=train_set)
    p = {"objective": objective, "metric": metric, "num_threads": threads, "verbosity": -1, "seed": 0, "max_bin": 255, **params}
    return lgb.train(p, train_set, num_boost_round=rounds, valid_sets=[valid_set], callbacks=[lgb.early_stopping(stop, verbose=False)])


def score(kind, y, p):
    return float(average_precision_score(y, p)) if kind == "binary" else float(np.mean(np.abs(y - p)))


def save(results):
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(OUT)


def tune(slug, t, trials, threads):
    start, kind = time.time(), t["kind"]
    d = t["load"]()
    val, test = np.flatnonzero(d.split == VAL), np.flatnonzero(d.split == TEST)
    y_val, y_test = np.asarray(d.y[val]), np.asarray(d.y[test])
    X_val = np.asarray(d.X[val])
    better = (lambda a, b: a > b) if kind == "binary" else (lambda a, b: a < b)
    base = fit(d, kind, CORE, 500, 30, threads)  # fit_lightgbm 과 같은 반복 수와 조기 종료
    best = (CORE, score(kind, y_val, base.predict(X_val)), base)
    val_default = best[1]
    rng = np.random.default_rng(0)
    n_train = int((d.split == TRAIN).sum())
    for i in range(trials):
        params = sample(rng, n_train)
        model = fit(d, kind, params, 3000, 50, threads)
        v = score(kind, y_val, model.predict(X_val))
        if better(v, best[1]):
            best = (params, v, model)
        print(f"  {t['task']} {i + 1}/{trials} 검증 {v:.4f} (최고 {best[1]:.4f}, 기본 {val_default:.4f})", flush=True)
    X_test = np.asarray(d.X[test])
    saved = Path(f"outputs/{slug}/model_results/{t['task']}__lightgbm.json")
    metric = "pr_auc" if kind == "binary" else "mae"
    return {"slug": slug, "kind": kind, "metric": metric, "params": None if best[0] is CORE else best[0], "best_iteration": best[2].best_iteration,
            "val_default": val_default, "val_tuned": best[1],
            "test_default": score(kind, y_test, base.predict(X_test)), "test_tuned": score(kind, y_test, best[2].predict(X_test)),
            "test_saved": json.loads(saved.read_text(encoding="utf-8"))[metric] if saved.exists() else None,
            "trials": trials, "seconds": round(time.time() - start)}


def main():
    opt = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    trials, threads = int(opt.get("trials", 30)), int(opt.get("threads", 2))
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for arg in [a for a in sys.argv[1:] if not a.startswith("--")] or DEFAULT_TASKS:
        slug, _, only = arg.partition(":")
        for t in importlib.import_module(f"categories.{slug}.models").tasks():
            if (only and t["task"] != only) or t["kind"] == "multiclass":
                continue
            print(f"=== {slug}/{t['task']} 탐색 {trials}회", flush=True)
            r = results[t["task"]] = tune(slug, t, trials, threads)
            save(results)
            print(f"끝 {t['task']}: 검증 {r['val_default']:.4f} → {r['val_tuned']:.4f}, 평가 {r['test_default']:.4f} → {r['test_tuned']:.4f} (저장된 값 {r['test_saved']}), {r['seconds']}초", flush=True)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    assert all(5 <= sample(rng, 100)["min_data_in_leaf"] <= 5 for _ in range(20)), "행이 적으면 잎당 최소 행을 줄인다"
    assert score("regression", np.array([1.0, 3.0]), np.array([2.0, 2.0])) == 1.0
    main()
