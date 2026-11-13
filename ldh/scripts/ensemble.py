"""저장된 모델들의 예측을 평균한 앙상블. 다시 학습하지 않는다.

조합은 검증 구간으로만 고른다: 검증 지표 순으로 상위 k개 모델의 평균(k=2..전체) 중 가장 좋은 것.
단일 최고 모델보다 검증에서 나을 때만 '앙상블' 결과로 저장한다(평가 지표·건별 예측·다음 기간 예측, 화면의 모델 목록에 나옴).
이진 분류는 예측 점수 평균, 회귀는 예측값 평균, 다중 분류는 저장 모델로 다시 낸 클래스별 확률의 평균.

사용법: python -m ldh.scripts.ensemble [카테고리 또는 카테고리:과제 ...]   (기본: 저장 모델이 있는 전체)
"""
import importlib
import json
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, precision_recall_curve

from ldh.core.models import FUTURE, MODELS, TEST, TRAIN, VAL, binary_scores, load_model, merge_summary, multiclass_scores, regression_scores, save_predictions
from ldh.core.registry import discover
from ldh.core.store import Store

NAME = "ensemble"


def val_score(kind, y, p):
    """높을수록 좋은 검증 점수. 다중 분류는 클래스별 확률 평균의 macro F1."""
    if kind == "multiclass":
        return float(f1_score(y, np.asarray(p).argmax(axis=1), average="macro", zero_division=0))
    return float(average_precision_score(y, p)) if kind == "binary" else -float(mean_absolute_error(y, p))


def choose(kind, y_val, preds):
    """preds: {모델: 검증 예측}. (고른 모델 목록, 검증 점수, 단일 최고 점수)."""
    ranked = sorted(preds, key=lambda m: val_score(kind, y_val, preds[m]), reverse=True)
    single = val_score(kind, y_val, preds[ranked[0]])
    best = ([ranked[0]], single)
    for k in range(2, len(ranked) + 1):
        s = val_score(kind, y_val, np.mean([preds[m] for m in ranked[:k]], axis=0))
        if s > best[1]:
            best = (ranked[:k], s)
    return best[0], best[1], single


def check():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    members, score, single = choose("regression", y, {"a": y + 1, "b": y - 1, "c": y + 5})
    assert sorted(members) == ["a", "b"] and score == 0 and single == -1, "서로 반대로 틀린 두 모델의 평균이 뽑힘"
    members, _, _ = choose("regression", y, {"a": y, "b": y + 3})
    assert members == ["a"], "평균이 나쁘면 단일 모델"
    yc = np.array([0, 1, 1])
    a, b = np.array([[0.6, 0.4], [0.6, 0.4], [0.1, 0.9]]), np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    assert val_score("multiclass", yc, np.mean([a, b], axis=0)) == 1.0, "다중 분류는 확률을 평균해 argmax"


def run(slug, task_filter=None):
    store = Store(slug)
    results = store.dir / "model_results"
    for t in importlib.import_module(f"ldh.categories.{slug}.models").tasks():
        task, kind = t["task"], t["kind"]
        if task_filter and task != task_filter:
            continue
        rows = {m: json.loads((results / f"{task}__{m}.json").read_text(encoding="utf-8"))
                for m in MODELS if (results / f"{task}__{m}.json").exists() and (store.dir / "saved_models" / f"{task}__{m}" / "meta.json").exists()}
        if len(rows) < 2:
            print(f"{task}: 저장 모델 {len(rows)}개라 건너뜀", flush=True)
            continue
        start = time.time()
        d = t["load"]()
        val, test, future = (np.flatnonzero(d.split == s) for s in (VAL, TEST, FUTURE))
        y_val, y = np.asarray(d.y[val]), np.asarray(d.y[test])
        loaded = {m: load_model(store, task, m)[0] for m in rows}
        preds = {m: np.asarray(f(d, val), dtype=np.float64) for m, f in loaded.items()}
        members, score, single = choose(kind, y_val, preds)
        out = results / f"{task}__{NAME}.json"
        if len(members) == 1:
            print(f"{task}: 검증에서 평균이 단일 최고({rows[members[0]]['model']})보다 낫지 않아 저장하지 않음", flush=True)
            out.unlink(missing_ok=True)  # 예전 앙상블 결과가 남아 있으면 화면에서 빠지게 한다
            continue
        p_val = np.mean([preds[m] for m in members], axis=0)
        p = np.mean([np.asarray(loaded[m](d, test), dtype=np.float64) for m in members], axis=0)
        threshold = None
        if kind == "binary":
            pv, rv, tv = precision_recall_curve(y_val, p_val)
            threshold = float(tv[np.argmax((2 * pv * rv / np.maximum(pv + rv, 1e-12))[:-1])])
            scores = binary_scores(y, p, threshold)
        elif kind == "multiclass":
            scores = multiclass_scores(y, p)
        else:
            train_mean = float(np.nanmean(np.asarray(d.y[np.flatnonzero(d.split == TRAIN)])))
            scores = regression_scores(y, p, train_mean, None if d.baseline is None else np.asarray(d.baseline[test]))
        first = rows[members[0]]
        row = {k: first[k] for k in ("task", "title", "group", "train_period", "test_period", "note", "kind", "n_train", "n_val", "n_test")}
        row |= {"model": "앙상블", "members": [rows[m]["model"] for m in members], "val_score": score, "val_score_single": single,
                "seconds": sum(rows[m]["seconds"] for m in members), "fit_seconds": sum(rows[m].get("fit_seconds", 0) for m in members), "tuned": False, **scores}
        save_predictions(store, task, NAME, kind, d, test, p.astype(np.float32), threshold)
        if len(future):
            p_future = np.mean([np.asarray(loaded[m](d, future), dtype=np.float64) for m in members], axis=0)
            save_predictions(store, task, NAME, kind, d, future, p_future.astype(np.float32), threshold, prefix="forecast")
        out.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        main = {"binary": "pr_auc", "regression": "mae", "multiclass": "macro_f1"}[kind]
        best_single = (min if kind == "regression" else max)(r[main] for r in rows.values())
        print(f"{task}: {'+'.join(row['members'])} 평균. 평가 {main} {scores[main]:.4f} (단일 최고 {best_single:.4f}), {time.time() - start:.0f}초", flush=True)
    merge_summary(store)


if __name__ == "__main__":
    check()
    targets = sys.argv[1:] or [c["slug"] for c in discover()[0]]
    for target in targets:
        slug, _, task = target.partition(":")
        if (Store(slug).dir / "saved_models").exists():
            print(f"=== {slug}", flush=True)
            run(slug, task or None)
