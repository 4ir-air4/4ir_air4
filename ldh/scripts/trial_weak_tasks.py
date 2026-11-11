"""기준선(직전 달 값)을 못 이기거나 겨우 이긴 월별 패널 과제 6개에서 대안 방법을 비교한다. 고르는 기준은 검증 구간 MAE.

방법
  persistence   직전 달 값 (지금 기준선)
  seasonal_a    직전 달 값 + a x (작년 같은 달 - 작년 전달). a 는 검증 구간에서 고름
  arima011      대상별 ARIMA(0,1,1) (= 단순 지수평활). 학습 구간으로 계수를 맞추고, 이후는 계수 고정으로 한 달 앞 예측
  lgbm_level    LightGBM, 값 자체를 맞춤 (지금 방식과 같은 성격)
  lgbm_delta    LightGBM, 직전 달 대비 변화량을 맞춤 -> 직전 달 값 + 예측 변화량
평가 행: 직전 달 값이 있는 검증·평가 행 (지금 기준선과 같은 행). 결과: ldh/results/trial_weak_tasks.json

사용법: python -m ldh.scripts.trial_weak_tasks [과제 ...]
"""
import importlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ldh.core.models import TEST, TRAIN, VAL

OUT = Path("ldh/results/trial_weak_tasks.json")
TASKS = {"type_passengers": "korea_demand", "kac_delay_hour": "korea_ops", "euro_taxi_in": "europe",
         "icn_passengers": "korea_demand", "euro_taxi_out": "europe", "euro_asma": "europe"}
LAGS = (1, 2, 3, 12, 13)


def frame(d):
    """Data -> 대상(sid)·월(t)·값(y)·분할 표."""
    f = pd.DataFrame({k: np.asarray(v) for k, v in d.keys.items()})
    per = pd.Series(np.asarray(d.period)).astype(str)
    f["t"] = per.str[:4].astype(int) * 12 + per.str[5:7].astype(int)
    f["y"] = np.asarray(d.y, dtype=float)
    f["split"] = np.asarray(d.split)
    f["sid"] = f.groupby(list(d.keys), sort=False, dropna=False).ngroup()
    return f[f.split.isin([TRAIN, VAL, TEST])].reset_index(drop=True)


def add_lags(f, lags=LAGS):
    """같은 대상의 k 달 전 값 (그 달 행이 없으면 NaN)."""
    base = f.loc[f.y.notna(), ["sid", "t", "y"]]
    for k in lags:
        f = f.merge(base.assign(t=base.t + k).rename(columns={"y": f"lag{k}"}), on=["sid", "t"], how="left")
    return f


def seasonal(f, a):
    step = (f.lag12 - f.lag13).fillna(0)
    return f.lag1 + a * step


def arima011(f):
    """대상별 ARIMA(0,1,1). 학습 구간 관측이 6개 미만이거나 맞추기 실패면 직전 달 값."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    pred = f.lag1.to_numpy(dtype=float).copy()
    for _, g in f.groupby("sid"):
        train_t = g.t[g.split == TRAIN]
        if len(train_t) == 0:
            continue
        s = g.set_index("t").y.reindex(range(g.t.min(), g.t.max() + 1))
        if s.loc[:train_t.max()].notna().sum() < 6:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(s.loc[:train_t.max()].to_numpy(), order=(0, 1, 1)).fit(disp=False)
                full = SARIMAX(s.to_numpy(), order=(0, 1, 1)).filter(res.params)
            one_step = pd.Series(full.predict(), index=s.index)  # t 의 예측은 t-1 까지의 값만 씀
        except Exception:
            continue
        p = one_step.reindex(g.t).to_numpy()
        ok = np.isfinite(p)
        pred[g.index[ok]] = p[ok]
    return pred


def lgbm(f, keys, delta):
    import lightgbm as lgb
    X = pd.DataFrame({f"lag{k}": f[f"lag{k}"] for k in LAGS})
    X["month"] = (f.t - 1) % 12 + 1
    X["d1"], X["d12"] = f.lag1 - f.lag2, f.lag12 - f.lag13
    X["r12"] = X.d12 / f.lag13.where(f.lag13.abs() > 1e-9)
    tr_rows = f.split == TRAIN
    for k in keys:  # 범주 코드는 학습 구간 기준
        cats = f.loc[tr_rows, k].astype("string").value_counts().index
        X[k] = pd.Categorical(f[k].astype("string"), categories=cats).codes
    target = f.y - f.lag1 if delta else f.y
    tr, va = tr_rows & target.notna(), (f.split == VAL) & target.notna()
    params = {"objective": "l1" if delta else "l2", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 20,
              "feature_fraction": 0.8, "verbosity": -1, "seed": 0, "num_threads": 4}
    model = lgb.train(params, lgb.Dataset(X[tr], target[tr], categorical_feature=list(keys)), 2000,
                      valid_sets=[lgb.Dataset(X[va], target[va], categorical_feature=list(keys))],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    p = model.predict(X)
    return p + f.lag1.to_numpy() if delta else p


def evaluate(f, preds):
    rows = {}
    for name, p in preds.items():
        r = {}
        for part, code in (("val", VAL), ("test", TEST)):
            m = ((f.split == code) & f.lag1.notna() & f.y.notna()).to_numpy()
            r[f"{part}_mae"] = float(np.abs(f.y.to_numpy()[m] - np.asarray(p)[m]).mean())
            r[f"{part}_rows"] = int(m.sum())
        rows[name] = r
    return rows


def check():
    f = pd.DataFrame({"sid": [0, 0, 0, 1], "t": [1, 2, 4, 2], "y": [10.0, 20.0, 40.0, 5.0], "split": [TRAIN] * 4})
    g = add_lags(f, (1, 2))
    assert np.isnan(g.lag1[0]) and g.lag1[1] == 10 and np.isnan(g.lag1[2]), "빈 달은 NaN"
    assert g.lag2[2] == 20 and np.isnan(g.lag1[3]), "2달 전, 다른 대상과 섞이지 않음"
    assert seasonal(g.assign(lag12=[1.0, np.nan, 3, 4], lag13=[0.0, 1, 1, 4]), 0.5)[1] == 10, "작년 값 없으면 직전 달 값"


def run(task):
    slug = TASKS[task]
    t = next(t for t in importlib.import_module(f"ldh.categories.{slug}.models").tasks() if t["task"] == task)
    d = t["load"]()
    keys = list(d.keys)
    f = add_lags(frame(d))
    start = time.time()
    preds = {"persistence": f.lag1.to_numpy()}
    for a in (0.25, 0.5, 0.75, 1.0):
        preds[f"seasonal_{a}"] = seasonal(f, a).to_numpy()
    preds["arima011"] = arima011(f)
    preds["lgbm_level"] = lgbm(f, keys, delta=False)
    preds["lgbm_delta"] = lgbm(f, keys, delta=True)
    rows = evaluate(f, preds)
    best = min(rows, key=lambda k: rows[k]["val_mae"])
    return {"methods": rows, "chosen": best, "seconds": round(time.time() - start)}


if __name__ == "__main__":
    check()
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for task in sys.argv[1:] or list(TASKS):
        results[task] = run(task)
        r = results[task]
        print(f"\n{task} (검증으로 고름: {r['chosen']})")
        for name, m in r["methods"].items():
            print(f"  {name:16} 검증 MAE {m['val_mae']:12.4f}   평가 MAE {m['test_mae']:12.4f}")
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
