"""LightGBM, XGBoost, RandomForest, LSTM 공통 학습 엔진. 평가 구간의 건별 예측값과 다음 기간 예측값을 저장한다.

RandomForest 는 수억 행을 처리하려고 LightGBM 의 rf 모드(부트스트랩 + 특성 샘플링, 히스토그램 방식)를 쓴다.
LSTM 입력은 같은 대상(기체, 공항, 노선 등)의 이전 기록이다. 예측 시점에 아직 모르는 값은 가린다.
"""
import gc
import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, mean_absolute_error, median_absolute_error,
                             precision_recall_curve, precision_score, r2_score, recall_score, roc_auc_score, root_mean_squared_error)

MODELS = ["lightgbm", "xgboost", "random_forest", "lstm"]
MODEL_NAMES = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "random_forest": "RandomForest", "lstm": "LSTM", "ensemble": "앙상블"}  # ensemble 은 학습하지 않고 ldh/scripts/ensemble.py 가 만든다
TRAIN, VAL, TEST, FUTURE = 0, 1, 2, 3
THREADS = min(8, os.cpu_count() or 8)  # 코랩은 코어가 적어 8개를 띄우면 오히려 느려진다
TUNED_PATH = Path(__file__).resolve().parents[1] / "config" / "tuned_params.json"  # ldh/config


def tuned_params(task):
    """tune_small.py(2026-11-11 삭제, 커밋 2d31ba4e)가 검증 구간만 보고 고른 LightGBM 설정. 없거나 기본 설정이 뽑힌 과제는 None."""
    if not TUNED_PATH.exists():
        return None
    return json.loads(TUNED_PATH.read_text(encoding="utf-8")).get(task, {}).get("params")


@dataclass
class Data:
    """X: 행×특성 float32 (범주는 정수 코드) / split: TRAIN·VAL·TEST·FUTURE / keys: 평가·미래 행에 함께 저장할 열."""
    X: np.ndarray
    columns: list
    y: np.ndarray
    split: np.ndarray
    cat_idx: list = field(default_factory=list)
    keys: dict = field(default_factory=dict)
    decode: dict = field(default_factory=dict)  # keys 중 정수 코드인 열의 이름표
    period: np.ndarray = None  # 평가·미래 행의 기간 라벨(문자열 또는 YYYYMM 정수). 기간별 실제 대 예측 집계에 사용
    classes: list = None  # 다중 분류 클래스 이름
    S: np.ndarray = None  # LSTM 시퀀스 값 (행×시퀀스 특성)
    prev: np.ndarray = None  # 이전 행 번호 (행×길이, 최근 순, 없으면 -1)
    known_t: np.ndarray = None  # 그 행의 시퀀스 값을 알게 되는 시각 (행×시퀀스 특성)
    pred_t: np.ndarray = None  # 예측 시점
    baseline: np.ndarray = None  # 과거 값 그대로 쓰기 기준선 (회귀, 전체 행 기준)
    train_period: str = ""
    test_period: str = ""
    group: str = ""
    note: str = ""
    balanced: bool = False
    delta: bool = False  # True 면 모델은 y - baseline(직전 값 대비 변화)을 배우고 예측에 baseline 을 더한다 (회귀 패널)
    meta: dict = field(default_factory=dict)  # 실서비스 입력을 만들 때 필요한 값 (범주 코드표, 판정 기준 등). 저장 모델과 함께 남긴다


def prev_indices(entity, order_time, k):
    """같은 대상의 이전 k개 행 번호. entity 가 음수면 대상 없음."""
    n = len(entity)
    prev = np.full((n, k), -1, dtype=np.int32)
    order = np.lexsort((order_time, entity))
    e = entity[order]
    for j in range(1, k + 1):
        same = (e[j:] == e[:-j]) & (e[j:] >= 0)
        prev[order[j:][same], j - 1] = order[:-j][same]
    return prev


def history_rows(prev, rows, limit, rng):
    """rows 의 시퀀스 입력으로 들어가는 이전 행 번호 (중복 없이 정렬, 최대 limit 개)."""
    hist = np.asarray(prev[rows]).ravel()
    hist = np.unique(hist[hist >= 0])
    return np.sort(rng.choice(hist, size=limit, replace=False)) if len(hist) > limit else hist


def only_year(d, year):
    """그 해의 행만 학습·검증·평가에 쓴다. 마지막 25% 달은 평가, 그 앞 한 달은 검증, 나머지는 학습. 달이 3개 미만이면 None.

    입력의 과거 값(직전 운항, 전년 값)은 그대로라 이전 해의 값이 입력으로 들어갈 수 있다.
    """
    period = np.asarray(d.period)
    if np.issubdtype(period.dtype, np.integer):
        ym = period.astype(np.int64)
    else:
        text = pd.Series(period).astype("string")
        ym = (pd.to_numeric(text.str[:4], errors="coerce") * 100 + pd.to_numeric(text.str[5:7], errors="coerce")).fillna(0).to_numpy(np.int64)
    source = np.asarray(d.split)
    used = np.isin(source, (TRAIN, VAL, TEST)) & (ym // 100 == year)
    months = np.unique(ym[used])
    if len(months) < 3:
        return None
    n_test = max(1, round(len(months) * 0.25))
    val_month, test_from = months[-n_test - 1], months[-n_test]
    split = np.full(len(ym), -1, dtype=np.int8)
    split[source == FUTURE] = FUTURE
    split[used & (ym < val_month)] = TRAIN
    split[used & (ym == val_month)] = VAL
    split[used & (ym >= test_from)] = TEST
    label = lambda m: f"{m // 100}-{m % 100:02d}"
    return replace(d, split=split, train_period=f"{label(months[0])}~{label(months[-n_test - 2])}", test_period=f"{label(test_from)}~{label(months[-1])}")


def balanced_weight(y):
    """sklearn class_weight='balanced' 와 같은 가중치."""
    values, counts = np.unique(y, return_counts=True)
    w = np.zeros(int(values.max()) + 1, dtype=np.float32)
    w[values.astype(int)] = len(y) / (len(values) * counts)
    return w[y.astype(int)]


def fit_lightgbm(d, kind, tr, val, rf=False, tuned=None):
    import lightgbm as lgb
    objective, metric = {"binary": ("binary", "binary_logloss"), "regression": ("regression", "l2"), "multiclass": ("multiclass", "multi_logloss")}[kind]
    y_tr = np.asarray(d.y[tr])
    params = {"objective": objective, "metric": metric, "learning_rate": 0.1, "num_leaves": 63, "num_threads": THREADS, "verbosity": -1, "seed": 0,
              "max_bin": 255, "min_data_in_leaf": 20}
    if kind == "multiclass":
        params["num_class"] = len(d.classes)
    if rf:
        params |= {"boosting": "rf", "bagging_freq": 1, "bagging_fraction": 0.632, "feature_fraction": 0.7, "num_leaves": 255, "learning_rate": 1.0}
    rounds, stop = (100 if rf else 500), 30
    if tuned:
        params |= tuned
        rounds, stop = 3000, 50  # 튜닝할 때와 같게. 학습률이 낮은 설정은 500회로 모자란다
    weight = balanced_weight(y_tr) if d.balanced else None
    cat = d.cat_idx or "auto"
    # 평가 행은 예측 때 다시 읽으므로 학습·검증 행만 메모리에 올린다
    train_set = lgb.Dataset(np.asarray(d.X[tr]), label=y_tr, weight=weight, categorical_feature=cat, free_raw_data=True)
    y_val = np.asarray(d.y[val])
    # 검증에도 같은 가중치를 줘야 한다. 없으면 손실이 첫 반복부터 나빠져 바로 조기 종료된다
    valid_set = lgb.Dataset(np.asarray(d.X[val]), label=y_val, weight=balanced_weight(y_val) if d.balanced else None, reference=train_set, free_raw_data=True)
    callbacks = [lgb.log_evaluation(5)]
    if not rf:
        callbacks.append(lgb.early_stopping(stop, verbose=True))
    model = lgb.train(params, train_set, num_boost_round=rounds, valid_sets=[valid_set], callbacks=callbacks)

    def predict(rows):
        return model.predict(np.asarray(d.X[rows]))
    predict.save = lambda folder: save_pkl(model, folder)
    return predict


def fit_xgboost(d, kind, tr, val):
    import xgboost as xgb
    y_tr = np.asarray(d.y[tr])
    objective = {"binary": "binary:logistic", "regression": "reg:squarederror", "multiclass": "multi:softprob"}[kind]
    params = {"objective": objective, "tree_method": "hist", "max_bin": 255, "grow_policy": "lossguide", "max_leaves": 63, "max_depth": 0,
              "eta": 0.1, "nthread": THREADS, "seed": 0}
    if kind == "multiclass":
        params["num_class"] = len(d.classes)
    types = ["c" if i in d.cat_idx else "q" for i in range(d.X.shape[1])]
    weight = balanced_weight(y_tr) if d.balanced else None
    dtrain = xgb.QuantileDMatrix(np.asarray(d.X[tr]), label=y_tr, weight=weight, feature_types=types, enable_categorical=True, nthread=THREADS, max_bin=255)
    y_val = np.asarray(d.y[val])
    dval = xgb.QuantileDMatrix(np.asarray(d.X[val]), label=y_val, weight=balanced_weight(y_val) if d.balanced else None, ref=dtrain, feature_types=types, enable_categorical=True, nthread=THREADS, max_bin=255)
    model = xgb.train(params, dtrain, num_boost_round=500, evals=[(dval, "val")], early_stopping_rounds=30, verbose_eval=5)
    del dtrain, dval
    best = (0, model.best_iteration + 1)

    def predict(rows):
        return model.predict(xgb.DMatrix(np.asarray(d.X[rows]), feature_types=types, enable_categorical=True, nthread=THREADS), iteration_range=best)
    predict.save = lambda folder: save_pkl(model, folder)
    return predict


def _progress(keras, total_steps, every=500):
    """에포크 안에서도 진행률과 남은 시간을 출력한다."""
    class Progress(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.start, self.epoch = time.time(), epoch

        def on_train_batch_end(self, batch, logs=None):
            if (batch + 1) % every == 0:
                used = time.time() - self.start
                print(f"    에포크 {self.epoch + 1} {batch + 1:,}/{total_steps:,} 배치, 손실 {logs['loss']:.4f}, 남은 시간 약 {used / (batch + 1) * (total_steps - batch - 1) / 60:.0f}분", flush=True)

        def on_epoch_end(self, epoch, logs=None):
            print(f"    에포크 {epoch + 1} 완료 {time.time() - self.start:.0f}초, " + ", ".join(f"{k} {v:.4f}" for k, v in logs.items()), flush=True)
    return Progress()


def lstm_inputs(norm, d, rows):
    """LSTM 입력. 학습과 저장 모델 예측이 같은 계산을 쓴다. 예측 시점에 아직 모르는 값은 가린다."""
    k, fs = d.prev.shape[1], d.S.shape[1]
    p = np.asarray(d.prev[rows])
    clipped = np.maximum(p, 0).ravel()
    values = ((np.asarray(d.S[clipped], dtype=np.float32) - norm["sm"]) / norm["ss"]).reshape(len(rows), k, fs)
    known = np.asarray(d.known_t[clipped]).reshape(len(rows), k, fs) <= np.asarray(d.pred_t[rows])[:, None, None]
    mask = known & (p[:, :, None] >= 0) & np.isfinite(values)
    values = np.where(mask, values, 0.0)
    seq = np.concatenate([values, mask.astype(np.float32)], axis=2)[:, ::-1, :].astype(np.float32)  # 과거 -> 최근 순서
    X = np.asarray(d.X[rows], dtype=np.float32)
    out = {"seq": np.clip(seq, -10, 10), "num": np.clip(np.nan_to_num((X[:, norm["num_idx"]] - norm["xm"]) / norm["xs"], nan=0.0, posinf=0.0, neginf=0.0), -10, 10).astype(np.float32)}
    for j, c in enumerate(d.cat_idx):
        size = int(norm["cat_size"][j])
        out[f"cat{j}"] = np.clip(np.nan_to_num(X[:, c], nan=size - 1), 0, size - 1).astype(np.int32)
    return out


def lstm_predict(model, norm, kind, d, rows):
    n_out = model.output_shape[-1]
    out = np.empty((len(rows), n_out) if n_out > 1 else len(rows), dtype=np.float32)
    for start in range(0, len(rows), 32768):
        p = model.predict_on_batch(lstm_inputs(norm, d, rows[start:start + 32768]))
        out[start:start + 32768] = p if n_out > 1 else p.ravel()
    return out * norm["y_std"] + norm["y_mean"] if kind == "regression" else out


def fit_lstm(d, kind, tr, val, seed=0):
    from tensorflow import keras

    n_rows = len(tr)
    k = d.prev.shape[1]
    fs = d.S.shape[1]
    num_idx = [i for i in range(d.X.shape[1]) if i not in d.cat_idx]
    # 정규화 값은 학습 행만으로 계산한다. 큰 데이터는 200만 행이면 값이 안정적이다 (학습 자체는 전체 행 사용).
    rng = np.random.default_rng(0)
    probe = np.sort(rng.choice(tr, size=min(2_000_000, n_rows), replace=False))
    x_probe = np.asarray(d.X[probe])
    xm, xs = np.nanmean(x_probe[:, num_idx], axis=0), np.nanstd(x_probe[:, num_idx], axis=0) + 1e-6
    # 학습 구간에서 값이 일정한 열은 표준편차가 0이라, 그대로 나누면 평가 구간 값이 폭주한다
    xm, xs = np.nan_to_num(xm), np.where(np.nan_to_num(xs, nan=1.0) < 1e-3, 1.0, np.nan_to_num(xs, nan=1.0))
    # 시퀀스 입력은 이전 행의 값이다. 자기 행으로 계산하면 결항편처럼 자기 값이 비는 과제에서 정규화가 망가진다
    s_probe = np.asarray(d.S[history_rows(d.prev, probe, 2_000_000, rng)], dtype=np.float32)
    sm, ss = np.nan_to_num(np.nanmean(s_probe, axis=0)), np.nan_to_num(np.nanstd(s_probe, axis=0), nan=1.0) + 1e-6
    y_probe = np.asarray(d.y[probe])
    y_mean, y_std = (float(np.nanmean(y_probe)), float(np.nanstd(y_probe)) + 1e-6) if kind == "regression" else (0.0, 1.0)
    cat_size = [int(np.nanmax(np.asarray(d.X[:, i][tr[:: max(1, n_rows // 2_000_000)]]))) + 2 for i in d.cat_idx]

    norm = {"sm": sm, "ss": ss, "xm": xm, "xs": xs, "num_idx": np.array(num_idx), "cat_size": np.array(cat_size), "y_mean": y_mean, "y_std": y_std}

    def inputs(rows):
        return lstm_inputs(norm, d, rows)

    batch = int(np.clip(n_rows // 200, 128, 4096))
    weights = None
    if d.balanced:
        values, counts = np.unique(y_probe, return_counts=True)
        weights = np.zeros(int(values.max()) + 1, dtype=np.float32)
        weights[values.astype(int)] = len(y_probe) / (len(values) * counts)

    class Batches(keras.utils.PyDataset):
        def __init__(self, rows, shuffle):
            super().__init__()
            self.rows, self.shuffle, self.rng = rows.copy(), shuffle, np.random.default_rng(seed)  # 다시 학습해도 같은 순서
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.rows) / batch))

        def on_epoch_end(self):
            if self.shuffle:
                self.rng.shuffle(self.rows)

        def __getitem__(self, i):
            rows = np.sort(self.rows[i * batch:(i + 1) * batch])  # memmap 은 정렬해 읽는 편이 빠르다
            y = np.asarray(d.y[rows], dtype=np.float32)
            target = (y - y_mean) / y_std if kind == "regression" else y
            if weights is not None:
                return inputs(rows), target, weights[y.astype(int)]
            return inputs(rows), target

    keras.utils.set_random_seed(seed)  # 층을 만들 때 가중치 초기값이 정해지므로 모델 생성 전에 고정한다
    seq_in = keras.Input((k, fs * 2), name="seq")
    num_in = keras.Input((len(num_idx),), name="num")
    parts, cat_inputs = [keras.layers.LSTM(32)(seq_in), num_in], []
    for j, size in enumerate(cat_size):
        c = keras.Input((), dtype="int32", name=f"cat{j}")
        cat_inputs.append(c)
        parts.append(keras.layers.Embedding(size, min(16, size // 2 + 1))(c))
    h = keras.layers.Dense(64, activation="relu")(keras.layers.Concatenate()(parts))
    n_out, activation, loss = {"binary": (1, "sigmoid", "binary_crossentropy"), "regression": (1, None, "mse"),
                               "multiclass": (len(d.classes or []), "softmax", "sparse_categorical_crossentropy")}[kind]
    model = keras.Model([seq_in, num_in] + cat_inputs, keras.layers.Dense(n_out, activation=activation)(h))
    model.compile(keras.optimizers.Adam(1e-3), loss=loss)
    train_seq = Batches(tr, True)
    epochs = 30 if n_rows < 1_000_000 else 3
    every = max(1, min(500, len(train_seq) // 5))
    model.fit(train_seq, validation_data=Batches(val, False), epochs=epochs, verbose=0,
              callbacks=[_progress(keras, len(train_seq), every), keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])

    def predict(rows):
        return lstm_predict(model, norm, kind, d, rows)

    def save(folder):
        model.save(folder / "model.keras")
        np.savez(folder / "norm.npz", **norm)
    predict.save = save
    return predict


def save_pkl(model, folder):
    # 부스터는 피클 안에 자체 모델 문자열을 담으므로 라이브러리 버전이 조금 달라도 읽힌다
    import joblib
    joblib.dump(model, folder / "model.pkl")


def load_tree(folder, name):
    """model.pkl 을 읽는다. pkl 이전에 저장한 폴더는 model.txt / model.ubj 를 읽는다."""
    if (folder / "model.pkl").exists():
        import joblib
        return joblib.load(folder / "model.pkl")
    if name == "xgboost":
        import xgboost as xgb
        return xgb.Booster(model_file=str(folder / "model.ubj"))
    import lightgbm as lgb
    return lgb.Booster(model_file=str(folder / "model.txt"))


def save_model(store, task, name, kind, d, predict, threshold):
    """모델과 예측에 필요한 값을 saved_models/<과제>__<모델>/ 에 저장한다."""
    folder = store.dir / "saved_models" / f"{task}__{name}"
    folder.mkdir(parents=True, exist_ok=True)
    predict.save(folder)
    meta = {"task": task, "model": name, "kind": kind, "columns": list(d.columns), "cat_idx": list(d.cat_idx), "threshold": threshold, "classes": d.classes, "decode": d.decode, "delta": d.delta, **d.meta}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)), encoding="utf-8")


def load_model(store, task, name):
    """저장한 모델을 불러와 (predict(d, rows), meta) 를 돌려준다. d 는 학습 때와 같은 구조의 Data 다."""
    folder = store.dir / "saved_models" / f"{task}__{name}"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    if name == "xgboost":
        import xgboost as xgb
        model = load_tree(folder, name)
        types = ["c" if i in meta["cat_idx"] else "q" for i in range(len(meta["columns"]))]

        def predict(d, rows):
            return model.predict(xgb.DMatrix(np.asarray(d.X[rows]), feature_types=types, enable_categorical=True, nthread=THREADS), iteration_range=(0, model.best_iteration + 1))
    elif name == "lstm":
        from tensorflow import keras
        model = keras.models.load_model(folder / "model.keras")
        with np.load(folder / "norm.npz") as saved:  # Windows 에서 파일이 잠기지 않도록 바로 닫는다
            norm = {k: saved[k] for k in saved.files}

        def predict(d, rows):
            return lstm_predict(model, norm, meta["kind"], d, rows)
    else:
        model = load_tree(folder, name)

        def predict(d, rows):
            return model.predict(np.asarray(d.X[rows]))
    if meta.get("delta"):
        change = predict
        return (lambda d, rows: add_baseline(lambda r: change(d, r), d)(rows)), meta
    return predict, meta


def binary_scores(y, proba, threshold):
    pred = proba >= threshold
    pr_auc = float(average_precision_score(y, proba))
    return {"roc_auc": float(roc_auc_score(y, proba)), "pr_auc": pr_auc, "baseline_pr_auc": float(np.mean(y)), "lift": pr_auc / float(np.mean(y)),
            "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)), "threshold": threshold}


def regression_scores(y, pred, train_mean, baseline):
    s = {"mae": float(mean_absolute_error(y, pred)), "rmse": float(root_mean_squared_error(y, pred)), "medae": float(median_absolute_error(y, pred)), "r2": float(r2_score(y, pred))}
    mean_mae = float(np.abs(y - train_mean).mean())
    s["mean_mae"] = mean_mae
    if baseline is not None:
        ok = np.isfinite(baseline)
        if ok.any():
            s["persistence_mae"] = float(np.abs(y[ok] - baseline[ok]).mean())
    s["baseline_mae"] = min(mean_mae, s.get("persistence_mae", mean_mae))
    s["improve"] = 1 - s["mae"] / s["baseline_mae"]
    return s


def multiclass_scores(y, proba):
    pred = proba.argmax(axis=1)
    counts = np.bincount(y.astype(int))
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)), "majority_rate": float(counts.max() / counts.sum())}


def train_one(d, kind, name, task=None, seed=0):
    """task 를 주면 tuned_params.json 의 LightGBM 설정을 쓴다. seed 는 LSTM 에만 쓴다(트리 모델은 0 고정)."""
    if d.delta:
        change = replace(d, y=np.asarray(d.y) - np.asarray(d.baseline), delta=False)
        if name == "lightgbm":
            # 튜닝값은 값 자체를 맞출 때 고른 것이라 쓰지 않는다. 변화량은 큰 값이 튀어 평가 지표(MAE)에 맞춘 L1 이 낫다 (trial_weak_tasks)
            tr, val = np.flatnonzero(change.split == TRAIN), np.flatnonzero(change.split == VAL)
            inner = fit_lightgbm(change, kind, tr, val, tuned={"objective": "l1", "metric": "l1", "learning_rate": 0.05})
        else:
            inner = train_one(change, kind, name, None, seed)
        predict = add_baseline(lambda rows: inner(rows), d)
        predict.save = inner.save
        return predict
    tr, val = np.flatnonzero(d.split == TRAIN), np.flatnonzero(d.split == VAL)
    if name == "lightgbm":
        return fit_lightgbm(d, kind, tr, val, tuned=tuned_params(task) if task else None)
    if name == "random_forest":
        return fit_lightgbm(d, kind, tr, val, rf=True)
    if name == "xgboost":
        return fit_xgboost(d, kind, tr, val)
    if d.S is None:
        raise ValueError("LSTM 시퀀스 입력이 없는 데이터")
    return fit_lstm(d, kind, tr, val, seed)


def add_baseline(predict_change, d):
    """변화량 예측 + 직전 값. predict_change(rows) -> 변화량."""
    return lambda rows: predict_change(rows) + np.asarray(d.baseline)[rows]


def run_model(store, task, title, kind, d, name, seed=0):
    """한 모델을 학습해 지표와 예측값을 저장하고 결과 행을 돌려준다."""
    start = time.time()
    predict = train_one(d, kind, name, task, seed)
    fit_seconds = time.time() - start
    val, test = np.flatnonzero(d.split == VAL), np.flatnonzero(d.split == TEST)
    p = predict(test)
    y = np.asarray(d.y[test])
    threshold = None
    if kind == "binary":
        pv, rv, tv = precision_recall_curve(np.asarray(d.y[val]), predict(val))
        threshold = float(tv[np.argmax((2 * pv * rv / np.maximum(pv + rv, 1e-12))[:-1])])
        scores = binary_scores(y, p, threshold)
    elif kind == "regression":
        train_mean = float(np.nanmean(np.asarray(d.y[np.flatnonzero(d.split == TRAIN)])))
        scores = regression_scores(y, p, train_mean, None if d.baseline is None else np.asarray(d.baseline[test]))
    else:
        scores = multiclass_scores(y, p)
    row = {"task": task, "title": title, "model": MODEL_NAMES[name], "group": d.group, "train_period": d.train_period, "test_period": d.test_period, "note": d.note,
           "kind": kind, "n_train": int((d.split == TRAIN).sum()), "n_val": len(val), "n_test": len(test), "seconds": round(time.time() - start), "fit_seconds": round(fit_seconds),
           "tuned": name == "lightgbm" and tuned_params(task) is not None, **scores}
    save_model(store, task, name, kind, d, predict, threshold)
    save_predictions(store, task, name, kind, d, test, p, threshold)
    future = np.flatnonzero(d.split == FUTURE)
    if len(future):
        save_predictions(store, task, name, kind, d, future, predict(future), threshold, prefix="forecast")
    results = store.dir / "model_results"
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{task}__{name}.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    return row


def save_predictions(store, task, name, kind, d, rows, p, threshold, prefix="pred"):
    """건별 예측값과 기간별 실제 대 예측 집계를 저장한다. 큰 평가 구간도 표본 없이 전부 저장한다."""
    t = pd.DataFrame({k: np.asarray(v)[rows] for k, v in d.keys.items()})
    for c, names in d.decode.items():
        t[c] = pd.Categorical.from_codes(t[c].astype(int), categories=pd.Index(names).drop_duplicates()) if len(set(names)) == len(names) else np.array(names, dtype=object)[t[c].astype(int)]
    if d.period is not None:
        period = np.asarray(d.period)[rows]
        t["기간"] = pd.Series(period).map(lambda p: f"{p // 100}-{p % 100:02d}") if np.issubdtype(period.dtype, np.integer) else period
    y = np.asarray(d.y[rows], dtype=np.float32)
    if kind == "binary":
        t["실제"], t["예측 점수"] = y, p.astype(np.float32)
        t["예측"] = (p >= threshold).astype(np.float32)
    elif kind == "regression":
        t["실제"], t["예측"] = y, p.astype(np.float32)
    else:
        names = np.array(d.classes, dtype=object)
        t["실제"] = [names[int(v)] if np.isfinite(v) else None for v in y]
        t["예측"] = names[p.argmax(axis=1)]
        t["예측 확률"] = p.max(axis=1).astype(np.float32)
    store.save_table(f"{prefix}_{task}__{name}", t)
    if prefix == "pred" and d.period is not None:
        if kind == "multiclass":
            agg = pd.concat([t.groupby("기간")["실제"].value_counts(normalize=True).rename("비율").reset_index().rename(columns={"실제": "클래스"}).assign(구분="실제"),
                             t.groupby("기간")["예측"].value_counts(normalize=True).rename("비율").reset_index().rename(columns={"예측": "클래스"}).assign(구분="예측")])
        else:
            agg = t.groupby("기간", as_index=False).agg(실제=("실제", "mean"), 예측=("예측 점수" if kind == "binary" else "예측", "mean"), 건수=("실제", "size"))
        store.save_table(f"predagg_{task}__{name}", agg)


def collect_rows(store):
    """저장된 결과를 모두 읽어 모델 표 행으로 돌려준다."""
    results = store.dir / "model_results"
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(results.glob("*.json"))] if results.exists() else []


def merge_summary(store):
    """파이프라인의 기존 요약은 두고 4개 모델 결과만 갱신한다."""
    path = store.dir / "summary.json"
    body = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"headline": "", "kpis": [], "models": [], "notes": [], "sources": [], "updated": ""}
    body["models"] = [m for m in body["models"] if "model" not in m] + collect_rows(store)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    tmp.replace(path)


def self_check():
    prev = prev_indices(np.array([0, 0, 1, 0, -1]), np.array([3, 1, 1, 2, 9]), 2)
    assert prev[0].tolist() == [3, 1], "최근 순으로 같은 대상의 이전 행이어야 함"
    assert prev[1].tolist() == [-1, -1] and prev[2].tolist() == [-1, -1] and prev[4].tolist() == [-1, -1], "이전 행이 없으면 -1"
    assert np.allclose(balanced_weight(np.array([0.0, 0.0, 0.0, 1.0])), [2 / 3, 2 / 3, 2 / 3, 2.0]), "소수 클래스 가중치"
    hist_prev = np.array([[3, 1], [1, -1], [-1, -1]])
    assert history_rows(hist_prev, np.array([0, 1]), 9, np.random.default_rng(0)).tolist() == [1, 3], "이전 행만, 중복 없이"
    assert len(history_rows(hist_prev, np.array([0, 1]), 1, np.random.default_rng(0))) == 1, "최대 개수"
    assert len(history_rows(hist_prev, np.array([2]), 9, np.random.default_rng(0))) == 0, "이전 행이 없는 경우"
    empty = np.zeros((6, 1))
    labels = ["2025-12", "2026-01", "2026-02", "2026-03-05", "2026-04", "2026-05"]
    picked = only_year(Data(X=empty, columns=["a"], y=empty[:, 0], split=np.array([TRAIN, TRAIN, -1, VAL, TEST, FUTURE]), period=np.array(labels, dtype=object)), 2026)
    assert picked.split.tolist() == [-1, TRAIN, -1, VAL, TEST, FUTURE], "다른 해·제외 행·미래 행의 처리"
    assert picked.train_period == "2026-01~2026-01" and picked.test_period == "2026-04~2026-04", "기간 표시"
    numbers = Data(X=empty, columns=["a"], y=empty[:, 0], split=np.array([TRAIN] * 6), period=np.array([202512, 202601, 202601, 202602, 202603, 202603]))
    assert only_year(numbers, 2026).split.tolist() == [-1, TRAIN, TRAIN, VAL, TEST, TEST], "정수 YYYYMM 기간"
    assert only_year(numbers, 2024) is None, "자료가 없는 해"
    gc.collect()
