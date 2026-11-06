"""시계열 패널(대상×기간) 데이터를 core.models.Data 로 바꾼다. 다음 기간 행을 추가해 미래 예측값도 만든다."""
import numpy as np
import pandas as pd

from core.models import FUTURE, TEST, TRAIN, VAL, Data, prev_indices


def panel_data(df, keys, target, train_range, val_range, test_range, lags, extra=(), extra_future=None, step=1, seq_len=12, label=None, kind="regression", y_fn=None,
               train_period="", test_period="", group="", note="", balanced=False):
    """df: keys + t + target. t 는 정수 기간(월 번호, 날짜 서수).

    입력은 과거 값(lags)과 범주(keys), 달력 값(extra)뿐이라 예측 시점에 모두 알 수 있다. 마지막 기간 다음 기간을 FUTURE 로 만든다.
    y_fn: 분류일 때 값 -> 정답 변환. label: 기간 라벨 함수(t -> 문자열).
    """
    base = df[keys + ["t", target] + list(extra)].dropna(subset=[target]).copy()
    t_max = int(base.t.max())
    last = base[base.t == t_max][keys].drop_duplicates()
    future = last.assign(t=t_max + step)
    future[target] = np.nan
    for col, fn in (extra_future or {}).items():
        future[col] = fn(t_max + step)
    d = pd.concat([base, future], ignore_index=True)
    for lag in lags:
        prev = base[keys + ["t", target]].assign(t=base.t + lag).rename(columns={target: f"lag{lag}"})
        d = d.merge(prev, on=keys + ["t"], how="left")

    split = np.full(len(d), -1, dtype=np.int8)
    split[d.t.between(*train_range).to_numpy()] = TRAIN
    split[d.t.between(*val_range).to_numpy()] = VAL
    split[d.t.between(*test_range).to_numpy()] = TEST
    split[(d.t == t_max + step).to_numpy()] = FUTURE
    split[(split == TEST) & d[f"lag{lags[0]}"].isna().to_numpy()] = -1  # 기준선과 같은 행으로 비교

    # 범주 코드는 학습 구간 기준으로 만든다. 처음 보는 값은 마지막 코드.
    train_rows = d[split == TRAIN]
    columns, cat_idx, arrays = [], [], []
    for c in keys:
        cats = train_rows[c].astype("string").value_counts().index[:250]
        codes = pd.Categorical(d[c].astype("string"), categories=cats).codes.astype(np.float32)
        codes[codes < 0] = len(cats)
        cat_idx.append(len(columns))
        columns.append(c)
        arrays.append(codes)
    for c in [f"lag{lag}" for lag in lags] + list(extra):
        columns.append(c)
        arrays.append(pd.to_numeric(d[c], errors="coerce").to_numpy(np.float32))
    X = np.column_stack(arrays).astype(np.float32)

    values = d[target].to_numpy(np.float32)
    entity = d.groupby(keys, sort=False, dropna=False).ngroup().to_numpy(np.int64)
    S = values.reshape(-1, 1).copy()
    prev = prev_indices(entity, d.t.to_numpy(np.int64), seq_len)
    t = d.t.to_numpy(np.int64)
    y = values if y_fn is None else y_fn(values)
    fmt = label or str
    return Data(X=X, columns=columns, y=np.asarray(y, dtype=np.float32), split=split, cat_idx=cat_idx,
                keys={c: d[c].astype("string").to_numpy() for c in keys}, period=d.t.map(fmt).to_numpy(), S=S, prev=prev,
                known_t=t.reshape(-1, 1), pred_t=t, baseline=d[f"lag{lags[0]}"].to_numpy(np.float32),
                train_period=train_period, test_period=test_period, group=group, note=note, balanced=balanced)


def month_label(t):
    t = int(t)
    return f"{(t - 1) // 12}-{(t - 1) % 12 + 1:02d}"


def day_label(t):
    return pd.Timestamp.fromordinal(int(t)).strftime("%Y-%m-%d")


def month_ranges(t_min, t_max, test_months=12, val_months=6):
    """시간순 분할: 마지막 test_months 평가, 그 앞 val_months 검증, 나머지 학습."""
    test_start = t_max - test_months + 1
    val_start = test_start - val_months
    return (t_min, val_start - 1), (val_start, test_start - 1), (test_start, t_max)
