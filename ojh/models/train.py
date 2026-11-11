"""항공 교통량(5분 뒤 항공기 수) 예측 모델 학습.

사용법
    python models/train.py                     # current + historical 모두 학습 (Optuna 튜닝 포함)
    python models/train.py current             # 실시간(current)만 학습
    python models/train.py --no-tune           # 튜닝 없이, 직전에 저장된 최적 파라미터(없으면 기본값)로 학습
    python models/train.py historical --no-tune

app.py의 [재학습] 버튼도 이 파일의 train_dataset()을 직접 호출합니다.
(버튼을 빠르게 하고 싶으면 train_dataset(key, config, tune=False)로 호출하세요.)
"""

from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # 로그는 아래 콜백에서 직접 출력
    OPTUNA_AVAILABLE = True
except ImportError:  # pip install optuna
    OPTUNA_AVAILABLE = False

warnings.filterwarnings("ignore")

# 출력 버퍼링 방지: 모든 print가 즉시 터미널에 나오도록 한다.
# (파이프/Streamlit 등에서 실행해도 로그가 멈춘 것처럼 보이지 않게 함)
print = partial(print, flush=True)


def step(message, since=None):
    """시각이 붙은 진행 로그. since를 주면 경과 시간(초)도 함께 표시한다."""
    now = datetime.now().strftime("%H:%M:%S")
    tail = f"  ({time.time() - since:.1f}초)" if since is not None else ""
    print(f"   [{now}] {message}{tail}")


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
SAVE_ROOT = ROOT_DIR / "models" / "saved"

DATASETS = {
    "current": {
        "label": "현재 (실시간 수집)",
        "path": RAW_DIR / "aircraftlive_counts.csv",
    },
    "historical": {
        "label": "과거 (BTS TranStats 샘플)",
        "path": RAW_DIR / "btstranstats_counts.csv",
    },
}

# ---------------------------------------------------------
# 학습 설정
# ---------------------------------------------------------
INTERVAL_MINUTES = 5
TEST_RATIO = 0.15          # 맨 뒤 15%는 마지막 최종 평가용(테스트)
MAX_CV_SPLITS = 5          # 교차검증 최대 fold 수 (데이터가 적으면 자동으로 줄임)
MIN_FOLD_SIZE = 10         # 검증 fold 하나당 최소 샘플 수
MAX_GAP_STEPS = 6          # 이 칸 수 이하의 짧은 결측은 직전값으로 메움 (5분 x 6 = 30분)
MIN_ROWS = 60              # 5분 관측값 최소 개수
MIN_SAMPLES = 40           # 피처 생성 후 학습 가능한 최소 샘플 수
RECOMMENDED_SAMPLES = 2000 # 약 1주일치. 이보다 적으면 결과가 불안정하다고 경고
MAX_MISSING_RATIO = 0.5    # 결측이 이 비율을 넘는 추가 피처는 사용하지 않음

LAGS = [1, 2, 3, 6, 12]
BASELINE_NAME = "Baseline (직전값)"

# ---------------------------------------------------------
# Optuna 튜닝 설정 (모델 하나당)
# ---------------------------------------------------------
TUNE_N_TRIALS = 30          # 시도 횟수 상한
TUNE_TIMEOUT_SEC = 180      # 시간 상한(초). 둘 중 먼저 도달하는 쪽에서 멈춤
TUNE_MAX_SPLITS = 3         # 튜닝 중 교차검증 fold 수 (속도를 위해 최종 CV보다 적게)
TUNE_MAX_SAMPLES = 50_000   # 튜닝에는 dev 구간의 '가장 최근' 이만큼만 사용 (과거 CSV가 매우 크기 때문)
TUNE_SEED = 42
BEST_PARAMS_FILE = "optuna_best_params.json"

WEATHER_COLUMNS = [
    "temp_mean",
    "precip_mean",
    "cloud_mean",
    "wind_mean",
    "gust_mean",
    "precip_max",
    "gust_max",
    "rain_share",
    "severe_weather_idx",
]

EXTRA_FEATURE_COLUMNS = [
    "avg_dep_delay",
    "cancelled_count",
    "diverted_count",
    "avg_distance",
    "rolling_delay_30m",
    "rolling_cancel_1h",
    "is_holiday",
]

MODEL_CONFIGS = {
    "Linear Regression": {
        "slug": "linear_regression",
        "model": LinearRegression(),
    },
    "Random Forest": {
        "slug": "random_forest",
        "model": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    },
    "XGBoost": {
        "slug": "xgboost",
        "model": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
    },
    "LightGBM": {
        "slug": "lightgbm",
        "model": LGBMRegressor(
            n_estimators=300,
            max_depth=-1,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        ),
    },
}


# ---------------------------------------------------------
# 0. Optuna 하이퍼파라미터 튜닝
# ---------------------------------------------------------
# 시작점(= 기존 기본값). 첫 번째 trial로 넣어서, 튜닝 결과가 기본값보다 나빠지지 않게 한다.
DEFAULT_START = {
    "Random Forest": {
        "n_estimators": 200, "max_depth": 12, "min_samples_leaf": 2, "max_features": 1.0,
    },
    "XGBoost": {
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.9,
        "colsample_bytree": 0.9, "min_child_weight": 1, "reg_lambda": 1.0,
    },
    "LightGBM": {
        "n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20,
        "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0,
    },
}  # Linear Regression은 튜닝할 하이퍼파라미터가 없어 제외


def suggest_params(model_name, trial):
    """모델별 탐색 범위. FixedTrial로도 호출되므로 trial.suggest_*만 사용한다."""
    if model_name == "Random Forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,  # LightGBM은 이 값이 있어야 subsample이 적용됨
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    return {}


def make_model(model_name, params=None):
    """기본 설정 모델을 복제한 뒤 params만 덮어쓴다."""
    return clone(MODEL_CONFIGS[model_name]["model"]).set_params(**(params or {}))


def tune_model(model_name, X_dev, yd_dev, n_splits, save_dir, slug):
    """Optuna(TPE)로 모델 하나의 하이퍼파라미터를 찾는다. 반환: 최적 params dict.

    - dev 구간만 사용 (테스트 구간은 절대 보지 않음 -> 누수 방지)
    - 시계열이므로 TimeSeriesSplit 사용, 목표값은 fold 평균 MAE
    - 시간이 오래 걸리는 데이터는 최근 TUNE_MAX_SAMPLES개만 사용
    """
    X_t = X_dev.iloc[-TUNE_MAX_SAMPLES:]
    y_t = yd_dev.iloc[-TUNE_MAX_SAMPLES:]
    folds = list(TimeSeriesSplit(n_splits=min(n_splits, TUNE_MAX_SPLITS)).split(X_t))
    step(f"{model_name} | Optuna 튜닝 시작 (최대 {TUNE_N_TRIALS}회 / {TUNE_TIMEOUT_SEC}초, 샘플 {len(X_t):,}개)")

    def objective(trial):
        params = suggest_params(model_name, trial)
        maes = []
        for i, (train_idx, val_idx) in enumerate(folds):
            model = make_model(model_name, params)
            model.fit(X_t.iloc[train_idx], y_t.iloc[train_idx])
            # 레벨 MAE와 델타 MAE는 같다 (실제 - 예측 오차가 동일). 속도를 위해 델타로 계산.
            maes.append(mean_absolute_error(y_t.iloc[val_idx], model.predict(X_t.iloc[val_idx])))
            trial.report(float(np.mean(maes)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(maes))

    def log_trial(study, trial):
        state = trial.state.name
        value = f"MAE {trial.value:.3f}" if trial.value is not None else state
        try:
            best = f"{study.best_value:.3f}"
        except ValueError:
            best = "-"
        print(f"      trial {trial.number + 1:>2}/{TUNE_N_TRIALS} | {value} | 현재 최고 {best}")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=TUNE_SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.enqueue_trial(DEFAULT_START[model_name])  # 기존 기본값을 첫 시도로
    study.optimize(
        objective,
        n_trials=TUNE_N_TRIALS,
        timeout=TUNE_TIMEOUT_SEC,
        callbacks=[log_trial],
    )

    try:
        best_params = suggest_params(model_name, optuna.trial.FixedTrial(study.best_params))
        best_value = study.best_value
    except ValueError:  # 완료된 trial이 하나도 없는 경우
        step(f"{model_name} | 완료된 trial이 없어 기본값을 사용합니다.")
        return {}

    study.trials_dataframe().to_csv(
        save_dir / f"optuna_trials_{slug}.csv", index=False, encoding="utf-8-sig"
    )
    step(f"{model_name} | 튜닝 완료 (최고 MAE {best_value:.3f}) -> {best_params}")
    return best_params


def load_saved_params(save_dir):
    path = save_dir / BEST_PARAMS_FILE
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


# ---------------------------------------------------------
# 1. 데이터 불러오기 / 정리
# ---------------------------------------------------------
def load_and_prepare_data(csv_path, min_rows=MIN_ROWS):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_columns = {"timestamp", "aircraft_count"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV에 필요한 컬럼이 없습니다: {sorted(missing)}\n"
            f"현재 컬럼: {list(df.columns)}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["aircraft_count"] = pd.to_numeric(df["aircraft_count"], errors="coerce")

    df = df.dropna(subset=["timestamp", "aircraft_count"])
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    if df.empty:
        raise ValueError("유효한 학습 데이터가 없습니다.")

    # 추가 피처: 숫자로 변환하되, 결측을 0으로 덮어쓰지 않는다.
    # (날씨 API가 실패해서 비어 있던 값이 '기온 0도'로 학습되는 것을 막기 위함)
    candidates = [c for c in WEATHER_COLUMNS + EXTRA_FEATURE_COLUMNS if c in df.columns]
    extra_cols = []
    for column in candidates:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().mean() > MAX_MISSING_RATIO:
            print(f"   [제외] '{column}' 컬럼은 결측이 {MAX_MISSING_RATIO:.0%}를 넘어 사용하지 않습니다.")
            continue
        extra_cols.append(column)

    # 5분 간격으로 맞추기
    frame = (
        df.set_index("timestamp")[["aircraft_count"] + extra_cols]
        .resample(f"{INTERVAL_MINUTES}min")
        .last()
    )
    counts = frame["aircraft_count"]

    # 짧은 결측(<= MAX_GAP_STEPS 칸)만 직전값으로 메우고, 긴 공백은 segment로 분리
    observed = counts.notna()
    group = observed.cumsum()
    gap_len = counts.isna().groupby(group).transform("sum")
    keep = observed | ((gap_len <= MAX_GAP_STEPS) & (group > 0))

    out = frame.ffill()[keep]

    # 남은 결측(맨 앞 행 등)은 해당 컬럼의 중앙값으로 채움
    for column in extra_cols:
        if out[column].isna().any():
            out[column] = out[column].fillna(out[column].median())

    out = out.reset_index()

    step = pd.Timedelta(minutes=INTERVAL_MINUTES)
    out["segment"] = (out["timestamp"].diff() != step).cumsum()

    if len(out) < min_rows:
        raise ValueError(
            f"학습 데이터가 너무 적습니다: {len(out)}개\n"
            f"최소 {min_rows}개 정도의 5분 관측값이 필요합니다."
        )

    out.attrs["extra_cols"] = extra_cols
    return out


# ---------------------------------------------------------
# 2. 피처 만들기
# ---------------------------------------------------------
def make_features(df):
    """모든 후보 피처와 타깃을 계산한다. (어떤 피처를 쓸지는 choose_feature_columns가 결정)"""
    data = df.copy()
    ts = data["timestamp"]

    hour = ts.dt.hour + ts.dt.minute / 60
    weekday = ts.dt.dayofweek
    month = ts.dt.month

    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    data["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    data["month_sin"] = np.sin(2 * np.pi * month / 12)
    data["month_cos"] = np.cos(2 * np.pi * month / 12)

    by_seg = data.groupby("segment")["aircraft_count"]

    # lag_0 = '지금' 값. 5분 뒤를 맞출 때 가장 중요한 정보다.
    data["lag_0"] = data["aircraft_count"]
    for lag in LAGS:
        data[f"lag_{lag}"] = by_seg.shift(lag)

    data["diff_1"] = data["lag_0"] - data["lag_1"]  # 최근 5분간 변화량

    # 지금 값까지 포함한 이동 평균/표준편차
    data["rolling_mean_3"] = by_seg.transform(lambda s: s.rolling(3).mean())
    data["rolling_mean_12"] = by_seg.transform(lambda s: s.rolling(12).mean())
    data["rolling_std_6"] = by_seg.transform(lambda s: s.rolling(6).std())

    # 타깃: 5분 뒤 값, 그리고 '지금 대비 변화량'
    data["target_next"] = by_seg.shift(-1)
    data["target_delta"] = data["target_next"] - data["lag_0"]

    return data


def choose_feature_columns(data, extra_cols):
    """데이터 기간/내용에 맞게 실제 사용할 피처를 고른다."""
    span_days = (
        data["timestamp"].iloc[-1] - data["timestamp"].iloc[0]
    ).total_seconds() / 86400

    columns = ["hour_sin", "hour_cos"]
    if span_days >= 14:       # 요일 패턴은 최소 2주는 있어야 학습 가능
        columns += ["weekday_sin", "weekday_cos"]
    if span_days >= 365:      # 계절 패턴은 1년 이상 있어야 학습 가능
        columns += ["month_sin", "month_cos"]

    columns += [f"lag_{lag}" for lag in [0] + LAGS]
    columns += ["diff_1", "rolling_mean_3", "rolling_mean_12", "rolling_std_6"]

    dropped_constant, dropped_duplicate = [], []
    kept_extra = []
    for column in extra_cols:
        values = data[column].to_numpy(dtype=float)
        if np.nanstd(values) == 0:
            dropped_constant.append(column)
            continue
        if any(np.allclose(values, data[k].to_numpy(dtype=float), equal_nan=True) for k in kept_extra):
            dropped_duplicate.append(column)
            continue
        kept_extra.append(column)

    columns += kept_extra
    info = {
        "span_days": span_days,
        "dropped_constant": dropped_constant,
        "dropped_duplicate": dropped_duplicate,
    }
    return columns, info


def build_latest_features(csv_path, feature_columns):
    """CSV의 가장 최근 시점으로 예측 입력 한 줄을 만든다. (app.py가 화면을 열 때마다 사용)

    반환: (입력 DataFrame 1행, 마지막 관측 시각) / 만들 수 없으면 (None, None)
    """
    df = load_and_prepare_data(csv_path, min_rows=1)
    data = make_features(df)

    if any(column not in data.columns for column in feature_columns):
        return None, None

    row = data[list(feature_columns)].iloc[[-1]].copy()
    return row, data["timestamp"].iloc[-1]


# ---------------------------------------------------------
# 3. 평가 지표
# ---------------------------------------------------------
def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # 실제값이 0인 행은 MAPE의 분모가 0이 되어 수조 %로 폭발한다.
    # (BTS처럼 새벽에 5분간 항공편이 0편인 구간이 있는 데이터에서 발생)
    # 그래서 실제값 > 0 인 행만으로 MAPE를 계산하고, 0이 섞여도 안정적인 WAPE를 함께 저장한다.
    nonzero = y_true > 0
    if nonzero.any():
        mape = mean_absolute_percentage_error(y_true[nonzero], y_pred[nonzero]) * 100
    else:
        mape = float("nan")

    total = np.abs(y_true).sum()
    wape = np.abs(y_true - y_pred).sum() / total * 100 if total > 0 else float("nan")

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE (%)": mape,
        "WAPE (%)": wape,
        "R2": r2_score(y_true, y_pred),
    }


def mean_of_folds(fold_metrics):
    fold_df = pd.DataFrame(fold_metrics)
    return {
        "MAE": fold_df["MAE"].mean(),
        "MAE_std": fold_df["MAE"].std(ddof=0),
        "RMSE": fold_df["RMSE"].mean(),
        "MAPE (%)": fold_df["MAPE (%)"].mean(),
        "R2": fold_df["R2"].mean(),
    }


# ---------------------------------------------------------
# 4. 학습
# ---------------------------------------------------------
def train_dataset(dataset_key, config, tune=True):
    label = config["label"]
    save_dir = SAVE_ROOT / dataset_key
    save_dir.mkdir(parents=True, exist_ok=True)

    if tune and not OPTUNA_AVAILABLE:
        print("   [주의] optuna가 설치되어 있지 않아 튜닝을 건너뜁니다. (pip install optuna)")
        tune = False
    saved_params = {} if tune else load_saved_params(save_dir)
    all_best_params = {}

    print(f"\n{'=' * 60}")
    print(f"[{label}] 학습 시작: {config['path']}")
    print("=" * 60)

    dataset_start = time.time()
    t0 = time.time()
    step("데이터 불러오는 중...")
    df = load_and_prepare_data(config["path"])
    extra_cols = df.attrs.get("extra_cols", [])
    step("데이터 불러오기 완료", t0)

    t0 = time.time()
    step("피처 생성 중...")
    data = make_features(df)
    feature_columns, info = choose_feature_columns(data, extra_cols)
    step("피처 생성 완료", t0)
    weather_features = [c for c in feature_columns if c in WEATHER_COLUMNS]
    non_weather_features = [c for c in feature_columns if c not in WEATHER_COLUMNS]

    latest_features = data[feature_columns].iloc[[-1]].copy()

    train_data = data.dropna(subset=feature_columns + ["target_delta"]).copy()
    n_samples = len(train_data)

    if n_samples < MIN_SAMPLES:
        raise ValueError(
            f"피처 생성 후 학습 샘플이 {n_samples}개뿐입니다. (최소 {MIN_SAMPLES}개)\n"
            "연속으로 수집된 데이터가 더 쌓이면 다시 실행해주세요."
        )

    X = train_data[feature_columns]
    y_delta = train_data["target_delta"]     # 모델이 맞추는 값: 5분 뒤 - 지금
    y_level = train_data["target_next"]      # 평가에 쓰는 실제 값: 5분 뒤 항공기 수
    base = train_data["lag_0"]               # 지금 값 (변화량을 실제 값으로 되돌릴 때 사용)
    target_time = train_data["timestamp"] + pd.Timedelta(minutes=INTERVAL_MINUTES)

    print(f"   관측값: {len(df):,} (연속 구간 segment {df['segment'].nunique()}개)")
    print(f"   학습 샘플: {n_samples:,} | 사용 피처: {len(feature_columns)}개 | 데이터 기간: {info['span_days']:.1f}일")
    if info["dropped_constant"]:
        print(f"   [제외] 값이 항상 같은 피처: {', '.join(info['dropped_constant'])}")
    if info["dropped_duplicate"]:
        print(f"   [제외] 다른 피처와 값이 같은 피처: {', '.join(info['dropped_duplicate'])}")
    if n_samples < RECOMMENDED_SAMPLES:
        print(
            f"   [주의] 샘플이 {RECOMMENDED_SAMPLES:,}개(약 1주일치)보다 적어 "
            "성능 수치가 불안정할 수 있습니다. 수집을 더 이어가 주세요."
        )

    # ----- 시간 순서대로 dev / test 분리 -----
    test_size = max(1, int(n_samples * TEST_RATIO))
    split = n_samples - test_size

    X_dev, X_test = X.iloc[:split], X.iloc[split:]
    yd_dev = y_delta.iloc[:split]
    yl_dev, yl_test = y_level.iloc[:split], y_level.iloc[split:]
    base_dev, base_test = base.iloc[:split], base.iloc[split:]
    test_times = target_time.iloc[split:]

    n_splits = int(np.clip(len(X_dev) // MIN_FOLD_SIZE - 1, 2, MAX_CV_SPLITS))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    print(f"   교차검증 fold: {n_splits}개 | dev {len(X_dev):,} / test {len(X_test):,}")

    cv_summary_rows = []
    cv_fold_rows = []
    test_rows = []
    saved_models = {}

    # ----- 기준선: '5분 뒤에도 지금과 같다' -----
    baseline_folds = []
    for fold_number, (_, val_idx) in enumerate(tscv.split(X_dev), start=1):
        metrics = calculate_metrics(yl_dev.iloc[val_idx], base_dev.iloc[val_idx])
        baseline_folds.append(metrics)
        cv_fold_rows.append({"Model": BASELINE_NAME, "Fold": fold_number, **{
            k: metrics[k] for k in ["MAE", "RMSE", "MAPE (%)", "R2"]}})

    baseline_cv = mean_of_folds(baseline_folds)
    baseline_test = calculate_metrics(yl_test, base_test)

    cv_summary_rows.append({
        "Model": BASELINE_NAME,
        "CV MAE": baseline_cv["MAE"],
        "CV MAE Std": baseline_cv["MAE_std"],
        "CV RMSE": baseline_cv["RMSE"],
        "CV MAPE (%)": baseline_cv["MAPE (%)"],
        "CV R2": baseline_cv["R2"],
    })
    test_rows.append({
        "Model": BASELINE_NAME,
        "Test MAE": baseline_test["MAE"],
        "Test RMSE": baseline_test["RMSE"],
        "Test MAPE (%)": baseline_test["MAPE (%)"],
        "Test R2": baseline_test["R2"],
    })
    print(f"\n   기준선(직전값) | CV MAE: {baseline_cv['MAE']:.2f} | Test MAE: {baseline_test['MAE']:.2f}")

    trained_at = datetime.now(timezone.utc).isoformat()

    total_models = len(MODEL_CONFIGS)
    for model_index, (model_name, model_config) in enumerate(MODEL_CONFIGS.items(), start=1):
        print(f"\n   모델 [{model_index}/{total_models}]: {model_name}")
        model_start = time.time()
        fold_metrics = []

        # ----- Optuna 튜닝 (dev 구간만 사용) 또는 저장된 최적값 불러오기 -----
        if tune and model_name in DEFAULT_START:
            model_params = tune_model(
                model_name, X_dev, yd_dev, n_splits, save_dir, model_config["slug"]
            )
        else:
            model_params = saved_params.get(model_name, {})
            if model_name in DEFAULT_START:
                note = "저장된 최적 파라미터 사용" if model_params else "기본 파라미터 사용"
                step(f"{model_name} | 튜닝 없음 ({note})")
        estimator = make_model(model_name, model_params)
        all_best_params[model_name] = model_params

        for fold_number, (train_idx, val_idx) in enumerate(tscv.split(X_dev), start=1):
            fold_start = time.time()
            step(f"{model_name} | 교차검증 fold {fold_number}/{n_splits} 학습 중...")
            fold_model = clone(estimator)
            fold_model.fit(X_dev.iloc[train_idx], yd_dev.iloc[train_idx])

            val_pred = base_dev.iloc[val_idx].to_numpy() + fold_model.predict(X_dev.iloc[val_idx])
            val_pred = np.maximum(val_pred, 0)
            metrics = calculate_metrics(yl_dev.iloc[val_idx], val_pred)
            fold_metrics.append(metrics)

            cv_fold_rows.append({
                "Model": model_name,
                "Fold": fold_number,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE (%)": metrics["MAPE (%)"],
                "R2": metrics["R2"],
            })
            step(f"{model_name} | fold {fold_number}/{n_splits} 완료 (MAE {metrics['MAE']:.2f})", fold_start)

        cv = mean_of_folds(fold_metrics)
        cv_summary_rows.append({
            "Model": model_name,
            "CV MAE": cv["MAE"],
            "CV MAE Std": cv["MAE_std"],
            "CV RMSE": cv["RMSE"],
            "CV MAPE (%)": cv["MAPE (%)"],
            "CV R2": cv["R2"],
        })

        # dev 전체로 학습 -> 테스트 평가
        step(f"{model_name} | 테스트 평가용 모델 학습 중...")
        dev_model = clone(estimator)
        dev_model.fit(X_dev, yd_dev)
        test_pred = np.maximum(base_test.to_numpy() + dev_model.predict(X_test), 0)
        test_metrics = calculate_metrics(yl_test, test_pred)

        test_row = {
            "Model": model_name,
            "Test MAE": test_metrics["MAE"],
            "Test RMSE": test_metrics["RMSE"],
            "Test MAPE (%)": test_metrics["MAPE (%)"],
            "Test R2": test_metrics["R2"],
        }

        # 날씨를 뺐을 때와 비교 (날씨 피처가 실제로 도움이 되는지 확인용)
        if weather_features:
            step(f"{model_name} | 날씨 제외 비교 모델 학습 중...")
            nw_model = clone(estimator)
            nw_model.fit(X_dev[non_weather_features], yd_dev)
            nw_pred = np.maximum(base_test.to_numpy() + nw_model.predict(X_test[non_weather_features]), 0)
            test_row["Test MAE (날씨 제외)"] = mean_absolute_error(yl_test, nw_pred)

        test_rows.append(test_row)

        # 실제 서비스(app.py)에서 쓰는 모델은 가장 최근 데이터까지 전부 넣고 다시 학습
        step(f"{model_name} | 최종 모델(전체 데이터) 학습 중...")
        final_model = clone(estimator)
        final_model.fit(X, y_delta)

        importance = None
        if hasattr(final_model, "feature_importances_"):
            importance = dict(
                zip(feature_columns, map(float, final_model.feature_importances_))
            )

        bundle = {
            "model": final_model,
            "model_name": model_name,
            "dataset_key": dataset_key,
            "dataset_label": label,
            "feature_columns": feature_columns,
            "weather_columns": weather_features,
            "feature_importance": importance,
            "latest_features": latest_features,
            "interval_minutes": INTERVAL_MINUTES,
            # 모델 출력은 '지금 대비 변화량'이므로, 예측값 = lag_0 + model.predict(...)
            "target_mode": "delta",
            "base_column": "lag_0",
            "refit_on_all": True,
            "tuned": bool(tune and model_params),
            "best_params": model_params,
            "n_samples": n_samples,
            "trained_at": trained_at,
            "first_data_timestamp": df["timestamp"].iloc[0],
            "last_data_timestamp": df["timestamp"].iloc[-1],
            "cv_metrics": {
                "MAE": cv["MAE"],
                "MAE_std": cv["MAE_std"],
                "RMSE": cv["RMSE"],
                "MAPE (%)": cv["MAPE (%)"],
                "R2": cv["R2"],
            },
            "test_metrics": test_metrics,
            "baseline_test_metrics": baseline_test,
            "test_predictions": pd.DataFrame({
                "timestamp": test_times.values,
                "actual": yl_test.values,
                "predicted": np.asarray(test_pred),
                "baseline": base_test.values,
            }),
        }

        joblib.dump(bundle, save_dir / f"model_{model_config['slug']}.pkl")
        saved_models[model_name] = bundle

        step(f"{model_name} 완료 | CV MAE: {cv['MAE']:.2f} | Test MAE: {test_metrics['MAE']:.2f}", model_start)

    if tune:
        (save_dir / BEST_PARAMS_FILE).write_text(
            json.dumps(all_best_params, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            "\n   [참고] 튜닝에 쓴 dev 구간으로 CV를 다시 계산하므로 CV MAE는 다소 낙관적입니다. "
            "모델 비교는 한 번도 안 본 Test MAE 기준으로 해주세요."
        )

    cv_summary = pd.DataFrame(cv_summary_rows).sort_values("CV MAE")
    cv_folds = pd.DataFrame(cv_fold_rows)
    test_results = pd.DataFrame(test_rows)

    comparison = cv_summary.merge(test_results, on="Model")
    comparison["기준선 대비 Test MAE 개선 (%)"] = (
        (baseline_test["MAE"] - comparison["Test MAE"]) / baseline_test["MAE"] * 100
    ).round(2)
    comparison = comparison.sort_values("CV MAE")

    cv_summary.to_csv(save_dir / "cross_validation_results.csv", index=False, encoding="utf-8-sig")
    cv_folds.to_csv(save_dir / "cross_validation_folds.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(save_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")

    # 최고 모델은 기준선을 제외하고 선택
    model_only = cv_summary[cv_summary["Model"] != BASELINE_NAME]
    best_model_name = model_only.iloc[0]["Model"]
    joblib.dump(saved_models[best_model_name], save_dir / "demand_model.pkl")

    best_cv_mae = float(model_only.iloc[0]["CV MAE"])
    if best_cv_mae >= baseline_cv["MAE"]:
        print(
            f"\n   [주의] 어떤 모델도 기준선(직전값)보다 CV MAE가 낮지 않습니다. "
            "데이터가 더 쌓이기 전에는 모델의 이점이 거의 없다는 뜻입니다."
        )
    print(f"\n   [{label}] 최고 모델 저장 완료: {best_model_name}")
    step(f"[{label}] 전체 학습 종료", dataset_start)

    return {
        "best_model": best_model_name,
        "n_samples": n_samples,
        "best_cv_mae": best_cv_mae,
        "baseline_cv_mae": float(baseline_cv["MAE"]),
        "trained_at": trained_at,
    }


def main(targets=None, tune=True):
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    for dataset_key, config in DATASETS.items():
        if targets and dataset_key not in targets:
            continue
        try:
            train_dataset(dataset_key, config, tune=tune)
        except Exception as error:
            print(f"\n[{config['label']}] 건너뜀: {error}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(
        targets=[a for a in args if not a.startswith("--")] or None,
        tune="--no-tune" not in args,
    )