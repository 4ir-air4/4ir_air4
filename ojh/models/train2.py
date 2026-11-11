"""공항 월간 이용객 수요 예측 모델 학습 (미국 · 유럽 · 한국 공항).

사용법
    python ojh/models/train2.py              # Optuna 튜닝 포함 전체 학습
    python ojh/models/train2.py --no-tune    # 튜닝 없이 (직전 최적 파라미터가 있으면 재사용)

입력   data/airport_demand/airport_demand_passengers_monthly.csv, airport_demand_gdp_growth.csv   (fetch_airports.py가 생성)
출력   models/saved/airport/
         model_<slug>.pkl            모델별 번들 (Linear Regression, Random Forest, XGBoost, LightGBM)
         demand_model.pkl            Validation 12개월 재귀 WAPE가 가장 낮은 모델
         airport_context.pkl         공항 이력 / 메타 / GDP 표 (앱에서 시뮬레이션에 사용)
         model_comparison.csv        모델 비교표
         airport_test_mape.csv       공항별 테스트 MAPE

설계
    - 타깃: log(이번 달 승객) - log(직전 12개월 평균)   -> 공항 규모와 무관한 '변화율'을 학습
    - 분할: 시점 기준  Train | Validation(24개월) | Test(최근 12개월)
    - 평가: 한 달 앞 예측(실제 lag 사용) + 12개월 재귀 예측(앱의 연간 곡선과 같은 방식)
    - 기준선: '작년 같은 달' 값
"""

import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:  # pragma: no cover
    optuna = None

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------
# 설정
# ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "airport_demand"   # 공항 이용객 탭 전용 폴더 (fetch_airports.py 가 만듦)
LEGACY_RAW_DIR = ROOT_DIR / "data" / "raw"         # 예전 위치 (자동 인식용)
SAVE_DIR = ROOT_DIR / "models" / "saved" / "airport"

TEST_MONTHS = 12            # 맨 뒤 12개월 = 최종 테스트
VAL_MONTHS = 24             # 그 앞 24개월 = 검증(튜닝/모델 선택)
MIN_HISTORY_MONTHS = 36     # 이력이 이보다 짧은 공항은 제외 (36 = 이스탄불처럼 최근에 생긴 대형 공항도 포함)
MIN_AVG_PASSENGERS = 20_000 # 월 평균 승객이 이보다 적은 공항은 제외
MAX_AIRPORTS = None         # 규모 상위 N개 공항만 사용. None = 조건(이력·월평균 승객)을 통과한 공항 전부 사용
MAX_GAP_RATIO = 0.6         # 마지막 달들의 공항 수가 평소의 60%보다 적으면 '미완성 달'로 보고 제외
STALE_MONTHS = 12           # 가장 최신 데이터보다 이만큼 넘게 끊긴 공항(폐쇄 등)은 예측 화면에서 뺀다
REGION_SOURCES = {"US": "BTS T-100", "Europe": "Eurostat · 영국 CAA", "Korea": "에어포탈"}

VAL_ORIGINS = [0, 6, 12]   # Validation 12개월 재귀 평가 시작점(검증 시작 후 N개월). 코로나 반등 한 구간에 치우치지 않도록 여러 번 평가
LAGS = [1, 2, 3, 6, 12]
COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2021-12-01")
RECOVERY_END = pd.Timestamp("2022-12-01")
REGIONS = ["US", "Europe", "Korea"]

TUNE_N_TRIALS = 20
TUNE_TIMEOUT_SEC = 120
TUNE_SEED = 42
BEST_PARAMS_FILE = "optuna_best_params.json"
BASELINE_NAME = "Baseline (작년 같은 달)"
# 모델 선정 지표: WAPE = 전체 절대오차 합 / 전체 실제값 합.
# MAPE(공항·월별 오차율의 평균)는 겨울에 승객이 수십 명뿐인 휴양지 공항 몇 곳이 수천 %를 만들어 평균을 망가뜨린다.
# WAPE는 '전체 승객 중 몇 %를 틀렸나'라서 이런 공항에 휘둘리지 않는다.
SELECTION_METRIC = "Val 12개월 재귀 WAPE (%)"

FEATURE_COLUMNS = (
    [f"lag{k}" for k in LAGS]
    + ["roll3", "roll6", "std6", "yoy", "log_scale", "month_sin", "month_cos", "covid", "covid_recovery", "gdp_growth"]
    + [f"region_{r}" for r in REGIONS]
)


def step(message, since=None):
    tail = f" ({time.time() - since:.1f}초)" if since else ""
    print(f"[{datetime.now():%H:%M:%S}] {message}{tail}", flush=True)


# ---------------------------------------------------------
# 1. 데이터 불러오기 / 정리
# ---------------------------------------------------------
def load_inputs(raw_dir=RAW_DIR):
    passengers_path = Path(raw_dir) / "airport_demand_passengers_monthly.csv"
    gdp_path = Path(raw_dir) / "airport_demand_gdp_growth.csv"
    if not passengers_path.exists() and (LEGACY_RAW_DIR / "airport_passengers_monthly.csv").exists():  # 예전 파일 이름도 인식
        passengers_path = LEGACY_RAW_DIR / "airport_passengers_monthly.csv"
        gdp_path = LEGACY_RAW_DIR / "gdp_growth.csv"
    if not passengers_path.exists():
        raise FileNotFoundError(f"{passengers_path} 이 없습니다. 먼저 `python ojh/collect/fetch_airports.py` 를 실행해주세요.")
    data = pd.read_csv(passengers_path, parse_dates=["date"])
    data["passengers"] = pd.to_numeric(data["passengers"], errors="coerce")
    data["date"] = data["date"].dt.to_period("M").dt.to_timestamp()
    data = data.dropna(subset=["passengers", "date"])
    if gdp_path.exists():
        gdp = pd.read_csv(gdp_path)
    else:
        step("[주의] GDP 파일이 없어 GDP 피처를 평균값으로 대체합니다.")
        gdp = pd.DataFrame(columns=["country", "year", "gdp_growth"])
    return data, gdp


def airport_funnel(data):
    """분석 공항이 몇 개에서 몇 개로 걸러지는지 단계별 개수 (앱 개요 화면에 표시)"""
    counts = data.groupby("date")["airport"].nunique().sort_index()
    ref = counts.tail(36).median()
    dates = list(counts.index)
    while dates and counts[dates[-1]] < MAX_GAP_RATIO * ref:
        dates.pop()
    stats = data[data["date"] <= dates[-1]].groupby("airport")["passengers"].agg(["count", "mean"])
    history_ok = stats[stats["count"] >= MIN_HISTORY_MONTHS]
    eligible = history_ok[history_ok["mean"] >= MIN_AVG_PASSENGERS]
    return {
        "total": int(data["airport"].nunique()),
        "history_ok": int(len(history_ok)),
        "eligible": int(len(eligible)),
        "used": int(len(eligible) if MAX_AIRPORTS is None else min(len(eligible), MAX_AIRPORTS)),
        "min_history_months": MIN_HISTORY_MONTHS,
        "min_avg_passengers": MIN_AVG_PASSENGERS,
        "max_airports": MAX_AIRPORTS,
    }


def prepare_panel(data):
    """긴 형태 데이터 -> (공항 x 월) 와이드 표 + 공항 메타 + 자르지 않은 전체 와이드 표

    wide      : 학습/평가용. 공항 수가 갑자기 줄어드는 마지막 '미완성 달'을 잘라 모든 공항의 시점을 맞춘다.
    wide_full : 예측용. 자르지 않아서 공항마다 자기 마지막 실측 달까지 들어 있다 (예: 한국은 미국보다 3달 더 최신).
    """
    data_all = data
    counts = data.groupby("date")["airport"].nunique().sort_index()
    ref = counts.tail(36).median()
    dates = list(counts.index)
    while dates and counts[dates[-1]] < MAX_GAP_RATIO * ref:  # 아직 덜 집계된 마지막 달 제거
        dates.pop()
    data = data[data["date"] <= dates[-1]]

    stats = data.groupby("airport")["passengers"].agg(["count", "mean"])
    keep = stats[(stats["count"] >= MIN_HISTORY_MONTHS) & (stats["mean"] >= MIN_AVG_PASSENGERS)]
    recent = data[data["date"] > dates[-1] - pd.DateOffset(months=24)].groupby("airport")["passengers"].mean()
    ranked = recent.reindex(keep.index).fillna(keep["mean"]).sort_values(ascending=False)
    airports = list(ranked.index[:MAX_AIRPORTS])
    data = data[data["airport"].isin(airports)]

    full_range = pd.date_range(data["date"].min(), data["date"].max(), freq="MS")
    wide = data.pivot_table(index="airport", columns="date", values="passengers", aggfunc="last").reindex(columns=full_range)
    wide = wide.loc[[a for a in airports if a in wide.index]]

    meta = (
        data.sort_values("date").groupby("airport")[["airport_name", "country", "region"]].last().reindex(wide.index)
    )
    full = data_all[data_all["airport"].isin(wide.index)]
    wide_full = full.pivot_table(index="airport", columns="date", values="passengers", aggfunc="last")
    wide_full = wide_full.reindex(index=wide.index, columns=pd.date_range(wide.columns[0], full["date"].max(), freq="MS"))
    return wide, meta, wide_full


def build_gdp_table(gdp, countries, years):
    """국가 x 연도 GDP 성장률 표. 빈 칸은 같은 나라의 가까운 연도 값, 그래도 없으면 전체 평균으로 채운다."""
    years = sorted(set(int(y) for y in years))
    if gdp is None or gdp.empty:
        return pd.DataFrame(2.0, index=list(countries), columns=years)
    piv = gdp.pivot_table(index="country", columns="year", values="gdp_growth", aggfunc="last")
    all_years = sorted(set(piv.columns) | set(years))
    table = piv.reindex(index=list(countries), columns=all_years).ffill(axis=1).bfill(axis=1)
    mean_value = float(np.nanmean(piv.values)) if piv.size else 2.0
    return table.reindex(columns=years).fillna(mean_value)


# ---------------------------------------------------------
# 2. 피처 생성 (학습과 앱 예측이 똑같이 사용)
# ---------------------------------------------------------
def make_features(H, month_date, regions, gdp_vec):
    """H: (공항 x 지난 달들) 승객 수 배열. month_date: 예측할 달. 반환: (피처 DataFrame, 직전 12개월 평균)"""
    n, t = H.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        last12 = H[:, t - 12:t]
        count12 = np.sum(~np.isnan(last12), axis=1)
        base = np.where(count12 >= 9, np.nanmean(last12, axis=1), np.nan)
        lb = np.log1p(base)

        cols = {}
        for k in LAGS:
            cols[f"lag{k}"] = np.log1p(H[:, t - k]) - lb
        cols["roll3"] = np.log1p(np.nanmean(H[:, t - 3:t], axis=1)) - lb
        cols["roll6"] = np.log1p(np.nanmean(H[:, t - 6:t], axis=1)) - lb
        cols["std6"] = np.nanstd(np.log1p(H[:, t - 6:t]), axis=1)
        cols["yoy"] = np.nan_to_num(np.log1p(H[:, t - 1]) - np.log1p(H[:, t - 13]), nan=0.0)
    cols["log_scale"] = lb

    month = month_date.month
    cols["month_sin"] = np.full(n, np.sin(2 * np.pi * month / 12))
    cols["month_cos"] = np.full(n, np.cos(2 * np.pi * month / 12))
    cols["covid"] = np.full(n, float(COVID_START <= month_date <= COVID_END))
    cols["covid_recovery"] = np.full(n, float(COVID_END < month_date <= RECOVERY_END))
    cols["gdp_growth"] = np.asarray(gdp_vec, dtype=float)
    regions = np.asarray(regions)
    for r in REGIONS:
        cols[f"region_{r}"] = (regions == r).astype(float)
    return pd.DataFrame(cols)[FEATURE_COLUMNS], base


def build_training_table(wide, meta, gdp_table):
    H = wide.to_numpy(dtype=float)
    dates = list(wide.columns)
    regions = meta["region"].to_numpy()
    countries = meta["country"].to_numpy()
    airports = np.array(wide.index)
    frames = []
    for t in range(13, len(dates)):
        gdp_vec = gdp_table.reindex(index=countries, columns=[dates[t].year]).iloc[:, 0].to_numpy()
        X, base = make_features(H[:, :t], dates[t], regions, gdp_vec)
        y = H[:, t]
        ok = np.isfinite(y) & np.isfinite(base) & np.isfinite(X.to_numpy()).all(axis=1)
        if not ok.any():
            continue
        part = X[ok].copy()
        part["airport"] = airports[ok]
        part["date"] = dates[t]
        part["y"] = y[ok]
        part["base"] = base[ok]
        part["baseline_pred"] = H[ok, t - 12]
        part["target"] = np.log1p(y[ok]) - np.log1p(base[ok])
        frames.append(part)
    if not frames:
        raise ValueError("학습 가능한 행이 없습니다. 데이터 기간(최소 4년)과 공항 수를 확인해주세요.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------
# 3. 평가 지표
# ---------------------------------------------------------
def level_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE (%)": np.nan, "WAPE (%)": np.nan, "R2": np.nan, "n": 0}
    nz = y_true > 0
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE (%)": float(np.mean(np.abs(y_true[nz] - y_pred[nz]) / y_true[nz]) * 100) if nz.any() else np.nan,
        "WAPE (%)": float(np.abs(y_true - y_pred).sum() / y_true.sum() * 100) if y_true.sum() > 0 else np.nan,
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "n": int(len(y_true)),
    }


def to_level(base, ratio_pred):
    return np.clip(np.expm1(np.log1p(base) + ratio_pred), 0, None)


# ---------------------------------------------------------
# 4. 모델 정의 / Optuna 튜닝
# ---------------------------------------------------------
MODEL_SLUGS = {
    "Linear Regression": "linear_regression",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
}


def available_models():
    names = ["Linear Regression", "Random Forest"]
    if XGBRegressor is not None:
        names.append("XGBoost")
    else:
        step("[건너뜀] xgboost 가 설치되어 있지 않습니다 (pip install xgboost)")
    if LGBMRegressor is not None:
        names.append("LightGBM")
    else:
        step("[건너뜀] lightgbm 이 설치되어 있지 않습니다 (pip install lightgbm)")
    return names


def make_model(name, params=None):
    params = dict(params or {})
    if name == "Linear Regression":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "Random Forest":
        base = {"n_estimators": 200, "max_depth": 14, "min_samples_leaf": 3, "max_features": 0.7, "n_jobs": -1, "random_state": 42}
        return RandomForestRegressor(**{**base, **params})
    if name == "XGBoost":
        base = {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": -1, "random_state": 42}
        return XGBRegressor(**{**base, **params})
    if name == "LightGBM":
        base = {"n_estimators": 400, "num_leaves": 63, "learning_rate": 0.05, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "n_jobs": -1, "random_state": 42, "verbose": -1}
        return LGBMRegressor(**{**base, **params})
    raise ValueError(name)


def suggest_params(name, trial):
    if name == "Random Forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 6, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        }
    if name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 700, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 700, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    return {}


def tune_model(name, X_train, y_train, X_val, y_val):
    """Train으로 학습 -> Validation 로그변화율 MAE가 가장 낮은 파라미터를 찾는다. (Test는 절대 보지 않음)"""
    if optuna is None or name == "Linear Regression":
        return {}
    step(f"{name} | Optuna 튜닝 시작 (최대 {TUNE_N_TRIALS}회 / {TUNE_TIMEOUT_SEC}초)")

    def objective(trial):
        model = make_model(name, suggest_params(name, trial))
        model.fit(X_train, y_train)
        return float(mean_absolute_error(y_val, model.predict(X_val)))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=TUNE_SEED), pruner=MedianPruner())
    study.optimize(objective, n_trials=TUNE_N_TRIALS, timeout=TUNE_TIMEOUT_SEC)
    try:
        best = suggest_params(name, optuna.trial.FixedTrial(study.best_params))
        step(f"{name} | 튜닝 완료 (Validation 로그오차 {study.best_value:.4f}) -> {best}")
        return best
    except ValueError:
        return {}


# ---------------------------------------------------------
# 5. 재귀 예측 (앱의 12개월 곡선과 백테스트가 같은 함수를 사용)
# ---------------------------------------------------------
def recursive_forecast(model, feature_columns, history, regions, countries, gdp_table, n_months=12, gdp_override=None):
    """history: (공항 x 월) 승객 표. 마지막 달 다음부터 n_months개월을 한 달씩 예측해서 (공항 x n_months) 표를 돌려준다.

    gdp_override: 모든 예측 달에 쓸 GDP 성장률(%)
    """
    H = history.to_numpy(dtype=float).copy()
    last = history.columns[-1]
    countries = np.asarray(countries)
    out_dates, out_cols = [], []
    for i in range(1, n_months + 1):
        date = last + pd.DateOffset(months=i)
        if gdp_override is None:
            gdp_vec = gdp_table.reindex(index=countries, columns=[date.year]).iloc[:, 0].to_numpy()
            gdp_vec = np.where(np.isfinite(gdp_vec), gdp_vec, np.nanmean(gdp_table.to_numpy()))
        else:
            gdp_vec = np.full(len(countries), float(gdp_override))
        X, base = make_features(H, date, regions, gdp_vec)
        X = X.fillna(0.0)
        pred = to_level(base, model.predict(X[feature_columns]))
        H = np.concatenate([H, pred.reshape(-1, 1)], axis=1)
        out_dates.append(date)
        out_cols.append(pred)
    return pd.DataFrame(np.column_stack(out_cols), index=history.index, columns=out_dates)


def rolling_recursive_eval(model, wide, val_start, regions, countries, gdp_table, baseline=False):
    """Validation 구간 안에서 시작점을 바꿔 가며 12개월 재귀 예측을 여러 번 평가하고 합친 지표를 돌려준다.
    baseline=True 이면 모델 대신 '작년 같은 달' 값으로 같은 구간을 평가한다."""
    actual, pred = [], []
    for offset in VAL_ORIGINS:
        origin = val_start + pd.DateOffset(months=offset)
        cols = [origin + pd.DateOffset(months=i) for i in range(12)]
        if cols[-1] > wide.columns[-1] or offset + 12 > VAL_MONTHS:
            continue
        act = wide.reindex(columns=cols).to_numpy().ravel()
        if baseline:
            guess = wide.reindex(columns=[c - pd.DateOffset(years=1) for c in cols]).to_numpy().ravel()
        else:
            history = wide.loc[:, wide.columns < origin]
            guess = recursive_forecast(model, FEATURE_COLUMNS, history, regions, countries, gdp_table, n_months=12).reindex(columns=cols).to_numpy().ravel()
        actual.append(act)
        pred.append(guess)
    if not actual:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE (%)": np.nan, "WAPE (%)": np.nan, "R2": np.nan, "n": 0}
    return level_metrics(np.concatenate(actual), np.concatenate(pred))


def full_history(context):
    """예측용 이력: 공항마다 자기 마지막 실측 달까지 들어 있는 표 (예전 .pkl 이면 학습용 표로 대신)"""
    return context.get("history_full", context["history"])


def last_observed(context, airport):
    return full_history(context).loc[airport].last_valid_index()


def forecast_for_airport(bundle, context, airport, n_months=12, gdp_growth=None):
    """앱 시뮬레이터용: 공항 하나의 향후 n_months개월 예측 (pd.Series, 인덱스=월)

    그 공항의 마지막 실측 달 바로 다음부터 예측한다.
    (예: 한국은 2026-10까지 있으니 2026-11부터, 데이터가 늦게 오는 파리는 2026-04까지 있으니 2026-05부터)
    """
    full = full_history(context)
    history = full.loc[[airport], full.columns <= last_observed(context, airport)]
    meta = context["meta"].loc[[airport]]
    out = recursive_forecast(
        bundle["model"], bundle["feature_columns"], history, meta["region"].to_numpy(), meta["country"].to_numpy(),
        context["gdp_table"], n_months=n_months, gdp_override=gdp_growth,
    )
    return out.iloc[0]


def active_airports(context, max_stale_months=STALE_MONTHS):
    """예측 화면에 보여줄 공항: 가장 최신 데이터보다 max_stale_months 넘게 끊기지 않은 공항.
    (베를린 테겔처럼 폐쇄된 공항은 빼고, 통계가 몇 달 늦게 올라오는 나라의 공항은 남긴다)"""
    last_obs = pd.to_datetime(full_history(context).apply(lambda r: r.last_valid_index(), axis=1).dropna())
    return list(last_obs.index[last_obs >= last_obs.max() - pd.DateOffset(months=max_stale_months)])


def region_summary(context, table=None):
    """지역별 공항 수 · 데이터 기간 · 학습 행 수 (앱 개요 화면의 설명에 사용)"""
    meta, wide, full = context["meta"], context["history"], full_history(context)
    if table is None:
        table = build_training_table(wide, meta, context["gdp_table"])
    rows_by_region = meta.loc[table["airport"], "region"].value_counts()
    first = pd.to_datetime(full.apply(lambda r: r.first_valid_index(), axis=1))
    last = pd.to_datetime(full.apply(lambda r: r.last_valid_index(), axis=1))
    active = set(active_airports(context))
    out = {}
    for region in [r for r in REGIONS if (meta["region"] == r).any()]:
        codes = meta.index[meta["region"] == region]
        out[region] = {
            "source": REGION_SOURCES.get(region, ""),
            "airports": int(len(codes)),
            "closed": int(sum(c not in active for c in codes)),
            "first": first[codes].min(),
            "last": last[codes].max(),
            "obs": int(wide.loc[codes].notna().sum().sum()),     # 학습 기간 안의 실제 공항-월 개수
            "rows": int(rows_by_region.get(region, 0)),           # 그중 학습에 쓰인 행
        }
    return out


def forecast_all_filled(bundle, context, n_months=12, max_stale_months=STALE_MONTHS):
    """순위 레이스용: 모든 공항의 '실제 + 예측' 월별 표.

    유럽은 나라마다 Eurostat 보고 시점이 달라 마지막 실측 달이 공항마다 다르다.
    그래서 공항별 마지막 실측 달 바로 다음부터 예측을 이어 붙여, 모든 공항이 (데이터 마지막 달 + n_months)까지 채워지게 한다.
    마지막 실측이 max_stale_months 보다 오래된 공항(예: 데이터가 끊긴 공항)은 뺀다.

    반환: (표: 공항 x 월, 공항별 마지막 실측 달 Series)
    """
    W = full_history(context)
    meta = context["meta"]
    last_obs = pd.to_datetime(W.apply(lambda r: r.last_valid_index(), axis=1).dropna())
    last = last_obs.max()                       # 가장 최신 실측 달 (공항마다 다름)
    end = last + pd.DateOffset(months=n_months)
    last_obs = last_obs[last_obs >= last - pd.DateOffset(months=max_stale_months)]

    out = W.loc[last_obs.index].reindex(columns=pd.date_range(W.columns[0], end, freq="MS"))
    for start, codes in last_obs.groupby(last_obs).groups.items():
        codes = list(codes)
        steps = (end.year - start.year) * 12 + end.month - start.month
        if steps <= 0:
            continue
        pred = recursive_forecast(
            bundle["model"], bundle["feature_columns"], W.loc[codes, W.columns <= start],
            meta.loc[codes, "region"].to_numpy(), meta.loc[codes, "country"].to_numpy(), context["gdp_table"], n_months=steps,
        )
        out.loc[codes, pred.columns] = pred.to_numpy()
    return out, last_obs


def normal_level(context, airport, month, years=6):
    """'평년' 값: 최근 몇 년 중 코로나 기간을 뺀 같은 달의 평균"""
    s = full_history(context).loc[airport].dropna()
    s = s[(s.index.month == month) & (s.index > s.index.max() - pd.DateOffset(years=years))]
    s = s[(s.index < COVID_START) | (s.index > RECOVERY_END)]
    return float(s.mean()) if len(s) else float("nan")


def latest_gdp(context, airport):
    country = context["meta"].loc[airport, "country"]
    row = context["gdp_table"].loc[country]
    year = min(context["last_date"].year, row.index.max())
    return float(row.loc[year])


# ---------------------------------------------------------
# 6. 전체 학습
# ---------------------------------------------------------
def feature_importance(model):
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        values = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        values = np.abs(np.asarray(est.coef_, dtype=float))
    else:
        return {}
    total = values.sum() or 1.0
    return dict(sorted(zip(FEATURE_COLUMNS, (values / total).tolist()), key=lambda kv: -kv[1]))


def comparison_row(name, val_m, rec_val_m, test_m, rec_m):
    return {
        "Model": name,
        "Val MAE": val_m["MAE"], "Val RMSE": val_m["RMSE"], "Val MAPE (%)": val_m["MAPE (%)"], "Val WAPE (%)": val_m["WAPE (%)"], "Val R2": val_m["R2"],
        "Val 12개월 재귀 MAPE (%)": rec_val_m["MAPE (%)"], "Val 12개월 재귀 WAPE (%)": rec_val_m["WAPE (%)"],
        "Test MAE": test_m["MAE"], "Test RMSE": test_m["RMSE"], "Test MAPE (%)": test_m["MAPE (%)"], "Test WAPE (%)": test_m["WAPE (%)"], "Test R2": test_m["R2"],
        "12개월 재귀 MAPE (%)": rec_m.get("MAPE (%)", np.nan), "12개월 재귀 WAPE (%)": rec_m.get("WAPE (%)", np.nan),
    }


def train_airport(tune=True, raw_dir=RAW_DIR, save_dir=SAVE_DIR):
    started = time.time()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    step("1/6 데이터 불러오기 / 공항 정리")
    data, gdp = load_inputs(raw_dir)
    wide, meta, wide_full = prepare_panel(data)
    dates = list(wide.columns)
    need = MIN_HISTORY_MONTHS
    if len(dates) < need + TEST_MONTHS + VAL_MONTHS - 24:
        raise ValueError(f"기간이 너무 짧습니다({len(dates)}개월).")
    gdp_years = list(range(dates[0].year - 1, dates[-1].year + 3))
    gdp_table = build_gdp_table(gdp, sorted(meta["country"].unique()), gdp_years)
    step(f"   공항 {len(wide)}개 | {dates[0]:%Y-%m} ~ {dates[-1]:%Y-%m} ({len(dates)}개월) | 지역별 {meta['region'].value_counts().to_dict()}")

    step("2/6 피처 엔지니어링")
    table = build_training_table(wide, meta, gdp_table)
    test_start = dates[-TEST_MONTHS]
    val_start = dates[-(TEST_MONTHS + VAL_MONTHS)]
    is_train = table["date"] < val_start
    is_val = (table["date"] >= val_start) & (table["date"] < test_start)
    is_test = table["date"] >= test_start
    step(f"   행 {len(table):,}개 | Train {is_train.sum():,} (~{val_start - pd.DateOffset(months=1):%Y-%m}) | Validation {is_val.sum():,} ({val_start:%Y-%m}~) | Test {is_test.sum():,} ({test_start:%Y-%m}~)")
    if is_train.sum() < 200 or is_val.sum() < 20 or is_test.sum() < 20:
        raise ValueError("분할된 구간의 행 수가 너무 적습니다. 데이터를 더 모아주세요.")

    X, y = table[FEATURE_COLUMNS], table["target"]
    saved_params = {}
    params_path = save_dir / BEST_PARAMS_FILE
    if not tune and params_path.exists():
        saved_params = json.loads(params_path.read_text(encoding="utf-8"))

    history_before_test = wide.loc[:, wide.columns < test_start]
    regions, countries = meta["region"].to_numpy(), meta["country"].to_numpy()

    baseline_val = level_metrics(table.loc[is_val, "y"], table.loc[is_val, "baseline_pred"])
    baseline_test = level_metrics(table.loc[is_test, "y"], table.loc[is_test, "baseline_pred"])
    # 재귀 백테스트 기준선: 작년 같은 달 그대로
    actual_test = wide.loc[:, wide.columns >= test_start]
    last_year = wide.loc[:, (wide.columns >= test_start - pd.DateOffset(years=1)) & (wide.columns < test_start)]
    baseline_rec = level_metrics(actual_test.to_numpy().ravel(), last_year.to_numpy().ravel()) if actual_test.shape == last_year.shape else {"MAPE (%)": np.nan, "WAPE (%)": np.nan}

    baseline_rec_val = rolling_recursive_eval(None, wide, val_start, regions, countries, gdp_table, baseline=True)

    comparison_rows, bundles, all_params = [], {}, {}
    test_rows = table.loc[is_test, ["airport", "date", "y", "baseline_pred"]].reset_index(drop=True)
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    step("3/6 모델 학습 · 튜닝 · 평가")
    for name in available_models():
        t0 = time.time()
        if tune:
            params = tune_model(name, X[is_train], y[is_train], X[is_val], y[is_val])
        else:
            params = saved_params.get(name, {})
        all_params[name] = params

        # (a) Train으로 학습 -> Validation 평가
        m_val = make_model(name, params).fit(X[is_train], y[is_train])
        val_pred = to_level(table.loc[is_val, "base"], m_val.predict(X[is_val]))
        val_m = level_metrics(table.loc[is_val, "y"], val_pred)

        rec_val_m = rolling_recursive_eval(m_val, wide, val_start, regions, countries, gdp_table)

        # (b) Train+Validation으로 학습 -> Test 평가 (한 달 앞 예측 + 12개월 재귀 예측)
        is_dev = is_train | is_val
        m_dev = make_model(name, params).fit(X[is_dev], y[is_dev])
        test_pred = to_level(table.loc[is_test, "base"], m_dev.predict(X[is_test]))
        test_m = level_metrics(table.loc[is_test, "y"], test_pred)
        rec = recursive_forecast(m_dev, FEATURE_COLUMNS, history_before_test, regions, countries, gdp_table, n_months=TEST_MONTHS)
        rec_actual = actual_test.reindex(columns=rec.columns)
        rec_m = level_metrics(rec_actual.to_numpy().ravel(), rec.to_numpy().ravel())
        rec_long = (
            rec.stack().rename("predicted").reset_index().rename(columns={"level_1": "date"})
        )
        rec_long.columns = ["airport", "date", "predicted"]
        actual_long = rec_actual.stack().rename("actual").reset_index()
        actual_long.columns = ["airport", "date", "actual"]
        rec_long = rec_long.merge(actual_long, on=["airport", "date"], how="left")

        # (c) 전체 데이터로 최종 모델 학습 (앱에서 사용)
        final_model = make_model(name, params).fit(X, y)

        preds = test_rows.copy()
        preds["predicted"] = np.asarray(test_pred)
        preds = preds.rename(columns={"y": "actual", "baseline_pred": "baseline"})

        bundles[name] = {
            "model": final_model,
            "model_name": name,
            "slug": MODEL_SLUGS[name],
            "feature_columns": FEATURE_COLUMNS,
            "target_mode": "log_ratio_to_12m_mean",
            "best_params": params,
            "val_metrics": val_m,
            "test_metrics": test_m,
            "recursive_metrics": rec_m,
            "recursive_val_metrics": rec_val_m,
            "baseline_val_metrics": baseline_val,
            "baseline_test_metrics": baseline_test,
            "test_predictions": preds,
            "recursive_predictions": rec_long,
            "feature_importance": feature_importance(final_model),
            "trained_at": trained_at,
            "n_train_rows": int(len(table)),
        }
        comparison_rows.append(comparison_row(name, val_m, rec_val_m, test_m, rec_m))
        step(f"{name} 완료 | Val WAPE {val_m['WAPE (%)']:.2f}% | Val 12개월 재귀 WAPE {rec_val_m['WAPE (%)']:.2f}% | Test WAPE {test_m['WAPE (%)']:.2f}% | 12개월 재귀 WAPE {rec_m['WAPE (%)']:.2f}%", t0)

    comparison = pd.DataFrame(comparison_rows)
    comparison.loc[len(comparison)] = comparison_row(BASELINE_NAME, baseline_val, baseline_rec_val, baseline_test, baseline_rec)
    # 선정 기준: 앱의 연간 곡선과 같은 방식인 'Validation 12개월 재귀 WAPE' (없으면 한 달 앞 Validation WAPE)
    comparison["_key"] = comparison[SELECTION_METRIC].fillna(comparison["Val WAPE (%)"])
    comparison = comparison.sort_values("_key").reset_index(drop=True)

    step(f"4/6 모델 선정 ({SELECTION_METRIC} 기준, 기준선 제외)")
    ranked = comparison[comparison["Model"] != BASELINE_NAME]
    best_name = ranked.iloc[0]["Model"]
    best_row = ranked.iloc[0]
    baseline_key = comparison.loc[comparison["Model"] == BASELINE_NAME, "_key"].iloc[0]
    if best_row["_key"] >= baseline_key:
        step("   [주의] 어떤 모델도 '작년 같은 달' 기준선보다 Validation 오차가 낮지 않습니다. 데이터/피처를 점검하세요.")
    comparison = comparison.drop(columns="_key")

    step("5/6 저장 (.pkl)")
    for name, bundle in bundles.items():
        joblib.dump(bundle, save_dir / f"model_{bundle['slug']}.pkl")
    joblib.dump(bundles[best_name], save_dir / "demand_model.pkl")
    context = {
        "history": wide,             # 학습/평가용 (모든 공항 시점을 맞춰 자른 표)
        "history_full": wide_full,   # 예측용 (공항별 마지막 실측 달까지)
        "airport_funnel": airport_funnel(data),
        "meta": meta,
        "gdp_table": gdp_table,
        "last_date": dates[-1],
        "first_date": dates[0],
        "val_start": val_start,
        "test_start": test_start,
        "feature_columns": FEATURE_COLUMNS,
        "best_model": best_name,
        "selection_metric": SELECTION_METRIC,
        "trained_at": trained_at,
        "n_rows": int(len(table)),
        "split_rows": {"train": int(is_train.sum()), "val": int(is_val.sum()), "test": int(is_test.sum())},
    }
    context["region_summary"] = region_summary(context, table)
    joblib.dump(context, save_dir / "airport_context.pkl")
    comparison.to_csv(save_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    if tune:
        params_path.write_text(json.dumps(all_params, ensure_ascii=False, indent=2), encoding="utf-8")

    best_bundle = bundles[best_name]
    per_airport = best_bundle["test_predictions"].copy()
    per_airport["ape"] = (per_airport["actual"] - per_airport["predicted"]).abs() / per_airport["actual"].replace(0, np.nan) * 100
    (
        per_airport.groupby("airport")["ape"].mean().rename("Test MAPE (%)").reset_index()
        .merge(meta.reset_index()[["airport", "airport_name", "country", "region"]], on="airport", how="left")
        .sort_values("Test MAPE (%)")
        .to_csv(save_dir / "airport_test_mape.csv", index=False, encoding="utf-8-sig")
    )

    step(f"6/6 완료 | 최고 모델: {best_name} ({SELECTION_METRIC} {best_row[SELECTION_METRIC]:.2f}% vs 기준선 {baseline_key:.2f}%, Test WAPE {best_row['Test WAPE (%)']:.2f}%)", started)
    return {"best_model": best_name, "comparison": comparison}


if __name__ == "__main__":
    result = train_airport(tune="--no-tune" not in sys.argv[1:])
    print("\n" + result["comparison"].round(3).to_string(index=False))