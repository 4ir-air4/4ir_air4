"""app.py의 두 번째 탭: 📊 항공 교통량 예측 및 모델 비교.

app.py 가 너무 길어져서 따로 뺐습니다. 내용은 그대로이고, app.py 에서는
    from _2_traffic_tab import show_traffic_prediction_tab
    with tab_prediction:
        show_traffic_prediction_tab()
처럼 불러서 씁니다.
"""

from datetime import datetime
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# 설정 (이 탭 전용)
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent

MODEL_DIR = ROOT_DIR / "models" / "saved"
TRAIN_SCRIPT = ROOT_DIR / "models" / "train.py"
LIVE_DATA_PATH = ROOT_DIR / "data" / "raw" / "aircraftlive_counts.csv"
HISTORICAL_DATA_PATH = ROOT_DIR / "data" / "raw" / "btstranstats_counts.csv"

# 마지막 수집 후 이 시간(분)이 지나면 수집기가 멈춘 것으로 보고 경고한다
LIVE_STALE_MINUTES = 15
# 학습 샘플이 이보다 적으면(약 1주일치 미만) 성능 수치가 불안정하다고 안내한다
RECOMMENDED_SAMPLES = 2000
# 실시간 탭에서 보여줄 미래 예측 시점(분). 5분 단위 배수로 자유롭게 바꿔도 된다.
HORIZON_MINUTES = [5, 10, 30, 60, 180]

# 모델은 데이터 성격별로 따로 학습한다 (models/saved/<key>/ 아래에 저장)
DATASETS = {
    "현재 (실시간 수집)": {
        "key": "current",
        "is_live": True,
        "caption": (
            "실시간 API 스냅샷에서 위치가 확인되는 항공기 수를 5분마다 "
            "수집한 데이터로 학습한 모델입니다. "
            "실제 승객 수나 항공사 예약 수요를 예측하는 값은 아닙니다."
        ),
    },
    "과거 (BTS TranStats 샘플)": {
        "key": "historical",
        "is_live": False,
        "caption": (
            "BTS TranStats 과거 시간별 데이터에서 5분 구간별 고유 항공기(icao24) 수를 "
            "집계해 학습한 모델입니다. 측정 방식이 실시간 수집 데이터와 달라 "
            "두 결과의 수치를 직접 비교할 수는 없습니다."
        ),
    },
}

MODEL_SLUGS = {
    "Linear Regression": "linear_regression",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
}


def get_model_path(dataset_key, model_name):
    return MODEL_DIR / dataset_key / f"model_{MODEL_SLUGS[model_name]}.pkl"


def get_comparison_file(dataset_key):
    return MODEL_DIR / dataset_key / "model_comparison.csv"


# =========================================================
# 10. 모델 불러오기
# =========================================================

@st.cache_resource(show_spinner=False, max_entries=8)
def load_model_bundle(model_path, model_mtime=0.0):
    """model_mtime이 인자에 들어 있어서, 모델 파일이 바뀌면 캐시가 자동으로 새로 만들어진다."""

    path = Path(model_path)

    if not path.exists():
        return None

    return joblib.load(path)


def get_model_bundle(model_path):
    """터미널에서 train.py를 돌려 모델 파일이 바뀐 경우에도 화면에 바로 반영되게 한다."""

    path = Path(model_path)
    mtime = path.stat().st_mtime if path.exists() else 0.0

    return load_model_bundle(str(path), mtime)


def load_trainer():
    """models/train.py를 모듈로 불러온다. (재학습 버튼, 실시간 입력 계산에 사용)"""

    spec = importlib.util.spec_from_file_location(
        "aircraft_trainer",
        TRAIN_SCRIPT,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@st.cache_data(ttl=60, show_spinner=False)
def load_live_input(csv_path, csv_mtime, feature_columns):
    """CSV의 가장 최근 시점으로 예측 입력 1행을 만든다. (csv_mtime은 캐시 갱신용)"""

    trainer = load_trainer()

    return trainer.build_latest_features(
        csv_path,
        list(feature_columns),
    )


def render_retrain_controls(dataset_key):
    """실시간(current) 모델을 최신 CSV로 다시 학습하는 버튼."""

    message = st.session_state.pop("retrain_message", None)

    if message:
        st.success(message)

    bundle = get_model_bundle(
        MODEL_DIR / dataset_key / "demand_model.pkl"
    )

    col_button, col_info = st.columns([1, 3])

    with col_info:

        if bundle is not None and bundle.get("trained_at"):

            trained_at = pd.Timestamp(bundle["trained_at"]).tz_convert("Asia/Seoul")

            st.caption(
                f"마지막 학습: {trained_at:%Y-%m-%d %H:%M} (KST) · "
                f"학습 샘플 {bundle.get('n_samples', '?')}개"
            )

        elif bundle is None:

            st.caption("아직 학습된 모델이 없습니다. 버튼을 눌러 학습해주세요.")

        if LIVE_DATA_PATH.exists():

            modified = datetime.fromtimestamp(
                LIVE_DATA_PATH.stat().st_mtime
            )

            st.caption(f"수집 데이터 마지막 갱신: {modified:%Y-%m-%d %H:%M}")

    with col_button:

        clicked = st.button(
            "🔄 최신 데이터로 재학습",
            key=f"retrain_{dataset_key}",
            help="data_collector.py가 모은 최신 CSV로 모델을 다시 학습합니다.",
        )

    if not clicked:
        return

    with st.spinner("재학습 중입니다. 잠시만 기다려주세요..."):

        try:

            trainer = load_trainer()

            result = trainer.train_dataset(
                dataset_key,
                trainer.DATASETS[dataset_key],
            )

        except Exception as exc:

            st.error(f"재학습에 실패했습니다: {exc}")

            return

    # 캐시에 남아 있는 예전 모델/입력값을 비워야 새 결과가 화면에 반영된다
    load_model_bundle.clear()
    load_live_input.clear()

    st.session_state["retrain_message"] = (
        f"재학습이 끝났습니다. 최고 모델: {result['best_model']} "
        f"(학습 샘플 {result['n_samples']:,}개)"
    )

    st.rerun()


# =========================================================
# 11. 항공 교통량 예측
# =========================================================

# ---------------------------------------------------------
# 그래프 / 여러 시점 예측 도우미 (화면에 보이는 시각은 모두 UTC)
# ---------------------------------------------------------

THIN_STEPS = [
    "5min", "10min", "15min", "30min", "1h", "2h", "3h", "6h", "12h", "1D", "7D", "30D",
]

WINDOW_OPTIONS = {
    "6시간": pd.Timedelta(hours=6),
    "1일": pd.Timedelta(days=1),
    "7일": pd.Timedelta(days=7),
    "30일": pd.Timedelta(days=30),
    "1년": pd.Timedelta(days=365),
    "전체": None,
}

# Streamlit 1.37 이상이면 fragment로 감싸서, 구간을 바꿀 때 그래프 부분만 다시 그린다.
# (지도 탭 전체가 다시 실행되지 않는다. 이전 버전이면 그냥 일반 함수로 동작)
st_fragment = getattr(st, "fragment", None) or (lambda func: func)


def fmt_utc(ts):
    """시각을 'YYYY-MM-DD HH:MM' (UTC) 문자열로. tz가 없으면 UTC로 본다."""

    ts = pd.Timestamp(ts)

    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")

    return ts.strftime("%Y-%m-%d %H:%M")


def horizon_label(minutes):
    if minutes % 60 == 0:
        return f"{minutes // 60}시간 뒤"

    return f"{minutes}분 뒤"


@st.cache_resource(show_spinner=False, max_entries=4)
def load_counts_csv(csv_path, csv_mtime):
    """timestamp / aircraft_count 두 컬럼만 읽는다. 시각은 UTC (tz 없음). csv_mtime은 캐시 갱신용"""

    df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        usecols=["timestamp", "aircraft_count"],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    df["aircraft_count"] = pd.to_numeric(df["aircraft_count"], errors="coerce")

    return (
        df.dropna()
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def thin_for_chart(df, time_col, value_cols, start=None, end=None, max_points=1500):
    """보려는 구간[start, end]만 잘라서, 점이 너무 많으면 시간 단위로 평균낸다.

    - 구간이 짧을수록 더 촘촘하게(최대 5분 원본) 보여준다.
    - 데이터가 없는 시간대는 NaN이 되어 선이 이어지지 않고 끊겨 보인다.
    - 반환: (그래프용 DataFrame, 사용한 간격 문자열). 시각은 UTC (tz 없음)
    """

    data = df[[time_col] + list(value_cols)].copy()
    data[time_col] = pd.to_datetime(data[time_col], utc=True, errors="coerce")
    data = data.dropna(subset=[time_col]).set_index(time_col).sort_index()
    data.index = data.index.tz_convert(None)

    if start is not None or end is not None:
        data = data.loc[start:end]

    if data.empty:
        return data.reset_index(), None

    low = start if start is not None else data.index[0]
    high = end if end is not None else data.index[-1]
    span = max(high - low, pd.Timedelta(minutes=5))

    freq = THIN_STEPS[-1]

    for candidate in THIN_STEPS:
        if span / pd.Timedelta(candidate) <= max_points:
            freq = candidate
            break

    return data.resample(freq).mean().reset_index(), freq


def pick_window(data_min, data_max, key, default_label, pad=pd.Timedelta(0)):
    """보기 구간(6시간~전체)과 좌우 이동 슬라이더를 보여주고 (시작, 끝) UTC를 돌려준다.

    pad: 가장 최근 끝에 미래 예측 구간만큼 여유를 더한다.
    """

    labels = list(WINDOW_OPTIONS)

    label = st.radio(
        "보기 구간",
        labels,
        index=labels.index(default_label),
        horizontal=True,
        key=f"{key}_window",
    )

    window = WINDOW_OPTIONS[label]
    bound_end = data_max + pad

    if window is None or bound_end - data_min <= window:
        return data_min, bound_end

    shift = window / 2
    max_offset = int(np.ceil((bound_end - data_min - window) / shift))
    offset = 0

    if max_offset > 0:

        # 범위가 바뀌면(새 데이터가 쌓이면) 슬라이더를 새로 만들어 가장 최근으로 돌아오게 한다.
        value = st.slider(
            "구간 이동 (오른쪽 끝 = 가장 최근, 왼쪽 = 과거)",
            min_value=-max_offset,
            max_value=0,
            value=0,
            key=f"{key}_offset_{label}_{max_offset}",
        )

        offset = -value

    end = bound_end - shift * offset
    start = end - window

    if start < data_min:
        start = data_min
        end = start + window

    return start, end


# ---------------------------------------------------------
# 실시간: 여러 시점 미래 예측
# ---------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def compute_live_forecast(csv_path, csv_mtime, model_path, model_mtime, feature_columns):
    """저장된 5분 예측 모델로 (1) 지난 구간의 5분 뒤 예측, (2) 앞으로 최대 3시간 예측 경로를 만든다.

    미래 예측은 5분 모델을 반복 적용한다. 예측값을 다음 입력(lag, 이동평균 등)으로 넣고,
    날씨·휴일 같은 추가 피처는 마지막 관측값을 그대로 유지한다.
    train.py의 make_features를 그대로 써서 학습 때와 같은 방식으로 피처를 만든다.
    """

    bundle = load_model_bundle(model_path, model_mtime)

    if bundle is None or bundle.get("target_mode") != "delta":
        return {
            "status": "error",
            "message": "이전 방식 모델이라 예측할 수 없습니다. '최신 데이터로 재학습'을 눌러주세요.",
        }

    model = bundle["model"]
    cols = list(feature_columns)

    trainer = load_trainer()
    interval = pd.Timedelta(minutes=trainer.INTERVAL_MINUTES)

    df = trainer.load_and_prepare_data(csv_path, min_rows=1)
    data = trainer.make_features(df)

    missing = [c for c in cols if c not in data.columns]

    if missing:
        return {
            "status": "error",
            "message": (
                f"모델이 쓰는 피처({', '.join(missing)})가 현재 CSV에 없습니다. "
                "재학습해주세요."
            ),
        }

    last_segment = df[df["segment"] == df["segment"].iloc[-1]]

    result = {
        "status": "short",
        "segment_len": int(len(last_segment)),
        "last_ts": df["timestamp"].iloc[-1].tz_convert(None),
        "current": float(df["aircraft_count"].iloc[-1]),
        "past": None,
        "path": None,
    }

    # (1) 지난 구간: 각 시점에서 '5분 뒤'를 예측한 값 (예측값이 놓일 시각 = 시점 + 5분)
    features = data[cols]
    usable = features.notna().all(axis=1)

    if usable.any():

        delta = model.predict(features[usable])

        result["past"] = pd.DataFrame(
            {
                "timestamp": (data.loc[usable, "timestamp"] + interval).dt.tz_convert(None),
                "pred": np.maximum(data.loc[usable, "lag_0"].to_numpy() + delta, 0),
            }
        )

    if not bool(usable.iloc[-1]):
        return result

    # (2) 미래: 5분씩 반복 예측
    max_steps = int(max(HORIZON_MINUTES) // trainer.INTERVAL_MINUTES)
    work = last_segment.tail(40).reset_index(drop=True)
    rows = []

    for step_number in range(1, max_steps + 1):

        feats = trainer.make_features(work)[cols].iloc[[-1]]

        if feats.isna().to_numpy().any():
            break

        current = float(work["aircraft_count"].iloc[-1])
        predicted = max(0.0, current + float(model.predict(feats)[0]))

        new_row = work.iloc[[-1]].copy()
        new_row["timestamp"] = work["timestamp"].iloc[-1] + interval
        new_row["aircraft_count"] = predicted
        work = pd.concat([work, new_row], ignore_index=True)

        rows.append(
            {
                "minutes": step_number * trainer.INTERVAL_MINUTES,
                "timestamp": new_row["timestamp"].iloc[0].tz_convert(None),
                "pred": predicted,
            }
        )

    if rows:
        result["status"] = "ok"
        result["path"] = pd.DataFrame(rows)

    return result


def render_horizon_metrics(forecast, bundle):
    """5분 / 10분 / 30분 / 1시간 / 3시간 뒤 예측값을 카드로 보여준다."""

    path = forecast["path"]
    current = forecast["current"]

    columns = st.columns(len(HORIZON_MINUTES))

    for column, minutes in zip(columns, HORIZON_MINUTES):

        row = path[path["minutes"] == minutes]

        if row.empty:
            column.metric(horizon_label(minutes), "-")
            continue

        value = round(float(row["pred"].iloc[0]))

        column.metric(
            horizon_label(minutes),
            f"{value:,}대",
            delta=f"{value - round(current):+,}대",
        )

    note = (
        f"{fmt_utc(forecast['last_ts'])} UTC 관측값 {round(current):,}대 기준이고, 증감은 이 값 대비입니다. "
        "10분 이후는 5분 예측 모델을 반복 적용한 추정이라 (날씨·휴일은 마지막 값 유지) "
        "멀수록 오차가 커집니다."
    )

    n_samples = bundle.get("n_samples")

    if n_samples is not None and n_samples < RECOMMENDED_SAMPLES:
        note += f" 학습 샘플이 {n_samples:,}개뿐이라 1시간 이상 뒤 예측은 참고용으로만 보세요."

    st.caption(note)


def build_live_figure(history, forecast, start, end):
    """최근 실제값 + 지난 5분 뒤 예측 + 앞으로의 예측 경로"""

    fig = go.Figure()

    actual, freq = thin_for_chart(history, "timestamp", ["aircraft_count"], start, end)

    fig.add_trace(
        go.Scatter(
            x=actual["timestamp"],
            y=actual["aircraft_count"],
            mode="lines",
            name="실제 항공기 수",
            line=dict(color="#1f77b4"),
            connectgaps=False,
        )
    )

    past = forecast.get("past") if forecast else None

    if past is not None and not past.empty:

        past_view, _ = thin_for_chart(past, "timestamp", ["pred"], start, end)

        fig.add_trace(
            go.Scatter(
                x=past_view["timestamp"],
                y=past_view["pred"],
                mode="lines",
                name="5분 뒤 예측 (저장된 모델)",
                line=dict(color="#ff7f0e", dash="dot"),
                connectgaps=False,
            )
        )

    path = forecast.get("path") if forecast else None

    if path is not None and not path.empty:

        origin = pd.DataFrame(
            {"timestamp": [forecast["last_ts"]], "pred": [forecast["current"]]}
        )
        line = pd.concat([origin, path[["timestamp", "pred"]]], ignore_index=True)

        fig.add_trace(
            go.Scatter(
                x=line["timestamp"],
                y=line["pred"],
                mode="lines",
                name="미래 예측 경로",
                line=dict(color="#d62728", dash="dash"),
            )
        )

        marks = path[path["minutes"].isin(HORIZON_MINUTES)]

        fig.add_trace(
            go.Scatter(
                x=marks["timestamp"],
                y=marks["pred"],
                mode="markers+text",
                text=[horizon_label(int(m)) for m in marks["minutes"]],
                textposition="top center",
                name="예측 시점",
                marker=dict(color="#d62728", size=9),
            )
        )

    fig.update_layout(
        title="실시간 항공기 수: 수집 데이터와 미래 예측",
        hovermode="x unified",
        xaxis_title="시각 (UTC)",
        yaxis_title="항공기 수",
        legend_title_text="",
    )
    fig.update_xaxes(range=[start, end])

    return fig, freq


@st_fragment
def render_live_chart_panel(history, forecast):

    has_path = bool(forecast) and forecast.get("path") is not None
    pad = pd.Timedelta(minutes=max(HORIZON_MINUTES)) if has_path else pd.Timedelta(0)

    start, end = pick_window(
        history["timestamp"].iloc[0],
        history["timestamp"].iloc[-1],
        "live",
        "6시간",
        pad,
    )

    fig, freq = build_live_figure(history, forecast, start, end)

    st.plotly_chart(fig, use_container_width=True)

    note = (
        f"수집 {len(history):,}개 · {fmt_utc(history['timestamp'].iloc[0])} ~ "
        f"{fmt_utc(history['timestamp'].iloc[-1])} (UTC) · 선이 끊긴 곳은 수집이 멈춘 시간입니다."
    )

    if freq not in (None, "5min"):
        note += f" 점이 많아 {freq} 평균으로 표시합니다."

    st.caption(note)


def render_live_section(model_path, bundle, feature_columns, is_legacy):
    """실시간 탭: 여러 시점 미래 예측 카드 + 최근 수집 데이터 그래프"""

    if not LIVE_DATA_PATH.exists():
        return

    csv_mtime = LIVE_DATA_PATH.stat().st_mtime

    try:
        history = load_counts_csv(str(LIVE_DATA_PATH), csv_mtime)
    except Exception as exc:
        st.warning(f"수집 데이터를 불러오지 못했습니다: {exc}")
        return

    if history.empty:
        return

    forecast = None

    if not is_legacy:

        path = Path(model_path)

        try:
            forecast = compute_live_forecast(
                str(LIVE_DATA_PATH),
                csv_mtime,
                str(path),
                path.stat().st_mtime if path.exists() else 0.0,
                tuple(feature_columns),
            )
        except Exception as exc:
            st.warning(f"미래 예측을 계산하지 못했습니다: {exc}")

    status = forecast.get("status") if forecast else None

    if status == "error":
        st.warning(forecast["message"])

    elif status == "ok":
        render_horizon_metrics(forecast, bundle)

    elif status == "short":
        st.caption(
            f"현재 연속 수집 {forecast['segment_len']}개 · "
            "미래 예측에는 연속 13개(약 1시간)가 필요합니다."
        )

    render_live_chart_panel(
        history,
        forecast if status in ("ok", "short") else None,
    )


# ---------------------------------------------------------
# 테스트 구간 / 과거 데이터 그래프
# ---------------------------------------------------------

LABELS = {"actual": "실제", "predicted": "예측", "baseline": "직전값(기준선)"}


def render_test_chart(test_predictions):
    """학습에 쓰지 않은 맨 뒤 구간의 실제 vs 예측 (실시간 탭에서는 접어서 보여줌)"""

    value_cols = [c for c in LABELS if c in test_predictions.columns]

    chart_df, freq = thin_for_chart(test_predictions, "timestamp", value_cols)
    plot_df = chart_df.rename(columns=LABELS)

    fig = px.line(
        plot_df,
        x="timestamp",
        y=[LABELS[c] for c in value_cols],
        title="테스트 구간: 실제 vs 예측 항공기 수",
        labels={"timestamp": "시각 (UTC)", "value": "항공기 수", "variable": ""},
    )
    fig.update_layout(hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True)

    note = (
        f"학습 샘플의 맨 뒤 일부(기본 15%)만 보여줍니다 ({len(test_predictions):,}개, "
        f"{fmt_utc(test_predictions['timestamp'].iloc[0])} ~ "
        f"{fmt_utc(test_predictions['timestamp'].iloc[-1])} UTC)."
    )

    if freq not in (None, "5min"):
        note += f" 점이 많아 {freq} 평균으로 표시합니다."

    st.caption(note)


def build_historical_figure(counts, test_predictions, start, end):

    fig = go.Figure()
    freq = None

    if counts is not None:

        actual, freq = thin_for_chart(counts, "timestamp", ["aircraft_count"], start, end)
        actual = actual.rename(columns={"aircraft_count": "actual"})

    else:

        actual, freq = thin_for_chart(test_predictions, "timestamp", ["actual"], start, end)

    fig.add_trace(
        go.Scatter(
            x=actual["timestamp"],
            y=actual["actual"],
            mode="lines",
            name="실제",
            line=dict(color="#1f77b4"),
            connectgaps=False,
        )
    )

    colors = {"predicted": "#ff7f0e", "baseline": "#d62728"}

    for column in ["predicted", "baseline"]:

        if column not in test_predictions.columns:
            continue

        view, _ = thin_for_chart(test_predictions, "timestamp", [column], start, end)

        fig.add_trace(
            go.Scatter(
                x=view["timestamp"],
                y=view[column],
                mode="lines",
                name=LABELS[column],
                line=dict(color=colors[column]),
                connectgaps=False,
            )
        )

    fig.update_layout(
        title="과거 데이터: 실제 항공기 수와 테스트 구간 예측",
        hovermode="x unified",
        xaxis_title="시각 (UTC)",
        yaxis_title="항공기 수",
        legend_title_text="",
    )
    fig.update_xaxes(range=[start, end])

    return fig, freq


@st_fragment
def render_historical_panel(test_predictions):
    """CSV 전체 기간의 실제값 위에, 테스트 구간 예측을 겹쳐서 보여준다."""

    counts = None

    if HISTORICAL_DATA_PATH.exists():

        try:
            counts = load_counts_csv(
                str(HISTORICAL_DATA_PATH),
                HISTORICAL_DATA_PATH.stat().st_mtime,
            )
        except Exception as exc:
            st.warning(f"과거 CSV를 불러오지 못해 테스트 구간만 표시합니다: {exc}")

    if counts is not None and counts.empty:
        counts = None

    test_times = pd.to_datetime(
        test_predictions["timestamp"], utc=True
    ).dt.tz_convert(None)

    if counts is not None:
        data_min, data_max = counts["timestamp"].iloc[0], counts["timestamp"].iloc[-1]
    else:
        data_min, data_max = test_times.min(), test_times.max()

    # 기본은 가장 최근 30일(예측이 있는 구간). '전체'나 '1년'으로 넓히고 슬라이더로 과거까지 옮겨 볼 수 있다.
    start, end = pick_window(data_min, data_max, "hist", "30일")

    fig, freq = build_historical_figure(counts, test_predictions, start, end)

    st.plotly_chart(fig, use_container_width=True)

    note = (
        f"실제값: {fmt_utc(data_min)} ~ {fmt_utc(data_max)} (UTC) 전체 기간. "
        f"예측값은 학습에 쓰지 않은 테스트 구간({fmt_utc(test_times.min())} ~ "
        f"{fmt_utc(test_times.max())} UTC)에만 있습니다."
    )

    if freq not in (None, "5min"):
        note += f" 점이 많아 {freq} 평균으로 표시합니다. 구간을 줄이면 더 촘촘하게 보입니다."

    st.caption(note)


def show_model_prediction(dataset_key):

    dataset = next(
        d for d in DATASETS.values() if d["key"] == dataset_key
    )

    st.subheader("📈 항공 교통량 예측")

    selected_model_name = st.selectbox(
        "예측 모델 선택",
        list(MODEL_SLUGS.keys()),
        key=f"model_select_{dataset_key}",
    )

    model_path = get_model_path(dataset_key, selected_model_name)

    bundle = get_model_bundle(model_path)

    if bundle is None:

        st.warning(
            f"모델 파일이 없습니다.\n\n{model_path}\n\n"
            "먼저 `python ojh/models/train.py`를 실행해주세요."
        )

        return

    model = bundle.get("model")

    feature_columns = bundle.get(
        "feature_columns",
        [],
    )

    latest_features = bundle.get(
        "latest_features",
        None,
    )

    interval_minutes = bundle.get(
        "interval_minutes",
        5,
    )

    if model is None or not feature_columns:

        st.error(
            "모델 파일의 데이터 구조를 확인해주세요."
        )

        return

    first_ts = bundle.get("first_data_timestamp")
    last_ts = bundle.get("last_data_timestamp")

    if first_ts is not None and last_ts is not None:
        st.caption(
            f"학습 데이터 기간: {fmt_utc(first_ts)} ~ {fmt_utc(last_ts)} (UTC)"
        )

    n_samples = bundle.get("n_samples")

    if n_samples is not None and n_samples < RECOMMENDED_SAMPLES:

        st.warning(
            f"학습 샘플이 {n_samples:,}개로 아직 적습니다(약 1주일치 미만). "
            "성능 수치가 불안정할 수 있으니 수집을 더 이어간 뒤 재학습해주세요."
        )

    # 실시간 모델에서만 레거시 모델 경고 및 재학습 안내를 제공
    is_legacy_model = bundle.get("target_mode") != "delta"

    if dataset["is_live"] and is_legacy_model:
        st.warning(
            "이전 방식으로 학습된 모델입니다. "
            "'최신 데이터로 재학습'을 눌러 새 모델로 갱신해주세요."
        )

    # 현재 모델만 "다음 5분 예측"을 보여준다.
    # 과거 데이터는 이미 지난 시점이라 미래 예측이 의미가 없다.
    if dataset["is_live"]:

        input_df = None
        last_observed = last_ts

        # 화면을 열 때마다 CSV의 최신 값으로 예측 입력을 새로 만든다.
        # (학습할 때 저장해 둔 값을 그대로 쓰면 학습 시점의 예측만 계속 보이게 된다)
        if not is_legacy_model and LIVE_DATA_PATH.exists():

            try:

                input_df, live_ts = load_live_input(
                    str(LIVE_DATA_PATH),
                    LIVE_DATA_PATH.stat().st_mtime,
                    tuple(feature_columns),
                )

                if live_ts is not None:
                    last_observed = live_ts

            except Exception as exc:

                st.warning(f"최신 수집 데이터를 불러오지 못했습니다: {exc}")

        if input_df is None:

            if isinstance(latest_features, pd.DataFrame):
                input_df = latest_features[feature_columns]
            elif latest_features is not None:
                input_df = pd.DataFrame(
                    [latest_features],
                    columns=feature_columns,
                )

        if input_df is None or input_df[feature_columns].isna().to_numpy().any():

            st.info(
                "최근 연속 수집 데이터가 부족해 지금은 예측할 수 없습니다. "
                "수집이 중간에 끊겼다면 연속으로 약 1시간(관측값 13개) 이상 "
                "쌓여야 예측할 수 있습니다."
            )

        if last_observed is not None:

            age_minutes = (
                pd.Timestamp.now(tz="UTC") - pd.Timestamp(last_observed)
            ).total_seconds() / 60

            st.caption(
                f"마지막 수집: {pd.Timestamp(last_observed):%Y-%m-%d %H:%M} UTC "
                f"({age_minutes:.0f}분 전)"
            )

            if age_minutes > LIVE_STALE_MINUTES:

                st.warning(
                    "수집이 멈춘 것 같습니다. 수집기가 실행 중인지 확인해주세요. "
                    "(저장소 최상위에서 `python ojh/data_collector.py`)"
                )

        render_live_section(model_path, bundle, feature_columns, is_legacy_model)

    else:

        st.info(
            "과거 데이터 모델은 미래 예측 대신, 테스트 구간에서 "
            "모델이 예측한 값과 실제 값을 비교해서 보여줍니다."
        )

    cv_metrics = bundle.get(
        "cv_metrics",
        {},
    )

    test_metrics = bundle.get(
        "test_metrics",
        {},
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "교차검증 MAE",
        f"{cv_metrics.get('MAE', float('nan')):.2f}",
    )

    col2.metric(
        "교차검증 MAPE",
        f"{cv_metrics.get('MAPE (%)', float('nan')):.2f}%",
    )

    col3.metric(
        "테스트 MAE",
        f"{test_metrics.get('MAE', float('nan')):.2f}",
    )

    test_predictions = bundle.get("test_predictions")
    has_test = isinstance(test_predictions, pd.DataFrame) and not test_predictions.empty

    if dataset["is_live"]:

        if has_test:
            with st.expander("테스트 구간 평가 (학습에 쓰지 않은 맨 뒤 일부)"):
                render_test_chart(test_predictions)

    elif has_test:

        render_historical_panel(test_predictions)

    weather_columns = bundle.get("weather_columns") or []

    if weather_columns:

        WEATHER_LABELS = {
            "temp_mean": "평균 기온",
            "precip_mean": "평균 강수량",
            "cloud_mean": "평균 구름량",
            "wind_mean": "평균 풍속",
            "gust_mean": "평균 돌풍",
            "precip_max": "최대 강수량",
            "gust_max": "최대 돌풍",
            "rain_share": "비 오는 비율",
            "severe_weather_idx": "악천후 지수",
        }

        st.caption(
            "사용한 날씨 피처: "
            + ", ".join(WEATHER_LABELS.get(c, c) for c in weather_columns)
        )

    importance = bundle.get("feature_importance")

    if importance:

        importance_df = (
            pd.DataFrame(
                {
                    "피처": list(importance.keys()),
                    "중요도": list(importance.values()),
                }
            )
            .sort_values("중요도", ascending=False)
            .head(10)
            .sort_values("중요도")
        )

        importance_df["구분"] = importance_df["피처"].map(
            lambda c: "날씨" if c in weather_columns else "시간/과거값"
        )

        fig_importance = px.bar(
            importance_df,
            x="중요도",
            y="피처",
            color="구분",
            orientation="h",
            title="피처 중요도 (상위 10개)",
        )

        with st.expander("피처 중요도 보기"):
            st.plotly_chart(
                fig_importance,
                use_container_width=True,
            )

    st.caption(dataset["caption"])


# =========================================================
# 12. 모델 성능 비교
# =========================================================

def show_model_comparison(dataset_key):

    st.subheader("🏆 모델 성능 비교")

    comparison_file = get_comparison_file(dataset_key)

    if not comparison_file.exists():

        st.warning(
            "모델 비교 파일이 없습니다. "
            "먼저 `python ojh/models/train.py`를 실행해주세요."
        )

        return

    try:

        comparison_df = pd.read_csv(
            comparison_file
        )

    except Exception as exc:

        st.error(
            f"모델 비교 파일을 읽을 수 없습니다: {exc}"
        )

        return

    if comparison_df.empty:

        st.info(
            "비교할 모델 성능 데이터가 없습니다."
        )

        return

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Baseline (직전값)은 '5분 뒤에도 지금과 같다'고 가정한 기준선입니다. "
        "모델의 오차(MAE)가 이 기준선보다 낮아야 모델을 쓰는 의미가 있습니다."
    )

    # train.py가 저장하는 컬럼명: Model, CV MAE, CV MAPE (%)
    model_column = "Model" if "Model" in comparison_df.columns else None
    mae_column = "CV MAE" if "CV MAE" in comparison_df.columns else None
    mape_column = (
        "CV MAPE (%)" if "CV MAPE (%)" in comparison_df.columns else None
    )

    if model_column and mae_column:

        fig_mae = px.bar(
            comparison_df,
            x=model_column,
            y=mae_column,
            title="모델별 교차검증 MAE",
            labels={
                model_column: "모델",
                mae_column: "MAE",
            },
        )

        st.plotly_chart(
            fig_mae,
            use_container_width=True,
        )

    if model_column and mape_column:

        fig_mape = px.bar(
            comparison_df,
            x=model_column,
            y=mape_column,
            title="모델별 교차검증 MAPE",
            labels={
                model_column: "모델",
                mape_column: "MAPE (%)",
            },
        )

        st.plotly_chart(
            fig_mape,
            use_container_width=True,
        )

    # 날씨 포함/제외 비교 (날씨 데이터가 있는 경우에만)
    if (
        "Test MAE" in comparison_df.columns
        and "Test MAE (날씨 제외)" in comparison_df.columns
    ):

        weather_compare = comparison_df.dropna(
            subset=["Test MAE (날씨 제외)"]
        ).melt(
            id_vars="Model",
            value_vars=["Test MAE (날씨 제외)", "Test MAE"],
            var_name="구분",
            value_name="테스트 MAE",
        )

        weather_compare["구분"] = weather_compare["구분"].map(
            {
                "Test MAE (날씨 제외)": "날씨 제외",
                "Test MAE": "날씨 포함",
            }
        )

        fig_weather = px.bar(
            weather_compare,
            x="Model",
            y="테스트 MAE",
            color="구분",
            barmode="group",
            title="날씨 포함 vs 제외 (테스트 MAE, 낮을수록 좋음)",
            labels={"Model": "모델"},
        )

        st.plotly_chart(
            fig_weather,
            use_container_width=True,
        )

        st.caption(
            "날씨를 넣었을 때 오차가 줄어드는지 확인하는 비교입니다. "
            "차이가 거의 없다면 이 데이터에서는 날씨의 영향이 작다는 뜻입니다."
        )

    folds_file = (
        MODEL_DIR / dataset_key / "cross_validation_folds.csv"
    )

    if folds_file.exists():

        with st.expander("교차검증 Fold별 결과"):

            try:

                folds_df = pd.read_csv(
                    folds_file
                )

                st.dataframe(
                    folds_df,
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as exc:

                st.error(
                    f"Fold 결과를 읽을 수 없습니다: {exc}"
                )


# =========================================================
# 탭 진입점
# =========================================================

def show_traffic_prediction_tab():
    """app.py 두 번째 탭: 📊 항공 교통량 예측 및 모델 비교"""

    dataset_label = st.radio(
        "데이터 기준",
        list(DATASETS.keys()),
        horizontal=True,
        key="dataset_radio",
    )

    dataset_key = DATASETS[dataset_label]["key"]

    if DATASETS[dataset_label]["is_live"]:

        render_retrain_controls(dataset_key)

    prediction_tab, comparison_tab = st.tabs(
        [
            "🔮 항공 교통량 예측",
            "🏆 모델 성능 비교",
        ]
    )

    with prediction_tab:

        show_model_prediction(dataset_key)

    with comparison_tab:

        show_model_comparison(dataset_key)