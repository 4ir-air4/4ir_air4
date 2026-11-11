"""인천 지연 위험 예보 시연. 저장된 평가 구간 예측으로 서비스 화면을 보여 준다. 실전 기록이 쌓이면 입력만 바꾼다."""
import json

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score

from ldh.core.models import prev_indices
from ldh.core.store import Store
from ldh.core.ui import time_chart

TASKS = {"도착": "icn_arr_delay", "출발": "icn_dep_delay"}
GRADES = ["낮음", "보통", "높음"]
MODEL = "lightgbm"  # 입력을 맞추면 성능 차이가 작고, 점수가 실제 지연률에 가장 가깝다. WORK_LOG.md 참고
SHIFT ="2025-11"  # 출발 지연 분포가 이 달 초부터 단절된다. WORK_LOG.md 참고
store = Store("korea_ops")


@st.cache_data(max_entries=2)
def load(task, mtime):
    d = pd.read_parquet(store.table_path(f"pred_{task}__{MODEL}")).sort_values(["운항일", "계획"], ignore_index=True)
    d["등급"] = pd.cut(d["예측 점수"], [-np.inf, 0.3, 0.5, np.inf], right=False, labels=GRADES).astype(str)
    y = d["실제"].to_numpy()
    prev = prev_indices(d["편명"].factorize()[0], np.arange(len(d)), 5)
    known = prev >= 0
    count = known.sum(axis=1)
    # 단순 기준: 같은 편명의 직전 5회 지연률. 평가 구간 앞쪽의 첫 운항은 전체 평균을 쓴다
    d["단순 기준"] = np.where(count > 0, (y[prev] * known).sum(axis=1) / np.maximum(count, 1), y.mean())
    rows = []
    for name, part in (("2025-07~10", d[d["기간"] < SHIFT]), ("2025-11~2026-06", d[d["기간"] >= SHIFT])):
        rate = part.groupby("등급")["실제"].mean()
        rows.append({"구간": name, "편수": len(part), "실제 지연률": part["실제"].mean(), "평균 점수": part["예측 점수"].mean(),
                     "모델 PR-AUC": average_precision_score(part["실제"], part["예측 점수"]), "단순 기준 PR-AUC": average_precision_score(part["실제"], part["단순 기준"]),
                     **{f"{g} 지연률": rate.get(g, np.nan) for g in GRADES}, "높음 비중": (part["등급"] == "높음").mean()})
    return d, pd.DataFrame(rows)


st.info("실전 수집은 시작했지만 아직 모델과 연결하지 않아, 2025년 7월~2026년 6월 평가 구간의 저장된 예측으로 보여 줍니다. 실시간 예보가 아닙니다.", title="과거 자료 시연", icon=":material/history:")
direction = st.segmented_control("방향", list(TASKS), default="도착", required=True, help="기획안의 첫 대상은 도착편입니다.")
path = store.table_path(f"pred_{TASKS[direction]}__{MODEL}")
if not path.exists():
    st.warning(f"예측 파일이 없습니다. 드라이브의 `ldh/results/full/korea_ops/{path.name}` 을 이 프로젝트로 복사하세요.", icon=":material/folder_off:")
    st.stop()
data, periods = load(TASKS[direction], path.stat().st_mtime)
if direction == "출발":
    st.warning("2025년 11월 1일 전후로 출발 지연 분포 전체가 약 6분 이동했습니다. 원인은 확인 중이며, 단절 후에는 점수가 실제 지연을 높게 잡습니다.", title="출발 자료 단절", icon=":material/warning:")

recent = periods.iloc[-1]
st.caption(f"성능 지표는 최근 구간인 {recent['구간']} 기준입니다.")
with st.container(horizontal=True):
    st.metric("평가 편수", f"{recent['편수']:,}", border=True)
    st.metric("모델 PR-AUC", f"{recent['모델 PR-AUC']:.3f}", border=True, help="LightGBM. 지연편을 얼마나 잘 골라내는지 나타내며, 높을수록 좋습니다.")
    st.metric("단순 기준 PR-AUC", f"{recent['단순 기준 PR-AUC']:.3f}", border=True, help="같은 편명의 직전 5회 지연률을 그대로 쓴 값입니다.")
    st.metric("무작위 PR-AUC", f"{recent['실제 지연률']:.3f}", border=True, help="실제 지연 비율과 같습니다.")

st.subheader("날짜별 위험 예보")
first, last = pd.Timestamp(data["운항일"].iloc[0]).date(), pd.Timestamp(data["운항일"].iloc[-1]).date()
left, right = st.columns(2, vertical_alignment="bottom")
day = left.date_input("운항일", value=last, min_value=first, max_value=last)
picked = right.pills("위험 등급", GRADES, selection_mode="multi", default=GRADES)
daily = data[data["운항일"] == day.isoformat()]
with st.container(horizontal=True):
    st.metric(f"{direction} 편수", f"{len(daily):,}", border=True)
    st.metric("높음 등급", f"{(daily['등급'] == '높음').sum():,}", border=True)
    st.metric("실제 지연", f"{int(daily['실제'].sum()):,}", border=True, help="15분 이상 지연된 편수입니다.")
shown = daily[daily["등급"].isin(picked)].assign(결과=lambda t: t["실제"].map({0.0: "정상", 1.0: "지연"})).sort_values("예측 점수", ascending=False)
small = {"width": "small"}
st.dataframe(shown[["계획", "편명", "공항", "등급", "예측 점수", "결과"]], hide_index=True,
             column_config={"계획": st.column_config.TextColumn("예정 시각", **small), "편명": st.column_config.TextColumn(**small), "공항": st.column_config.TextColumn(**small), "등급": st.column_config.TextColumn("위험 등급", **small),
                            "예측 점수": st.column_config.ProgressColumn("위험 점수", format="percent", min_value=0, max_value=1), "결과": st.column_config.TextColumn(**small)},
             alt=f"{day} 인천 {direction}편별 위험 등급과 실제 결과")
st.caption("위험 점수가 높은 편부터 보여 줍니다. 이 시연의 예측 시점은 예정 시각 2시간 전이며, 실제 서비스의 예측 시점은 팀에서 정합니다. 결과 열은 다음 날 채웁니다.")

st.subheader("위험 등급을 믿을 수 있는가")
st.write("등급이 높을수록 실제 지연률도 높아야 합니다. 실전 기록에서도 이 순서가 유지되는지가 성공 기준입니다.")
left, right = st.columns(2)
graded = data.groupby("등급", as_index=False).agg(편수=("실제", "size"), 지연률=("실제", "mean")).set_index("등급").loc[GRADES].reset_index()
left.markdown("**등급별 실제 지연률**")
left.bar_chart(graded, x="등급", y="지연률", sort=False, horizontal=True, height=220, x_label="위험 등급", y_label="실제 지연률", alt="위험 등급별 실제 지연률")
monthly = data.groupby("기간", as_index=False).agg(실제=("실제", "mean"), 예측=("예측 점수", "mean")).melt("기간", var_name="구분", value_name="비율")
right.markdown("**월별 실제 지연률과 평균 점수**")
time_chart(right, monthly, x="기간", y="비율", color="구분", height=220, x_label="월", y_label="비율", alt="월별 실제 지연률과 평균 위험 점수")

st.markdown("**2025년 11월 전후 비교**")
percent = st.column_config.NumberColumn(format="percent")
st.dataframe(periods, hide_index=True, column_config={"편수": st.column_config.NumberColumn(format="localized"), "모델 PR-AUC": st.column_config.NumberColumn(format="%.3f"), "단순 기준 PR-AUC": st.column_config.NumberColumn(format="%.3f"),
                                                     **{c: percent for c in ("실제 지연률", "평균 점수", "낮음 지연률", "보통 지연률", "높음 지연률", "높음 비중")}}, alt="2025년 11월 전후의 지연률, 모델 성능, 등급별 실제 지연률")
st.caption("등급 기준은 점수 0.3과 0.5로 둔 제안값입니다. 고객이 놓친 지연과 잘못된 경보 중 무엇을 더 꺼리는지에 따라 조정합니다.")

eval_path = store.dir / f"icn_eval_{direction}.json"
if eval_path.exists():
    folds = pd.DataFrame(json.loads(eval_path.read_text(encoding="utf-8"))["rows"])
    picked_rows = folds[folds["모델"] == "LightGBM"].set_index("구간")
    base = folds[folds["모델"] == "단순 기준"].set_index("구간")
    table = pd.DataFrame({"실제 지연률": picked_rows["지연률"], "모델 PR-AUC": picked_rows["PR-AUC"], "단순 기준 PR-AUC": base["PR-AUC"], "낮음 지연률": picked_rows["원점수 낮음 지연률"], "보통 지연률": picked_rows["원점수 보통 지연률"],
                          "높음 지연률": picked_rows["원점수 높음 지연률"], "높음 포착률": picked_rows["원점수 높음 포착률"], "보통 이상 포착률": picked_rows["원점수 보통 이상 포착률"]}).reset_index()
    st.subheader("여러 시점 재평가")
    st.write("각 구간은 그 이전 자료로만 학습했습니다. 모든 구간에서 단순 기준보다 낫고 등급 순서도 유지되지만, 낮음 등급에 든 지연이 많아 모든 지연을 미리 잡아 주지는 못합니다.")
    st.dataframe(table, hide_index=True, column_config={"모델 PR-AUC": st.column_config.NumberColumn(format="%.3f"), "단순 기준 PR-AUC": st.column_config.NumberColumn(format="%.3f"),
                                                        **{c: percent for c in ("실제 지연률", "낮음 지연률", "보통 지연률", "높음 지연률", "높음 포착률", "보통 이상 포착률")}}, alt="구간별 재평가 결과")
    st.caption("포착률은 전체 지연편 중 해당 등급에 든 비율입니다. 첫 구간의 범주 코드표에는 이후 자료가 조금 반영되어 있습니다.")

forecast_path = store.dir / "icn_monthly_forecast.json"
if direction == "도착" and forecast_path.exists():
    forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
    trial = forecast["시험"]
    st.subheader("월별 도착 지연률 추정")
    st.write(f"작년 같은 달 값에 최근 수준 변화({forecast['비율']:.2f}배)를 곱했습니다. 2025년 11월 이후의 달은 이미 낮아진 값이라 그대로 씁니다. 편별 모델의 결과가 아닌 단순 계산입니다.")
    months = pd.DataFrame(forecast["달"]).assign(실제=lambda t: t["실제"].map(lambda v: "공개 전" if pd.isna(v) else f"{v:.2%}"))
    st.dataframe(months, hide_index=True, column_config={c: percent for c in ("작년 같은 달", "추정")}, alt="2026년 7월부터 11월까지 월별 도착 지연률 추정과 실제 값")
    st.caption(f"{forecast['만든 시각'][:10]}에 고정한 값입니다. 실제 값은 원본 파일이 공개되면 채웁니다. 2026년 3~6월로 시험한 평균 오차는 {trial['추정 평균 오차'] * 100:.1f}%포인트이며(작년 값 그대로 쓰면 {trial['작년 값 평균 오차'] * 100:.1f}%포인트), 4개월뿐인 시험입니다. "
               "지연률이 낮아진 원인을 아직 모르며, 낮아진 수준이 이어진다고 가정합니다. 태풍이나 폭설 같은 사건은 반영하지 않습니다.")
