"""미국 운항: 통계, 모델 성능, 예측."""
import datetime
import json

import joblib
import pandas as pd
import streamlit as st

from core.ui import header, load, model_tables, notes, prediction_section, store_for, time_chart

store = store_for(__file__)
summary = header(store)
DOW = {1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토", 7: "일"}

view = st.segmented_control("보기", ["통계", "모델 성능", "예측값", "예측 입력"], default="통계", label_visibility="collapsed") or "통계"

if view == "통계":
    yearly = load(store, "yearly")
    st.subheader("연도별 추이")
    left, right = st.columns(2)
    left.markdown("**운항 편수**")
    time_chart(left, yearly, x="Year", y="flights", x_label="연도", y_label="편", alt="연도별 운항 편수")
    right.markdown("**도착 지연률**")
    time_chart(right, yearly, x="Year", y="delay_rate", x_label="연도", y_label="비율", alt="연도별 도착 지연률")
    st.markdown("**결항률**")
    time_chart(st, yearly, x="Year", y="cancel_rate", x_label="연도", y_label="비율", height=240, alt="연도별 결항률")

    st.subheader("언제 늦는가 (2023년 이후)")
    a, b, c = st.columns(3)
    a.markdown("**월별 지연률** (2023~2025)")
    a.bar_chart(load(store, "month"), x="Month", y="delay_rate", x_label="월", y_label="지연률", alt="월별 지연률")
    b.markdown("**요일별 지연률**")
    dow = load(store, "dow").assign(요일=lambda d: d.DayOfWeek.map(DOW))
    b.bar_chart(dow, x="요일", y="delay_rate", sort=False, y_label="지연률", alt="요일별 지연률")
    c.markdown("**출발 시간대별 지연률**")
    hour = load(store, "hour").query("flights > 10000")
    c.bar_chart(hour, x="hour", y="delay_rate", x_label="예정 출발 시각", y_label="지연률", alt="출발 시간대별 지연률")

    st.subheader("누가, 어디서 늦는가 (2023년 이후)")
    columns = {"flights": "편수", "delay_rate": "지연률", "cancel_rate": "결항률", "avg_delay": "평균 지연(분)"}
    config = {"지연률": st.column_config.NumberColumn(format="percent"), "결항률": st.column_config.NumberColumn(format="percent"),
              "편수": st.column_config.NumberColumn(format="localized")}
    pick = st.segmented_control("대상", ["항공사", "출발 공항", "노선"], default="항공사") or "항공사"
    keys, name = {"항공사": (["항공사"], "airline"), "출발 공항": (["Origin"], "origin"), "노선": (["Origin", "Dest"], "route")}[pick]
    table = load(store, name).sort_values("flights", ascending=False)
    st.dataframe(table[keys + list(columns)].rename(columns=columns).round(3), hide_index=True, column_config=config, alt=f"{pick}별 지연·결항 통계")
    st.caption("공항과 노선은 편수 상위 50개입니다.")

    st.subheader("왜 늦고, 왜 결항하는가")
    left, right = st.columns(2)
    left.markdown("**지연 시간 원인 비중**")
    time_chart(left, load(store, "cause"), x="Year", y="비중", color="원인", x_label="연도", y_label="비중", alt="연도별 지연 원인 비중", kind="area")
    right.markdown("**결항 사유 비중**")
    time_chart(right, load(store, "cancel_code"), x="Year", y="비중", color="사유", x_label="연도", y_label="비중", alt="연도별 결항 사유 비중", kind="area")

elif view == "모델 성능":
    st.write("같은 대상을 특성 구성별로 비교합니다. 2026년 1~8월이 평가 구간입니다.")
    model_tables(summary["models"])

elif view == "예측값":
    prediction_section(store)

else:
    from train import MODEL_DIR, TASKS, features

    @st.cache_resource
    def load_bundle(name):
        return joblib.load(MODEL_DIR / f"{name}.pkl")

    trained = [n for n in TASKS if (MODEL_DIR / f"{n}.pkl").exists()]
    if not trained:
        st.info("학습된 모델이 없습니다. `python train.py` 를 실행하세요.")
        st.stop()
    st.write("출발 전 일정 정보만 쓰는 모델입니다. 정확도가 높지 않아 경향을 보는 용도로만 쓰세요.")
    cats = load_bundle(trained[0])["cats"]
    with st.form("predict"):
        a, b = st.columns(2)
        airline = a.selectbox("항공사 코드", cats["Reporting_Airline"])
        date = b.date_input("운항일", datetime.date.today())
        origin = a.selectbox("출발 공항", cats["Origin"])
        dest = b.selectbox("도착 공항", cats["Dest"], index=min(1, len(cats["Dest"]) - 1))
        dep = a.time_input("예정 출발 시각", datetime.time(9, 0))
        arr = b.time_input("예정 도착 시각", datetime.time(12, 0))
        elapsed = a.number_input("예정 비행 시간(분)", 10, 1000, 180)
        distance = b.number_input("비행 거리(마일)", 10, 6000, 1000)
        submitted = st.form_submit_button("예측")
    if submitted:
        row = pd.DataFrame([{
            "Year": date.year, "Month": date.month, "DayOfWeek": date.isoweekday(),  # BTS: 1=월 … 7=일
            "Reporting_Airline": airline, "Origin": origin, "Dest": dest,
            "CRSDepTime": dep.hour * 100 + dep.minute, "CRSArrTime": arr.hour * 100 + arr.minute,
            "CRSElapsedTime": elapsed, "Distance": distance,
        }])
        results = []
        for n in trained:
            bnd = load_bundle(n)
            X = features(row, bnd["cats"])
            if bnd["kind"] == "regression":
                value = f"{bnd['model'].predict(X)[0]:+.0f}분"
            elif bnd["kind"] == "binary":
                p = bnd["model"].predict_proba(X)[0, 1]
                value = f"{p:.2f} → {'예' if p >= bnd['threshold'] else '아니오'}"
            else:
                proba = bnd["model"].predict_proba(X)[0]
                value = f"{bnd['model'].classes_[proba.argmax()]} ({proba.max():.2f})"
            results.append({"예측 대상": bnd["title"], "결과": value})
        st.dataframe(pd.DataFrame(results), hide_index=True, alt="모든 모델의 예측 결과")
        st.caption("이진 분류는 '점수 → 판정' 형식이며, class_weight='balanced' 모델의 점수는 실제 확률이 아닙니다.")

notes(summary)
