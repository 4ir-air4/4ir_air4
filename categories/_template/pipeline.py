"""원본 데이터를 읽어 표와 요약을 outputs/<폴더이름>/ 에 저장한다. 실행: python run_all.py <폴더이름>"""
import pandas as pd


def run(store):
    yearly = pd.DataFrame({"연도": [2024, 2025], "건수": [100, 120]})  # 여기를 실제 집계로 바꾼다
    store.save_table("yearly", yearly)
    store.save_summary(
        headline="한 줄 요약을 적습니다.",
        kpis=[("2025년 건수", f"{yearly.건수.iloc[-1]:,}")],
        models=[],  # core.ml.model_row(...) 결과를 넣는다
        notes=["결과를 읽을 때 주의할 점을 적습니다."],
        sources=["데이터 이름"],
    )
