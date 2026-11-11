"""2026년 자료로만 학습한 결과 화면. python run_models.py --year=2026 의 결과(outputs_2026/)를 보여 준다."""
import streamlit as st

from core.registry import discover
from core.store import OUTPUT_DIR, Store
from core.ui import prediction_section

ROOT = OUTPUT_DIR.parent / "outputs_2026"

categories = {c["slug"]: c["title"] for c in discover()[0] if any((ROOT / c["slug"] / "model_results").glob("*.json"))}
st.caption("2026년 행으로만 학습한 구동 확인 결과입니다. 앞쪽 달로 학습하고 마지막 25% 달로 평가하므로 계절이 달라, 최종 성능으로 읽으면 안 됩니다.")
if not categories:
    st.info("아직 결과가 없습니다. 코랩에서 `run_models.py --year=2026` 을 실행한 뒤 드라이브의 `outputs_2026` 을 이 프로젝트로 복사하세요.", icon=":material/hourglass_empty:")
    st.stop()
slug = st.selectbox("카테고리", list(categories), format_func=categories.get)
store = Store(slug)
store.dir = ROOT / slug
prediction_section(store)
