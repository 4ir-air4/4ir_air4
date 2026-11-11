"""전체 요약 화면. 카테고리별 summary.json 을 모아서 보여준다."""
import streamlit as st

from ldh.core.registry import ROOT, discover
from ldh.core.store import Store
from ldh.core.ui import model_tables

categories, problems = discover()
for p in problems:
    st.warning(f"카테고리를 읽지 못했습니다. {p}")

st.write("카테고리를 고르면 통계와 모델 결과를 볼 수 있습니다. 모든 모델은 시간순으로 나눠 미래 구간으로 평가했습니다.")
summaries = {c["slug"]: Store(c["slug"]).summary() for c in categories}

for start in range(0, len(categories), 2):
    for col, c in zip(st.columns(2), categories[start:start + 2]):
        with col.container(border=True, height="stretch"):
            st.page_link(c["page"].relative_to(ROOT).as_posix(), label=f"**{c['title']}**", icon=c["icon"])
            s = summaries[c["slug"]]
            if s:
                st.write(s["headline"])
                st.caption(f"모델 {len(s['models'])}개 · 통계 갱신 {s['updated']}")
            else:
                st.caption(f"결과 없음 · `python -m ldh.scripts.run_all {c['slug']}`")

st.subheader("전체 모델 성능")
rows = [{"category": c["title"], **m} for c in categories if summaries[c["slug"]] for m in summaries[c["slug"]]["models"]]
model_tables(rows, show_category=True)
