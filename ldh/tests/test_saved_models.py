"""저장한 모델의 예측이 학습 직후 저장한 예측과 같은지 확인한다.  사용법: python -m ldh.tests.test_saved_models [slug ...] [--year=2026] (기본 korea_ops)"""
import importlib
import sys

import numpy as np
import pandas as pd

from ldh.categories.korea_ops.models import CONTEXT_GROUPS, recent_context, recent_delays, window_rate
from ldh.core.models import MODELS, TEST, load_model, only_year
from ldh.core.store import OUTPUT_DIR, Store

TOLERANCE = 1e-3


def check_recent_delays():
    delay = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    prev = np.array([[-1, -1], [0, -1], [1, 0], [2, 1]])
    known_t = np.array([100, 300, 500, 900])  # 3번 행의 예측 시점(450)에 2번 행(500)의 결과는 아직 모른다
    pred_t = np.array([50, 200, 400, 450])
    result = recent_delays(delay, prev, known_t, pred_t)
    assert np.isnan(result[0, :3]).all() and result[0, 3] == 0, "이력이 없으면 값은 비고 건수는 0"
    assert result[1, 0] == 10 and result[1, 2] == 10 and result[1, 3] == 1, "알려진 직전 운항"
    assert result[2].tolist() == [20, 10, 15, 2], "최근 순, 평균, 건수"
    assert np.isnan(result[3, 0]) and result[3, 1] == 20 and result[3, 2] == 20 and result[3, 3] == 1, "예측 시점 이후에 끝난 운항은 쓰지 않음"


def check_window_rate():
    event_t, late = np.array([100, 200, 300]), np.array([1.0, 0.0, 1.0])
    rate, count = window_rate(event_t, late, np.array([200, 99, 300]), 100)  # 기간은 (query - 100, query]
    assert count.tolist() == [1, 0, 1] and np.isnan(rate[1]) and rate[0] == 0 and rate[2] == 1, "기간 경계와 이후 운항 제외"


def check_context_no_leak():
    """대상 편과 그 이후 운항의 결과를 바꿔도 대상 편의 입력은 그대로여야 한다."""
    times = pd.date_range("2025-01-01", periods=40, freq="30min")
    flights = pd.DataFrame({"sched": times, "delay": np.arange(40.0) % 30, "code": "NRT", "airline": "KE", "flight": "KE1", "direction": ["도착", "출발"] * 20})
    part = flights[flights.direction == "도착"].reset_index(drop=True)
    target = 15
    before, names = recent_context(flights, part, "도착", {"도착": 0.0, "출발": 0.0}, CONTEXT_GROUPS)
    changed = flights.copy()
    changed.loc[changed.sched >= part.sched[target], "delay"] = 500.0  # 대상 편부터 그 이후가 모두 큰 지연이 된다
    after, _ = recent_context(changed, part, "도착", {"도착": 0.0, "출발": 0.0}, CONTEXT_GROUPS)
    for name, a, b in zip(names, before, after):
        assert np.array_equal(a[target], b[target], equal_nan=True), f"{name}: 대상 편 이후 값이 입력에 들어감"
    early = flights.copy()
    early.loc[early.sched == part.sched[target], "delay"] = -200.0  # 일찍 끝난 기록은 자기 입력에 들어가면 안 된다
    self_in, _ = recent_context(early, part, "도착", {"도착": 0.0, "출발": 0.0}, CONTEXT_GROUPS)
    for name, a, b in zip(names, before, self_in):
        assert np.array_equal(a[target], b[target], equal_nan=True), f"{name}: 자기 기록이 입력에 들어감"


def check_saved(store, task, data, name):
    predict, meta = load_model(store, task, name)
    rows = np.flatnonzero(data.split == TEST)
    new = predict(data, rows)
    saved = pd.read_parquet(store.table_path(f"pred_{task}__{name}"))
    if meta["kind"] == "multiclass":
        assert (np.array(meta["classes"], dtype=object)[new.argmax(axis=1)] == saved["예측"].to_numpy()).all(), f"{task}/{name}: 예측 클래스가 다름"
        return 0.0
    gap = float(np.abs(new - saved["예측 점수" if meta["kind"] == "binary" else "예측"].to_numpy()).max())
    assert gap < TOLERANCE * max(1.0, float(np.abs(new).max())), f"{task}/{name}: 최대 차이 {gap}"
    return gap


if __name__ == "__main__":
    check_recent_delays()
    check_window_rate()
    check_context_no_leak()
    print("이력 입력 점검 통과")
    year = next((int(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--year=")), None)
    for slug in [a for a in sys.argv[1:] if not a.startswith("--")] or ["korea_ops"]:
        store = Store(slug)
        if year:  # ldh/scripts/run_models.py --year 와 같은 위치와 같은 행
            store.dir = OUTPUT_DIR.parent / str(year) / slug
        for t in importlib.import_module(f"ldh.categories.{slug}.models").tasks():
            names = [m for m in MODELS if (store.dir / "saved_models" / f"{t['task']}__{m}" / "meta.json").exists()]
            if not names:
                continue
            data = t["load"]()
            if year:
                data = only_year(data, year)
            for name in names:
                print(f"통과 {t['task']}/{name}: 최대 차이 {check_saved(store, t['task'], data, name):.6f}", flush=True)
