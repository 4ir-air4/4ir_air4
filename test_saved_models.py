"""저장한 모델의 예측이 학습 직후 저장한 예측과 같은지 확인한다.  사용법: python test_saved_models.py [slug ...] (기본 korea_ops)"""
import importlib
import sys

import numpy as np
import pandas as pd

from categories.korea_ops.models import recent_delays
from core.models import MODELS, TEST, load_model
from core.store import Store

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
    print("이력 입력 점검 통과")
    for slug in sys.argv[1:] or ["korea_ops"]:
        store = Store(slug)
        for t in importlib.import_module(f"categories.{slug}.models").tasks():
            names = [m for m in MODELS if (store.dir / "saved_models" / f"{t['task']}__{m}" / "meta.json").exists()]
            if not names:
                continue
            data = t["load"]()
            for name in names:
                print(f"통과 {t['task']}/{name}: 최대 차이 {check_saved(store, t['task'], data, name):.6f}", flush=True)
