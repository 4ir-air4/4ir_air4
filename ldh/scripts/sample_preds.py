"""git에 못 올리는 큰 건별 예측(pred_*.parquet)에서 화면용 무작위 표본(pred_sample_*.parquet)을 만든다.

미국 운항·기상(수백만 행)과 OpenSky(2천만 행)는 원본이 수백 MB라 드라이브로만 공유한다. 표본이 있으면 화면(ldh/core/store.py 의 pred_path)이
원본 대신 표본으로 미리보기와 모델 설명 곡선을 그린다. OpenSky 는 재배포 제한이 있어 노선·기종 열을 빼고 기간·실제·예측만 남긴다.
사용법: python -m ldh.scripts.sample_preds
"""
import numpy as np
import pyarrow.parquet as pq

from ldh.core.store import Store

SLUGS = ("us_flights", "us_weather", "world_traffic")
ROWS = 50_000
ONLY = {"world_traffic": ["기간", "실제", "예측"]}


def sample_rows(n, k, seed=0):
    """0..n-1 에서 k개를 뽑아 원래 순서로 돌려준다(미리보기가 시간 순으로 보이도록)."""
    return np.sort(np.random.default_rng(seed).choice(n, size=min(n, k), replace=False))


def run():
    for slug in SLUGS:
        store = Store(slug)
        for result in sorted((store.dir / "model_results").glob("*.json")):  # 결과가 있는 모델만 (예전 앙상블 찌꺼기 제외)
            src = store.table_path(f"pred_{result.stem}")
            if not src.exists():
                continue
            table = pq.read_table(src, columns=ONLY.get(slug))
            out = store.table_path(f"pred_sample_{result.stem}")
            pq.write_table(table.take(sample_rows(table.num_rows, ROWS)), out)
            print(f"{slug}/{out.name}: {table.num_rows:,} -> {min(table.num_rows, ROWS):,}행, {out.stat().st_size / 2**20:.1f}MB", flush=True)


if __name__ == "__main__":
    picked = sample_rows(10, 4)
    assert len(picked) == 4 and len(set(picked)) == 4 and (np.diff(picked) > 0).all(), "중복 없이 원래 순서"
    assert len(sample_rows(3, 10)) == 3, "행이 적으면 전부"
    run()
