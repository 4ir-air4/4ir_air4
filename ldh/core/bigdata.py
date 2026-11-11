"""메모리에 다 올리기 어려운 데이터를 디스크 배열로 저장하고 읽는다."""
import json
from pathlib import Path

import numpy as np


def create(root, name, shape, dtype="float32"):
    Path(root).mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(Path(root) / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)


def update(root, name):
    return np.load(Path(root) / f"{name}.npy", mmap_mode="r+")


def read(root, name):
    return np.load(Path(root) / f"{name}.npy", mmap_mode="r")


def save_meta(root, meta):
    """모든 배열을 다 쓴 뒤 마지막에 호출한다. 이 파일이 있어야 완성된 캐시로 본다."""
    (Path(root) / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def load_meta(root):
    path = Path(root) / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
