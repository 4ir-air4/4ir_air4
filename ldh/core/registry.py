"""ldh/categories/ 아래 폴더를 찾아 카테고리 목록을 만든다. 폴더를 추가하면 메뉴와 ldh/scripts/run_all.py 에 자동으로 나타난다."""
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 프로젝트 루트. 데이터 폴더(outputs, ldh/datasets/raw/reference/, data)는 여기에 있다
CATEGORY_DIR = ROOT / "ldh" / "categories"


def discover():
    """(카테고리 목록, 문제 있는 폴더 메시지 목록)을 돌려준다. 한 폴더의 오류가 앱 전체를 막지 않게 한다."""
    found, problems = [], []
    for path in sorted(p for p in CATEGORY_DIR.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))):
        try:
            meta = importlib.import_module(f"ldh.categories.{path.name}").META
            found.append({"slug": path.name, "page": path / "page.py", **meta})
        except Exception as e:
            problems.append(f"{path.name}: {e}")
    return sorted(found, key=lambda c: (c.get("order", 99), c["slug"])), problems
