"""카테고리별 결과 저장소. pipeline.py 가 쓰고 page.py 가 읽는다. 화면은 원본 데이터 없이 ldh/results/full/ 만으로 동작한다."""
import datetime
import json
from pathlib import Path

import pandas as pd

from ldh.core.registry import ROOT

OUTPUT_DIR = ROOT / "ldh" / "results" / "full"


class Store:
    def __init__(self, slug):
        self.slug = slug
        self.dir = OUTPUT_DIR / slug

    def _replace(self, path, write):
        """중간에 끊겨도 이전 결과가 깨지지 않도록 임시 파일에 쓴 뒤 교체한다."""
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        write(tmp)
        tmp.replace(path)

    def save_table(self, name, df):
        self._replace(self.dir / f"{name}.parquet", lambda p: df.to_parquet(p, index=False))

    def table_path(self, name):
        return self.dir / f"{name}.parquet"

    def table(self, name):
        return pd.read_parquet(self.table_path(name))

    def save_summary(self, headline, kpis=(), models=(), notes=(), sources=()):
        """headline: 한 줄 요약 / kpis: [(이름, 값, 도움말)] / models: core.ml 결과 행 / notes: 주의 사항 / sources: 데이터 출처."""
        body = {"headline": headline, "kpis": [{"label": k[0], "value": k[1], "help": k[2] if len(k) > 2 else None} for k in kpis],
                "models": list(models), "notes": list(notes), "sources": list(sources),
                "updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
        text = json.dumps(body, indent=2, ensure_ascii=False, default=float)
        self._replace(self.dir / "summary.json", lambda p: Path(p).write_text(text, encoding="utf-8"))

    def summary(self):
        path = self.dir / "summary.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
