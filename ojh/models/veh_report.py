"""고른 정답으로 계속 학습하고, 종류별로 얼마나 맞히는지·정답이 얼마나 더 필요한지 알려 준다.

- 정답이 30개 이상 늘 때마다(10분마다 확인) 다시 학습한다.
- 방법: 사전 학습 ConvNeXt 로 차량+주변 사진의 특징을 뽑고 로지스틱 회귀 (비교 시험에서 가장 나았음, 49.7%).
- 채점: 5겹 교차 검증 (고른 정답끼리 돌아가며 시험).
- 결과 veh_types/class_report.json → 고르기 화면 '학습 현황' 표, veh_priority.py 가 '먼저 고를 종류'로 씀.
실행: .venv/Scripts/python.exe ojh/models/veh_report.py
"""
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DIR = Path(__file__).resolve().parent / "saved" / "yolo" / "veh_types"
FEAT = DIR / "report_feats.npz"
OUT = DIR / "class_report.json"
TARGET = 0.7          # 종류마다 목표 정확도
torch.set_num_threads(2)   # 사진 받기·다시 보기와 CPU 를 나눠 쓴다


def ctx_img(r):
    photo = Image.open(DIR / "photos" / r["photo"]).convert("RGB")
    x1, y1, x2, y2 = (int(float(r[k])) for k in ("x1", "y1", "x2", "y2"))
    m = max(x2 - x1, y2 - y1) * 4
    ox, oy = max(0, x1 - m), max(0, y1 - m)
    im = photo.crop((ox, oy, min(photo.width, x2 + m), min(photo.height, y2 + m))).copy()
    ImageDraw.Draw(im).rectangle((x1 - ox, y1 - oy, x2 - ox, y2 - oy), outline=(255, 0, 0), width=3)
    return im


def main():
    w = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    net = torchvision.models.convnext_tiny(weights=w).eval()
    net.classifier[2] = torch.nn.Identity()
    tf = w.transforms()
    cache = {}
    if FEAT.exists():
        z = np.load(FEAT, allow_pickle=True)
        cache = dict(zip(z["names"], z["feats"]))
    last_n = -999
    while True:
        crops = {r["crop"]: r for r in csv.DictReader(open(DIR / "crops.csv", encoding="utf-8"))}
        labels = {r["crop"]: r["type"] for r in csv.DictReader(open(DIR / "labels.csv", encoding="utf-8"))
                  if r["type"] != "unknown" and r["crop"] in crops}
        if len(labels) - last_n < 30:
            time.sleep(600)
            continue
        new = [c for c in labels if c not in cache]
        with torch.no_grad():
            for i in range(0, len(new), 16):
                b = new[i:i + 16]
                ims_a, ims_b = [], []
                for c in b:
                    try:
                        ims_a.append(tf(Image.open(DIR / "crops" / c).convert("RGB")))
                        ims_b.append(tf(ctx_img(crops[c])))
                    except Exception:
                        ims_a.append(torch.zeros(3, 224, 224))
                        ims_b.append(torch.zeros(3, 224, 224))
                fa, fb = net(torch.stack(ims_a)).numpy(), net(torch.stack(ims_b)).numpy()
                for c, x, y in zip(b, fa, fb):
                    cache[c] = np.concatenate([x, y])
        np.savez(FEAT, names=np.array(list(cache)), feats=np.stack(list(cache.values())))
        names = list(labels)
        X = np.stack([cache[c] for c in names])
        y = np.array([labels[c] for c in names])
        cnt = Counter(y)
        keep = np.array([cnt[t] >= 5 for t in y])
        idx = np.where(keep)[0]
        ok = np.zeros(len(y), bool)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X[keep], y[keep]):
            tri = np.concatenate([idx[tr], np.where(~keep)[0]])
            sc = StandardScaler().fit(X[tri])
            clf = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(sc.transform(X[tri]), y[tri])
            ok[idx[te]] = clf.predict(sc.transform(X[idx[te]])) == y[idx[te]]
        classes = {}
        for t, n in cnt.items():
            m = keep & (y == t)
            acc = float(ok[m].mean()) if m.any() else None
            # 정답이 2배가 될 때마다 약 5%p 오른다는 전체 추세로 대략 계산 (종류마다 다를 수 있는 추정치)
            if acc is None:
                more = max(5 - n, 0) + 30
            elif acc >= TARGET:
                more = 0
            else:
                more = int(n * (2 ** ((TARGET - acc) / 0.05) - 1))
            classes[t] = {"n": n, "acc": acc, "more": min(more, 5000)}
        overall = float(ok[keep].mean()) if keep.any() else None
        # 모든 정답으로 다시 학습해 저장 → 고르기 화면의 AI 추측이 쓴다 (2026-11-25)
        import joblib
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
        tmp = DIR / "report_model.tmp"
        joblib.dump({"scaler": sc, "clf": clf, "labels": len(labels), "acc": overall}, tmp)
        tmp.replace(DIR / "report_model.joblib")
        OUT.write_text(json.dumps({"updated": time.time(), "labels": len(labels), "overall": overall, "target": TARGET,
                                   "classes": classes}, ensure_ascii=False), encoding="utf-8")
        print(time.strftime("%H:%M"), "정답", len(labels), "전체", f"{overall:.1%}" if overall else "-", flush=True)
        last_n = len(labels)
        time.sleep(600)


if __name__ == "__main__":
    main()
