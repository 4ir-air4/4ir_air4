"""계류장 화면(AERO)용 차량 종류 맞히기 (2026-11-26).

YOLO 가 찾은 차량 상자 → 차량 사진(상자 30% 넓혀 자름) + 주변 사진(상자 4배, 빨간 상자) → 13종류 확률.
- 빠르게: 그림 모델 5개 특징 + 로지스틱 회귀 (combo_model.joblib, veh_report2.py 가 정답이 늘 때마다 다시 학습). 5겹 채점 약 60%
- 정확하게: 코랩 최종 EVA02-L CLIP(veh_eva02l_final.pt, 68.4%) 80% + 빠른 모델 20% 섞기. 5겹 채점 70.6%. CPU 에서 차량 1대 약 12초
학습 때(veh_more.py, veh_report.ctx_img)와 같은 방식으로 자른다. 파일은 git 에 없어서 없으면 쓸 수 없다고 알려 준다.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DIR = Path(__file__).resolve().parent / "saved" / "yolo" / "veh_types"
COMBO = DIR / "combo_model.joblib"
# 정답 1,479개로 이어 학습한 것(final2, 2026-11-26)이 있으면 그것을, 없으면 1,201개 것을 쓴다
FINAL = next((p for p in (DIR / "veh_eva02l_final2.pt", DIR / "veh_eva02l_final.pt") if p.exists()), DIR / "veh_eva02l_final.pt")
W_FINAL = 0.8   # veh_blend.py 에서 고른 비율 (EVA02-L CLIP 80% + 빠른 모델 20%)
NAMES = {"baggage_truck": "수하물 차량", "belt_loader": "벨트 로더", "bus": "버스", "car": "승용차",
         "catering_truck": "케이터링 트럭", "container_loader": "컨테이너 로더", "fuel_truck": "급유차",
         "gpu": "지상전원장치", "jet_bridge": "탑승교", "not_vehicle": "차량 아님", "stairs": "계단차",
         "truck": "트럭", "tug": "토잉카"}


def available():
    """{'fast': bool, 'accurate': bool}"""
    try:
        import timm  # noqa: F401
        has_timm = True
    except ImportError:
        has_timm = False
    return {"fast": COMBO.exists(), "accurate": COMBO.exists() and FINAL.exists() and has_timm}


def cut(image, box):
    """(차량 사진, 주변 사진). box = (x1, y1, x2, y2) 원래 탐지 상자."""
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    crop = image.crop((max(0, x1 - .3 * w), max(0, y1 - .3 * h), min(image.width, x2 + .3 * w), min(image.height, y2 + .3 * h)))
    crop.thumbnail((256, 256))
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    m = max(x2 - x1, y2 - y1) * 4
    ox, oy = max(0, x1 - m), max(0, y1 - m)
    ctx = image.crop((ox, oy, min(image.width, x2 + m), min(image.height, y2 + m))).copy()
    ImageDraw.Draw(ctx).rectangle((x1 - ox, y1 - oy, x2 - ox, y2 - oy), outline=(255, 0, 0), width=3)
    return crop.convert("RGB"), ctx.convert("RGB")


class Fast:
    """빠른 모델: combo_model.joblib 의 그림 모델 순서대로 특징을 이어 붙인다."""

    def __init__(self):
        import joblib
        import torch
        n = torch.get_num_threads()
        from veh_backbones import MODELS   # 불러오면서 스레드 수를 바꾸므로 되돌린다
        torch.set_num_threads(n)
        self.bundle = joblib.load(COMBO)
        self.nets = [MODELS[k]() for k in self.bundle["models"]]
        self.classes = list(self.bundle["clf"].classes_)
        self.acc = self.bundle.get("acc")
        self.labels = self.bundle.get("labels")

    def proba(self, pairs):
        import torch
        import torchvision.transforms.functional as TF
        from veh_backbones import MEAN, STD, to_t
        out = []
        with torch.no_grad():
            for k in range(0, len(pairs), 8):
                b = pairs[k:k + 8]
                x = TF.normalize(torch.stack([to_t(a) for a, _ in b] + [to_t(c) for _, c in b]).float() / 255, MEAN, STD)
                parts = []
                for net in self.nets:
                    o = net(x).reshape(len(b) * 2, -1).numpy()
                    parts.append(np.concatenate([o[:len(b)], o[len(b):]], 1))
                X = self.bundle["scaler"].transform(np.concatenate(parts, 1))
                out.append(self.bundle["clf"].predict_proba(X))
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)))


class Accurate:
    """정확한 모델: EVA02-L CLIP 확률 80% + 빠른 모델 확률 20% (종류 이름으로 맞춰 섞는다)."""

    def __init__(self, fast):
        from veh_final import load
        self.fast = fast
        self.model, self.tf, self.classes = load(FINAL)

    def proba(self, pairs):
        from veh_final import predict
        if not pairs:
            return np.zeros((0, len(self.classes)))
        pf = self.fast.proba(pairs)
        pf = np.stack([pf[:, self.fast.classes.index(c)] if c in self.fast.classes else np.zeros(len(pf))
                       for c in self.classes], 1)   # 빠른 모델에 없는 종류는 0
        pe = np.concatenate([predict(self.model, self.tf, [a for a, _ in pairs[k:k + 4]], [c for _, c in pairs[k:k + 4]])
                             for k in range(0, len(pairs), 4)])
        return W_FINAL * pe + (1 - W_FINAL) * pf
