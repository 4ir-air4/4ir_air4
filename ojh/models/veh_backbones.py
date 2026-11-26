"""차량 종류 분류: 여러 그림 모델(특징 추출기)을 같은 정답·같은 채점으로 비교한다 (2026-11-25).

지금 쓰는 ConvNeXt-Tiny(ImageNet) 특징 + 로지스틱 회귀가 약 49~52%에서 멈춰서, 더 강한 모델을 시험한다.
- 차량 사진 + 주변 사진(빨간 상자) 두 장의 특징을 이어 붙인다 (veh_report.py 와 같은 방식)
- 채점: 5겹 교차 검증, 5개 미만 종류는 학습에만 (veh_report.py 와 같음)
- 모델마다 특징을 bb_feats/<이름>.npz 에 저장 → 다시 돌려도 이어서 함
실행: .venv/Scripts/python.exe ojh/models/veh_backbones.py [모델이름 ...]
"""
import csv
import json
import sys
import time
from collections import Counter

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from veh_report import DIR, ctx_img

torch.set_num_threads(4)
OUTD = DIR / "bb_feats"
OUTD.mkdir(exist_ok=True)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def tv(name, weights, strip):
    def make():
        m = getattr(torchvision.models, name)(weights=weights).eval()
        strip(m)
        return m
    return make


def hub(name):
    return lambda: torch.hub.load("facebookresearch/dinov2", name, trust_repo=True).eval()


def _id(attr, idx=None):
    def f(m):
        if idx is None:
            setattr(m, attr, torch.nn.Identity())
        else:
            getattr(m, attr)[idx] = torch.nn.Identity()
    return f


class YoloEmb(torch.nn.Module):
    """YOLO 를 특징 추출기로: 분류 모델은 마지막 판단 직전 1280개 값, 탐지 모델은 백본 끝(10번 층) 평균.
    입력은 ImageNet 정규화된 텐서 → YOLO 는 0~1 그대로 받으므로 되돌린다."""

    def __init__(self, path, cls):
        super().__init__()
        from ultralytics import YOLO
        self.m = YOLO(str(path)).model.float().eval()
        self.cls, self.f = cls, None
        if cls:
            self.m.model[-1].linear.register_forward_hook(lambda mod, i, o: setattr(self, "f", i[0]))
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = x * self.std + self.mean
        if self.cls:
            self.m(x)
            return self.f
        o = self.m.predict(x, embed=[10])
        return torch.stack(list(o)) if isinstance(o, (tuple, list)) else o


YDIR = DIR.parent   # ojh/models/saved/yolo
W = torchvision.models
MODELS = {
    "yolo26x_cls": lambda: YoloEmb(YDIR / "yolo26x-cls.pt", True),        # YOLO26 분류 가장 큰 것 (ImageNet)
    "apron_yolo26x_bb": lambda: YoloEmb(YDIR / "apron_cons_yolo26x.pt", False),   # 우리가 계류장으로 학습한 탐지 모델의 백본
   # 빠르고 기대되는 순서
    "vit_b16_swag": tv("vit_b_16", W.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1, _id("heads")),
    "regnet16_swag": tv("regnet_y_16gf", W.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_LINEAR_V1, _id("fc")),
    "convnext_base": tv("convnext_base", W.ConvNeXt_Base_Weights.IMAGENET1K_V1, _id("classifier", 2)),
    "effv2_m": tv("efficientnet_v2_m", W.EfficientNet_V2_M_Weights.IMAGENET1K_V1, _id("classifier", 1)),
    "dinov2_b": hub("dinov2_vitb14"),   # 사용자 허락으로 추가 (GitHub 코드 실행, 2026-11-25)
    "dinov2_l": hub("dinov2_vitl14"),
    "convnext_tiny": tv("convnext_tiny", W.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, _id("classifier", 2)),
}


def to_t(im):
    return TF.pil_to_tensor(im.convert("RGB").resize((224, 224), Image.BICUBIC))


def score(X, y):
    cnt = Counter(y)
    keep = np.array([cnt[t] >= 5 for t in y])
    idx = np.where(keep)[0]
    ok = np.zeros(len(y), bool)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X[keep], y[keep]):
        tri = np.concatenate([idx[tr], np.where(~keep)[0]])
        sc = StandardScaler().fit(X[tri])
        clf = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(sc.transform(X[tri]), y[tri])
        ok[idx[te]] = clf.predict(sc.transform(X[idx[te]])) == y[idx[te]]
    return float(ok[keep].mean())


def main():
    names = sys.argv[1:] or list(MODELS)
    crops = {r["crop"]: r for r in csv.DictReader(open(DIR / "crops.csv", encoding="utf-8"))}
    lab = {r["crop"]: r["type"] for r in csv.DictReader(open(DIR / "labels.csv", encoding="utf-8"))
           if r["type"] != "unknown" and r["crop"] in crops}
    cs = list(lab)
    y = np.array([lab[c] for c in cs])
    t0 = time.time()
    A, B = [], []
    for c in cs:   # 사진은 한 번만 읽어 둔다
        try:
            A.append(to_t(Image.open(DIR / "crops" / c)))
            B.append(to_t(ctx_img(crops[c])))
        except Exception:
            A.append(torch.zeros(3, 224, 224, dtype=torch.uint8))
            B.append(torch.zeros(3, 224, 224, dtype=torch.uint8))
    print(f"사진 {len(cs)}장 읽음 {time.time() - t0:.0f}초", flush=True)
    res_path = DIR / "bb_results.json"
    res = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
    for name in names:
        f = OUTD / f"{name}.npz"
        cache = {}
        if f.exists():
            z = np.load(f, allow_pickle=True)
            cache = dict(zip(z["names"], z["feats"]))
        todo = [i for i, c in enumerate(cs) if c not in cache]
        if todo:
            t0 = time.time()
            net = MODELS[name]()
            with torch.no_grad():
                for k in range(0, len(todo), 16):
                    b = todo[k:k + 16]
                    x = torch.stack([A[i] for i in b] + [B[i] for i in b]).float() / 255
                    x = TF.normalize(x, MEAN, STD)
                    o = net(x).reshape(len(b) * 2, -1).numpy()
                    for j, i in enumerate(b):
                        cache[cs[i]] = np.concatenate([o[j], o[len(b) + j]])
                    if (k // 16) % 10 == 9:   # 실제 진행 기록 (진행판이 읽는다)
                        print(f"진행 {name} {k + len(b)}/{len(todo)} {time.time() - t0:.0f}초", flush=True)
            np.savez(f, names=np.array(list(cache)), feats=np.stack(list(cache.values())))
            del net
            sec = time.time() - t0
        else:
            sec = 0
        acc = score(np.stack([cache[c] for c in cs]), y)
        res[name] = {"acc": acc, "labels": len(cs), "sec": round(sec)}
        res_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{time.strftime('%H:%M')} {name}: {acc:.1%} (정답 {len(cs)}, 특징 {sec:.0f}초)", flush=True)


if __name__ == "__main__":
    main()
