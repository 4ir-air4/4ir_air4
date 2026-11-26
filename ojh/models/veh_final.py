"""코랩에서 정답 전체로 학습한 최종 EVA02-L CLIP 차량 종류 모델을 불러와 예측한다 (2026-11-25).

파일: veh_types/veh_eva02l_final.pt (fp16, {name, classes, img, mean, std, labels, state})
입력: 차량 사진 + 주변 사진(빨간 상자) → 종류별 확률. 코랩과 같이 좌우 뒤집기 평균.
점검: .venv/Scripts/python.exe ojh/models/veh_final.py  (정답 일부를 넣어 불러오기·예측이 되는지만 본다.
      학습에 쓴 정답이라 정확도는 높게 나오는 게 정상이다. 진짜 성적은 5겹 채점 68.4%)
"""
import csv
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from veh_report import DIR, ctx_img

PT = DIR / "veh_eva02l_final.pt"


class Two(nn.Module):
    def __init__(self, name, n):
        super().__init__()
        self.b = timm.create_model(name, pretrained=False, num_classes=0, img_size=224)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.b.num_features * 2, n))

    def forward(self, a, c):
        return self.head(torch.cat([self.b(a), self.b(c)], 1))


def load(path=PT):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = Two(ck["name"], len(ck["classes"]))
    m.load_state_dict({k: v.float() for k, v in ck["state"].items()})
    tf = T.Compose([T.Resize((ck["img"], ck["img"])), T.ToTensor(), T.Normalize(ck["mean"], ck["std"])])
    return m.eval(), tf, ck["classes"]


@torch.no_grad()
def predict(m, tf, crops, ctxs):
    a = torch.stack([tf(im.convert("RGB")) for im in crops])
    c = torch.stack([tf(im.convert("RGB")) for im in ctxs])
    p = m(a, c).softmax(1) + m(a.flip(3), c.flip(3)).softmax(1)
    return (p / 2).numpy()


def main():
    torch.set_num_threads(4)
    t0 = time.time()
    m, tf, classes = load()
    print("불러옴", f"{time.time() - t0:.0f}초", len(classes), "종류", flush=True)
    crops = {r["crop"]: r for r in csv.DictReader(open(DIR / "crops.csv", encoding="utf-8"))}
    lab = [(r["crop"], r["type"]) for r in csv.DictReader(open(DIR / "labels.csv", encoding="utf-8"))
           if r["type"] in classes and r["crop"] in crops][:16]
    t0 = time.time()
    P = np.concatenate([predict(m, tf, [Image.open(DIR / "crops" / c) for c, _ in lab[k:k + 8]],
                                [ctx_img(crops[c]) for c, _ in lab[k:k + 8]]) for k in range(0, len(lab), 8)])
    ok = np.mean([classes[i] == t for i, (_, t) in zip(P.argmax(1), lab)])
    print(f"정답 {len(lab)}개 예측 {time.time() - t0:.0f}초 (장당 {(time.time() - t0) / len(lab):.2f}초), 맞힘 {ok:.0%}")


if __name__ == "__main__":
    main()
