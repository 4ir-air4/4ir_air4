"""활주로·계류장 혼잡도: 공항 CCTV·사진에서 YOLO 로 비행기·차량·사람을 세어 혼잡 등급을 매긴다.

사진 종류에 따라 다른 모델을 쓴다 (둘 다 Ultralytics 사전 학습 모델, 추가 학습 없음):
  - 지상·CCTV 사진 : YOLO26 (COCO 80종) - 비행기·버스·트럭·승용차·사람. 크기 x/m/s 중 선택
  - 위성·항공사진  : YOLO26x-OBB (DOTA 항공사진 15종) - 비행기·헬리콥터·대형/소형 차량을 회전 박스로 찾는다
  2026-11-18 비교: 인천 전경 사진에서 YOLO11m 13대 → YOLO26x 27대(대부분 실제 비행기), 뮌헨은 둘 다 실제 7대 근처.
넓은 사진은 겹치게 조각(타일)내서 따로 찾은 뒤 합쳐, 멀리 있는 작은 비행기·차량도 놓치지 않게 한다.
가중치는 처음 쓸 때 ojh/models/saved/yolo/ 에 내려받는다 (git 에는 올라가지 않음).
YOLO 는 '주기장'을 못 찾으므로, 사진에 보이는 자리 수(빈자리 포함)·장소는 버튼을 누르면 Gemini 에게 묻고(ai_stands)
'서 있는 비행기 ÷ 보이는 자리'(점유율)로 등급을 매긴다.
"""
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT_DIR / "models" / "saved" / "yolo"

GROUND_MODELS = {"가장 정확하게 (YOLO26x)": "yolo26x.pt", "균형 (YOLO26m)": "yolo26m.pt", "빠르게 (YOLO26s)": "yolo26s.pt"}
# 계류장 학습 모델: YOLO26x 를 공개 계류장 데이터(Roboflow Handling·airport_90, 516장)로 추가 학습 (ojh/models/train_apron_yolo.ipynb).
# 2026-11-18 Colab A100, 48번째 반복이 최고. 테스트 mAP50 0.798 (비행기 0.99, 벨트 로더·급유차·탑승교 0.99, 토잉카 0.89, 승용차 0).
# 가중치(118MB)는 GitHub 한도를 넘어 git 에 없다. 파일이 있을 때만 메뉴에 나온다.
# 학습 사진이 가까이서 찍은 CCTV·지상 사진이라, 높은 곳에서 찍은 공항 전경(뮌헨·인천)에서는 비행기를 7대 중 1대만 찾고
# 들판을 수하물 트럭으로 잡았다. 그래서 비행기·사람·승용차·버스는 COCO 모델(yolo26x)이 찾고, 학습 모델은 장비만,
# 그것도 신뢰도 APRON_MIN_CONF 이상만 쓴다.
APRON_MODEL = "apron_yolo26x.pt"
APRON_LABEL = "지상조업 장비까지 (CCTV·근접 사진용, 계류장 학습)"
APRON_MIN_CONF = 0.4
APRON_CLASSES = {"baggage_truck": ("수하물 트럭", "vehicle"),
                 "belt_loader": ("벨트 로더", "vehicle"), "fuel_truck": ("급유차", "vehicle"), "tug": ("토잉카", "vehicle"),
                 "stairs": ("계단차", "vehicle"), "catering_truck": ("케이터링 트럭", "vehicle"),
                 "container_loader": ("컨테이너 로더", "vehicle"), "gpu": ("지상전원장치", "vehicle"), "truck": ("트럭", "vehicle"),
                 "jet_bridge": ("탑승교", "facility")}
APRON_COCO = {4: ("비행기", "aircraft"), 0: ("사람", "person"), 2: ("승용차", "vehicle"), 3: ("오토바이", "vehicle"),
              5: ("버스", "vehicle")}
# 넓은 공항 전경 모델: 위키미디어 공항 사진 338장을 Grounding DINO·OWLv2·Gemini·COCO YOLO 가 각자 표시하고 서로 다른 모델 2개 이상이
# 같은 자리에 동의한 박스만 정답으로 써서(차량은 풀밭·너무 큰 박스 제거) 계류장 CCTV 사진과 합쳐(학습 617·검증 116) YOLO26x 를 다시 학습
# (ojh/models/consensus_apron.ipynb, 2026-11-19 Colab L4, 검증 mAP50 0.543). 한 모델만 표시한 지난번(apron_auto, mAP50 0.508)보다
# 뮌헨 비행기 10 -> 7대(정답 7), 지상 차량 69 -> 19대로 들판·건물 오탐이 줄었고, 인천은 먼 작은 비행기를 더 찾는다(50 -> 79).
# 탑승교는 건물 덩어리를 크게 잡는 일이 남아 쓰지 않고, 사람은 약해서(검증 mAP50-95 0.10) COCO 모델로 찾는다. 가중치는 git 에 없다.
AUTO_MODEL = "apron_cons_yolo26x.pt"
AUTO_LABEL = "넓은 공항 전경용 (여러 모델 합의 데이터로 추가 학습)"
AUTO_CLASSES = {0: ("비행기", "aircraft"), 2: ("지상 차량", "vehicle")}
AERIAL_MODELS = {"가장 정확하게 (YOLO26x-OBB)": "yolo26x-obb.pt", "빠르게 (YOLO11m-OBB)": "yolo11m-obb.pt"}
PHOTO_TYPES = ["지상·CCTV 사진", "위성·항공사진 (위에서 내려다본 사진)"]

# COCO 클래스 번호 -> (화면 이름, 묶음)
COCO_CLASSES = {4: ("비행기", "aircraft"), 2: ("승용차", "vehicle"), 5: ("버스", "vehicle"), 7: ("트럭·지상조업차", "vehicle"),
                3: ("오토바이", "vehicle"), 0: ("사람", "person")}
# DOTA 클래스 이름 -> (화면 이름, 묶음)
DOTA_CLASSES = {"plane": ("비행기", "aircraft"), "helicopter": ("헬리콥터", "aircraft"),
                "large vehicle": ("대형 차량", "vehicle"), "small vehicle": ("소형 차량", "vehicle")}
GROUP_LABELS = {"aircraft": "✈️ 비행기", "vehicle": "🚌 차량", "person": "🧍 사람"}
GROUP_INDEX = {"aircraft": 0, "vehicle": 1, "person": 2, "facility": 3}   # facility(탑승교)는 표에만 보이고 점수에는 안 넣는다
DEFAULT_WEIGHTS = {"aircraft": 5.0, "vehicle": 1.0, "person": 0.5}
BOX_COLS = ["x1", "y1", "x2", "y2"]   # 탐지 상자 (줄인 사진 기준). 차량 종류 맞히기에 쓴다
GRADES = [(10, "여유", "🟢"), (25, "보통", "🟡"), (45, "혼잡", "🟠"), (float("inf"), "매우 혼잡", "🔴")]
# 주기장 수를 알 때: 비행기 수 ÷ 주기장 수(점유율)로 등급을 매긴다. 공항 크기와 상관없이 '얼마나 찼나'를 본다
OCC_GRADES = [(0.4, "여유", "🟢"), (0.7, "보통", "🟡"), (0.9, "혼잡", "🟠"), (float("inf"), "매우 혼잡", "🔴")]
BUSY_VEHICLES_PER_AIRCRAFT = 4   # 비행기 1대당 차량이 이보다 많으면 '지상조업 활발'로 표시
TILE, OVERLAP = 640, 0.25
# 조각 방식: (조각 크기, 조각 분석 해상도). 세밀: 작게 잘라 2배로 키워 찾는다 (게이트에 붙은 작은 비행기·차량용, 약 5배 느림)
TILE_MODES = {"보통": (640, 640), "세밀하게 (느림)": (480, 960), "끄기": None}


# 사진 속 주기장(빈자리 포함) 수·장소는 YOLO 가 못 찾으므로 Gemini(사진 이해)에게 묻는다.
# 2026-11 테스트(하와이·뮌헨·두바이)에서 3.8-flash 가 가장 정확했다. 무료 등급은 flash 가 하루 20번이라
# 사용량 초과·혼잡(429/503)이면 다음 모델로 넘어간다. lite 는 하루 500번이지만 빽빽한 사진에서 비행기를 많이 센다.
AI_STAND_MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite"]
AI_LOW_ACCURACY = {"gemini-3.1-flash-lite"}
CONF_KO = {"high": "높음", "medium": "보통", "low": "낮음"}
AI_STAND_PROMPT = """You are an airport apron analyst. Look at this photo of an airport.
Count ONLY what is visible in this photo (not the whole airport).

1. aircraft_parked: aircraft standing still at a parking stand (contact gate with jet bridge OR remote stand).
   Exclude aircraft moving on runways/taxiways.
2. stands_occupied: parking stands with an aircraft on them.
3. stands_empty: parking stands that are clearly visible and EMPTY. Recognise them by jet bridges with no aircraft,
   painted lead-in lines / stop bars / stand numbers, or a gap in a row of parked aircraft. Do not guess stands you cannot see.
4. stands_total = stands_occupied + stands_empty.
5. location: which airport (and terminal/concourse if recognisable) this is, in Korean. "알 수 없음" if unsure.
6. location_confidence: "high" | "medium" | "low".
7. airport_total_stands: total number of aircraft parking stands of that WHOLE airport (contact + remote) from your knowledge,
   or null if the airport is unknown.

Return JSON: {"aircraft_parked": int, "stands_occupied": int, "stands_empty": int, "stands_total": int,
"confidence": "high"|"medium"|"low", "reason": "one short sentence in Korean",
"location": str, "location_confidence": "high"|"medium"|"low", "airport_total_stands": int|null}"""


def _gemini_key():
    from _4_ai_chat_tab import _secret   # AI 도우미와 같은 키 읽기 (st.secrets → secrets.toml → 환경변수)
    return _secret("GEMINI_API_KEY") or _secret("GOOGLE_API_KEY")


@st.cache_data(max_entries=200, persist="disk", show_spinner=False)
def ai_stands(data):
    """Gemini 로 사진 속 주기장 수·장소를 추정한다. 같은 사진은 디스크에 저장된 결과를 써서 하루 한도를 아낀다.
    모든 모델이 실패하면 예외를 던진다(실패는 저장되지 않음)."""
    import io
    import json
    import time

    from google import genai
    from google.genai import errors, types
    from PIL import Image

    key = _gemini_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY 가 없습니다. .streamlit/secrets.toml 에 넣어주세요.")
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((1568, 1568))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    client = genai.Client(api_key=key)
    failed = []
    for model in AI_STAND_MODELS:
        start = time.time()
        try:
            r = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"), AI_STAND_PROMPT],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
            text = r.text.strip()
            res = json.loads(text[text.find("{"):text.rfind("}") + 1])   # lite 는 [ {...} ] 로 감쌀 때가 있다
            res = {k: res.get(k) for k in ("aircraft_parked", "stands_occupied", "stands_empty", "stands_total", "confidence",
                                           "reason", "location", "location_confidence", "airport_total_stands")}
            for k in ("aircraft_parked", "stands_occupied", "stands_empty", "stands_total"):
                res[k] = max(int(res[k] or 0), 0)
            res["stands_total"] = max(res["stands_total"], res["stands_occupied"] + res["stands_empty"])
            res.update(model=model, seconds=round(time.time() - start, 1), skipped=failed,
                       tokens=getattr(r.usage_metadata, "total_token_count", None))
            return res
        except (errors.APIError, ValueError, TypeError, KeyError) as e:
            failed.append(f"{model}: {getattr(e, 'code', '') or type(e).__name__}")
    raise RuntimeError("모든 Gemini 모델이 실패했습니다 (" + ", ".join(failed) + "). 하루 한도를 다 썼거나 사용량이 많은 시간일 수 있습니다.")


@st.cache_resource(show_spinner="YOLO 모델을 불러오는 중... (처음 한 번은 내려받기 때문에 조금 걸립니다)")
def load_model(name):
    from ultralytics import YOLO

    path = WEIGHTS_DIR / name
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(str(path))
    return YOLO(str(path))


def grade(score, table=GRADES):
    for limit, name, icon in table:
        if score < limit:
            return name, icon
    return table[-1][1:]


def judge(counts, weights, stands, parked=None):
    """(등급 이름, 아이콘, 비교용 값, 화면에 보일 설명).
    stands(사진에 보이는 주기장 수)가 있으면 점유율로, 없으면 가중치 점수로 판단한다.
    parked: 주기장에 서 있는 비행기 수(AI 추정). 없으면 YOLO 가 찾은 비행기 수를 쓴다(유도로의 비행기도 포함될 수 있음)."""
    if stands:
        planes = counts["aircraft"] if parked is None else parked
        occ = planes / stands
        name, icon = grade(occ, OCC_GRADES)
        return name, icon, occ, f"주기장 점유율 {occ:.0%} (비행기 {planes} / 주기장 {stands})"
    score = score_of(counts, weights)
    name, icon = grade(score)
    return name, icon, score, f"점수 {score:.1f}"


def tiles_of(image, tile=TILE, overlap=OVERLAP):
    """겹치게 자른 조각 [(x0, y0, 조각 이미지)]. 사진이 조각보다 작으면 빈 목록"""
    w, h = image.size
    if max(w, h) <= tile * 1.25:
        return []
    step = int(tile * (1 - overlap))
    xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
    ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
    if xs[-1] + tile < w:
        xs.append(w - tile)
    if ys[-1] + tile < h:
        ys.append(h - tile)
    return [(x, y, image.crop((x, y, min(x + tile, w), min(y + tile, h)))) for y in ys for x in xs]


MAX_SIDE = 2048   # 이보다 큰 사진은 줄여서 분석 (속도. 작은 물체는 조각내기가 보완)


def _shrink(image):
    w, h = image.size
    if max(w, h) <= MAX_SIDE:
        return image
    s = MAX_SIDE / max(w, h)
    return image.resize((round(w * s), round(h * s)))


def _tile_boxes(boxes, x0, y0, tw, th, w, h, margin=4):
    """조각 결과 정리: 조각 안쪽 테두리에 걸쳐 잘린 박스, 조각을 거의 다 덮는 거대 박스는 버리고 전체 사진 좌표로 옮긴다.
    (잘린 물체는 전체 사진 탐지나 옆 조각이 온전하게 잡는다)"""
    if not len(boxes):
        return boxes
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cut = ((x1 <= margin) & (x0 > 0)) | ((y1 <= margin) & (y0 > 0)) |           ((x2 >= tw - margin) & (x0 + tw < w)) | ((y2 >= th - margin) & (y0 + th < h))
    huge = ((x2 - x1) * (y2 - y1)) > 0.6 * tw * th
    out = boxes[~(cut | huge)].clone()
    out[:, [0, 2]] += x0
    out[:, [1, 3]] += y0
    return out


def _drop_contained(xyxy, conf, groups, ratio=0.6):
    """같은 묶음에서 더 확실한 박스 안에 대부분(ratio 이상) 들어가는 박스를 지운다 (비행기 일부만 잡은 박스 등). 남길 번호를 돌려준다"""
    import torch

    order = torch.argsort(conf, descending=True)
    keep = []
    area = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=0) * (xyxy[:, 3] - xyxy[:, 1]).clamp(min=0)
    for i in order.tolist():
        inside = False
        for j in keep:
            if groups[i] != groups[j]:
                continue
            iw = (torch.min(xyxy[i, 2], xyxy[j, 2]) - torch.max(xyxy[i, 0], xyxy[j, 0])).clamp(min=0)
            ih = (torch.min(xyxy[i, 3], xyxy[j, 3]) - torch.max(xyxy[i, 1], xyxy[j, 1])).clamp(min=0)
            if iw * ih >= ratio * min(area[i], area[j]):
                inside = True
                break
        if not inside:
            keep.append(i)
    return torch.tensor(keep, dtype=torch.long)


def _group_ids(cls, mapping):
    import torch
    return torch.tensor([GROUP_INDEX[mapping(int(c))[1]] for c in cls.tolist()], dtype=torch.long)


def _ground_boxes(model, image, conf, tile_mode, classes):
    """한 모델로 전체 사진(640·1280) + 조각별 탐지. (박스 [x1,y1,x2,y2,신뢰도,클래스], 전체 사진 결과, 조각 수)"""
    import torch

    w, h = image.size
    whole = [model.predict(image, conf=conf, imgsz=s, classes=classes, verbose=False)[0] for s in (640, 1280)]
    datas = [r.boxes.data for r in whole]
    tile_cfg = TILE_MODES.get(tile_mode)
    pieces = tiles_of(image, tile=tile_cfg[0]) if tile_cfg else []
    if pieces:
        for (x0, y0, piece), r in zip(pieces, model.predict([p for _, _, p in pieces], conf=conf, imgsz=tile_cfg[1], classes=classes, verbose=False)):
            datas.append(_tile_boxes(r.boxes.data, x0, y0, piece.width, piece.height, w, h))
    return torch.cat(datas), whole[0], len(pieces)


def detect_ground(models, image, conf, tile_mode="보통"):
    """지상·CCTV 사진. models: [(YOLO 모델, {클래스 번호: (화면 이름, 묶음)}, 최소 신뢰도)] - 여러 모델의 결과를 합친다.
    (박스 그린 RGB 이미지, 탐지 표, 조각 수)"""
    import torch
    from torchvision.ops import batched_nms

    image = _shrink(image)
    labels, en_names, datas, base, n_tiles = [], [], [], None, 0
    for model, cmap, min_conf in models:
        data, r, n_tiles = _ground_boxes(model, image, max(conf, min_conf), tile_mode, list(cmap))
        if base is None:
            base = r
        ids = {}
        for c, lab in cmap.items():   # 모델마다 다른 클래스 번호를 하나의 번호 체계로 바꾼다
            if lab not in labels:
                labels.append(lab)
                en_names.append(model.names[c])
            ids[c] = labels.index(lab)
        if len(data):
            data = data.clone()
            data[:, 5] = torch.tensor([ids[int(c)] for c in data[:, 5].tolist()], dtype=data.dtype)
        datas.append(data)
    data = torch.cat(datas)
    if len(data):
        # 같은 물체를 승용차/트럭처럼 다른 이름으로 겹쳐 잡은 경우도 하나만 남기도록 '묶음' 단위로 겹침 제거
        groups = _group_ids(data[:, 5], labels.__getitem__)
        keep = batched_nms(data[:, :4], data[:, 4], groups, iou_threshold=0.45)
        data, groups = data[keep], groups[keep]
        data = data[_drop_contained(data[:, :4], data[:, 4], groups)]
    base.names = dict(enumerate(en_names))   # 그림 글씨는 영어 (기본 글꼴에 한글이 없음)
    base.update(boxes=data)
    rows = [{"종류": labels[int(c)][0], "묶음": labels[int(c)][1], "신뢰도": round(s, 2), **dict(zip(BOX_COLS, b))}
            for c, s, b in zip(base.boxes.cls.tolist(), base.boxes.conf.tolist(), base.boxes.xyxy.tolist())]
    return base.plot(line_width=2, font_size=12)[:, :, ::-1], pd.DataFrame(rows, columns=["종류", "묶음", "신뢰도", *BOX_COLS]), n_tiles


def ground_models(model_name):
    """화면에서 고른 모델 이름 -> detect_ground 에 넘길 [(모델, 클래스 표)]"""
    if model_name == AUTO_MODEL:
        return [(load_model(AUTO_MODEL), AUTO_CLASSES, 0), (load_model("yolo26x.pt"), {0: ("사람", "person")}, 0)]
    if model_name != APRON_MODEL:
        return [(load_model(model_name), COCO_CLASSES, 0)]
    apron = load_model(APRON_MODEL)
    return [(load_model("yolo26x.pt"), APRON_COCO, 0),
            (apron, {i: APRON_CLASSES[n] for i, n in apron.names.items() if n in APRON_CLASSES}, APRON_MIN_CONF)]


def detect_aerial(model, image, conf, tile_mode="보통"):
    """위성·항공사진: OBB 모델로 전체 + 조각별 탐지를 합친다 (회전 박스는 바깥 사각형 기준으로 겹침 제거)."""
    import torch
    from torchvision.ops import batched_nms

    image = _shrink(image)
    w, h = image.size
    names = model.names
    wanted = [i for i, n in names.items() if n in DOTA_CLASSES]
    mapping = lambda c: DOTA_CLASSES[names[c]]   # noqa: E731
    whole = model.predict(image, conf=conf, imgsz=1024, classes=wanted, verbose=False)[0]
    datas, boxes = [whole.obb.data], [whole.obb.xyxy]
    pieces = tiles_of(image, tile=1024 if tile_mode != "세밀하게 (느림)" else 768) if TILE_MODES.get(tile_mode) else []
    if pieces:
        for (x0, y0, piece), r in zip(pieces, model.predict([p for _, _, p in pieces], conf=conf, imgsz=1024, classes=wanted, verbose=False)):
            b = r.obb.xyxy
            if not len(b):
                continue
            pw, ph, m = piece.width, piece.height, 4
            cut = ((b[:, 0] <= m) & (x0 > 0)) | ((b[:, 1] <= m) & (y0 > 0)) | ((b[:, 2] >= pw - m) & (x0 + pw < w)) | ((b[:, 3] >= ph - m) & (y0 + ph < h))
            d, b = r.obb.data[~cut].clone(), b[~cut].clone()
            d[:, 0] += x0
            d[:, 1] += y0
            b[:, [0, 2]] += x0
            b[:, [1, 3]] += y0
            datas.append(d)
            boxes.append(b)
    data, xyxy = torch.cat(datas), torch.cat(boxes)
    if len(data):
        groups = _group_ids(data[:, 6], mapping)
        keep = batched_nms(xyxy, data[:, 5], groups, iou_threshold=0.3)
        data, xyxy, groups = data[keep], xyxy[keep], groups[keep]
        data = data[_drop_contained(xyxy, data[:, 5], groups)]
    whole.update(obb=data)
    rows = [{"종류": mapping(int(c))[0], "묶음": mapping(int(c))[1], "신뢰도": round(s, 2), **dict(zip(BOX_COLS, b))}
            for c, s, b in zip(whole.obb.cls.tolist(), whole.obb.conf.tolist(), whole.obb.xyxy.tolist())]
    return whole.plot(line_width=2, font_size=12)[:, :, ::-1], pd.DataFrame(rows, columns=["종류", "묶음", "신뢰도", *BOX_COLS]), len(pieces)


@st.cache_data(max_entries=24, show_spinner=False)
def detect_cached(data, model_name, aerial, conf, tile_mode):
    """사진 내용·설정이 같으면 저장된 탐지 결과를 돌려준다. (박스 그린 이미지, 탐지 표, 조각 수, 너비, 높이, 걸린 초)"""
    import io
    import time

    from PIL import Image

    image = Image.open(io.BytesIO(data)).convert("RGB")
    start = time.time()
    if aerial:
        plotted, dets, n_tiles = detect_aerial(load_model(model_name), image, conf, tile_mode)
    else:
        plotted, dets, n_tiles = detect_ground(ground_models(model_name), image, conf, tile_mode)
    return plotted, dets, n_tiles, image.width, image.height, time.time() - start


def counts_of(dets):
    return {g: int((dets["묶음"] == g).sum()) for g in GROUP_LABELS}


def score_of(counts, weights):
    return sum(counts[g] * weights[g] for g in GROUP_LABELS)


VTYPE_MODES = {"빠르게 (약 60%)": "fast", "정확하게 (약 70%, 느림)": "accurate"}
VTYPE_SEC = {"fast": 6, "accurate": 19}   # 이 PC CPU 에서 차량 1대 (2026-11-26, 뒤에서 학습이 돌 때 잰 값. 한가하면 더 빠름)


def _vtype_module():
    import sys
    p = str(ROOT_DIR / "models")
    if p not in sys.path:
        sys.path.insert(0, p)
    import veh_classify
    return veh_classify


@st.cache_resource(show_spinner="차량 종류 모델 불러오는 중... (처음 한 번 1~2분)")
def _vtype_model(mode):
    vc = _vtype_module()
    fast = vc.Fast()
    return fast if mode == "fast" else vc.Accurate(fast)


# 차량 종류 모델별 정확도 (2026-11-25~26, 팀이 고른 정답 1,201개, 5겹 교차 검증 StratifiedKFold(5, shuffle, 0)).
# 출처: ojh/models/saved/yolo/veh_types/finetune_results.json(코랩 A100 파인튜닝), blend_results.json(veh_blend.py),
#       class_report_combo.json(veh_report2.py), bb_combo.json(veh_backbones.py). 파일은 git 에 없어 숫자를 여기에 적어 둔다.
VTYPE_SCORES = [
    ("🏆 정확하게 (AERO)", "EVA02-L CLIP 파인튜닝 80% + 그림 모델 5개 묶음 20% 섞기", 70.6, "섞는 비율까지 따로 떼어 다시 재면 70.3%. 앱의 모델은 정답 1,479개로 이어 학습한 최신판"),
    ("큰 모델 앙상블", "EVA02-L CLIP ×2 + EVA02-L MIM + DINOv2-L 평균 (좋은 것부터 더해 가며 고름)", 69.4, "코랩 A100"),
    ("단일 모델 1등", "EVA02-L CLIP (eva02_large_patch14_clip_224.merged2b) 파인튜닝", 68.4, "코랩 A100, 15에폭, 약 22분"),
    ("파인튜닝", "EVA02-L MIM (eva02_large_patch14_224.mim_m38m)", 64.5, ""),
    ("⚡ 빠르게 (AERO)", "ConvNeXt-T + DINOv2-B + ViT-B(SWAG) + YOLO26x분류 + DINOv2-L 특징 → 로지스틱 회귀", 60.5, "PC, 정답이 30개 늘 때마다 자동 재학습"),
    ("파인튜닝", "DINOv2-L (vit_large_patch14_dinov2)", 58.7, ""),
    ("파인튜닝", "SigLIP2-B · EVA02-B · ConvNeXtV2-L · ConvNeXt-S/B", 55.0, "54.9~56.6%"),
    ("기준선", "ConvNeXt-Tiny 특징 하나 → 로지스틱 회귀", 49.2, "처음 고르기 화면용 모델"),
    ("파인튜닝", "EfficientNetV2-M · DINOv2-B · CLIP-B", 45.0, "44.6~45.2%"),
]
VTYPE_PER_CLASS = {"탑승교": 97, "버스": 80, "계단차": 80, "급유차": 77, "승용차": 77, "수하물 차량": 75, "토잉카": 69,
                   "케이터링 트럭": 67, "벨트 로더": 65, "지상전원장치": 65, "차량 아님": 64, "트럭": 43, "컨테이너 로더": 25}


def vehicle_model_info():
    """차량 종류 모델 설명: 어떻게 70%를 냈는지, 모델별 정확도, 종류별 정확도."""
    with st.expander("🚚 차량 종류 모델: 어떻게 만들었고 몇 % 인가요?"):
        st.markdown(
            "- **할 일**: YOLO 가 찾은 차량을 **13종류**(급유차·토잉카·버스·계단차·케이터링·벨트 로더·수하물 차량·컨테이너 로더·"
            "지상전원장치·트럭·승용차·탑승교·차량 아님) 중 하나로 맞힘\n"
            "- **정답**: 팀이 고르기 화면에서 직접 고른 공항 사진 속 차량 (채점 1,201개, 지금은 1,479개)\n"
            "- **입력**: 차량 사진 + 주변 사진(빨간 상자) 두 장을 같은 모델에 넣고 특징을 이어 붙임 → 주변(비행기 옆인지 등)도 단서로 씀\n"
            "- **채점**: 5겹 교차 검증 — 정답을 5묶음으로 나눠 4묶음으로 배우고 남은 1묶음을 맞히기를 5번. 처음 보는 차량으로만 채점\n"
            "- **70%까지 온 길**: ConvNeXt-Tiny 특징 49% → 더 좋은 그림 모델 5개 특징 묶기 60.5% → "
            "코랩 GPU(A100)에서 큰 모델 14종 파인튜닝, 1등 EVA02-L CLIP 68.4% → 큰 모델과 묶음 모델 확률 섞기 **70.6%**\n"
            "- **이어 학습 확인**: 정답 1,201개 모델이 처음 본 어려운 새 정답에서 39.6% → 새 정답 일부로 이어 학습하니 51.8% (+12.2%p)")
        st.dataframe(pd.DataFrame([{"구분": a, "AERO에서 쓰기": "✅ 고를 수 있음" if "(AERO)" in a else "비교 실험 기록",
                                    "모델·방법": b, "정확도(%)": c, "비고": d} for a, b, c, d in VTYPE_SCORES]),
                     hide_index=True, use_container_width=True,
                     column_config={"정확도(%)": st.column_config.ProgressColumn("정확도(%)", min_value=0, max_value=100, format="%.1f")})
        st.caption("AERO 에서는 ⚡빠르게·🏆정확하게 두 가지를 고를 수 있어요. 나머지는 코랩에서 같은 정답·같은 채점으로 비교한 기록이에요 "
                   "(최종 모델 파일을 만들지 않았고, CPU 에서 큰 모델 여러 개는 차량 1대에 1분 가까이 걸려 앱에 넣지 않음).")
        st.caption("종류별 정확도 (단일 1등 EVA02-L CLIP, 5겹 채점)")
        st.bar_chart(pd.Series(VTYPE_PER_CLASS, name="정확도(%)").sort_values(), horizontal=True, height=330)
        st.caption("트럭(급유차·케이터링과 헷갈림)과 컨테이너 로더(정답 15개뿐, 카트와 헷갈림)가 낮아요. "
                   "확신 50% 이상인 차량(60%)만 보면 83.7%, 후보 2개 안에 정답이 있는 비율은 82.4%예요.")


def vehicle_types_section(data, dets, photo_id):
    """YOLO 가 찾은 차량마다 13종류 중 무엇인지 맞힌다 (ojh/models/veh_classify.py). 모델 파일이 없으면 안내만."""
    import io

    from PIL import Image

    vc = _vtype_module()
    ok = vc.available()
    veh = dets[dets["묶음"] == "vehicle"].reset_index(drop=True)
    with st.container(border=True):
        st.markdown(f"**🚚 차량 종류 맞히기** (찾은 차량 {len(veh)}대)")
        if not ok["fast"]:
            st.caption("차량 종류 모델 파일(combo_model.joblib)이 없어 쓸 수 없습니다. 모델은 git 에 없고 드라이브로 공유합니다.")
            return
        modes = [k for k, v in VTYPE_MODES.items() if ok[v]]
        c1, c2 = st.columns([3, 2])
        pick = c1.segmented_control("방식", modes, default=modes[0], required=True, key=f"vt_mode_{photo_id}",
                                    help="빠르게: 그림 모델 5개 특징 + 로지스틱 회귀 (5겹 채점 약 60%, 정답이 늘 때마다 자동 재학습). "
                                         "정확하게: 코랩에서 학습한 EVA02-L CLIP 과 빠른 모델을 8:2로 섞음 (5겹 채점 70.6%). "
                                         "CPU 에서 차량 1대 약 20초라 느립니다.")
        mode = VTYPE_MODES[pick]
        # 탐지 모델·신뢰도를 바꾸면 차량 상자가 달라지므로 상자 목록까지 키에 넣는다 (옛 결과를 잘못 보여 주지 않게)
        key = f"vt_{photo_id}_{mode}_{hashlib.md5(veh[BOX_COLS].round(1).to_numpy().tobytes()).hexdigest()[:8]}"
        if key not in st.session_state:
            c2.caption(f"예상 약 {max(1, round(len(veh) * VTYPE_SEC[mode] / 60))}분 (차량 {len(veh)}대)")
            if not c1.button("차량 종류 맞히기", key=f"vt_btn_{key}", type="primary"):
                if not ok["accurate"]:
                    st.caption("정확하게(70%)는 코랩 최종 모델 파일(veh_eva02l_final.pt, 607MB)과 timm 이 있어야 나옵니다.")
                return
            image = Image.open(io.BytesIO(data)).convert("RGB")
            s = image.width / _shrink(image).width   # 탐지 상자는 줄인 사진 기준 → 원본 크기로 되돌려 선명하게 자른다
            pairs = [vc.cut(image, [v * s for v in r[BOX_COLS].tolist()]) for _, r in veh.iterrows()]
            model = _vtype_model(mode)
            with st.spinner(f"차량 {len(pairs)}대 종류 맞히는 중..."):
                P = model.proba(pairs)
            st.session_state[key] = [(pairs[k][0], model.classes[int(p.argmax())], float(p.max())) for k, p in enumerate(P)]
        res = st.session_state[key]
        cnt = pd.Series([vc.NAMES.get(t, t) for _, t, _ in res]).value_counts().rename("대수")
        st.dataframe(cnt, use_container_width=True)
        cols = st.columns(6)
        for k, (crop, t, p) in enumerate(res[:36]):
            cols[k % 6].image(crop, caption=f"{vc.NAMES.get(t, t)} {p:.0%}", use_container_width=True)
        if len(res) > 36:
            st.caption(f"사진은 처음 36대만 보여 줍니다 (전체 {len(res)}대는 위 표에 셈).")


def show_apron_congestion():
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        st.error("ultralytics 패키지가 없습니다. 저장소 최상위에서 `pip install -r requirements.txt` 를 실행해주세요.")
        return

    st.caption("공항 계류장·활주로 사진을 올리면 YOLO 가 비행기·차량·사람을 찾아 셉니다. "
               "**AI로 자리 수 알아보기**를 누르면 Gemini 가 사진에 보이는 주기장(빈자리 포함) 수와 장소를 알려주고, "
               "'서 있는 비행기 ÷ 보이는 자리'(점유율)로 혼잡 등급을 매깁니다.")

    photo_type = st.segmented_control("사진 종류", PHOTO_TYPES, default=PHOTO_TYPES[0], required=True,
                                      help="지상·CCTV: 옆이나 비스듬히 찍은 사진. 위성·항공: 바로 위에서 내려다본 사진(구글 지도 위성 화면 등). "
                                           "종류에 맞는 모델을 써야 정확합니다.")
    aerial = photo_type == PHOTO_TYPES[1]

    with st.expander("탐지·혼잡도 기준 조정", icon=":material/tune:"):
        c1, c2, c3 = st.columns(3)
        if aerial:
            model_name = AERIAL_MODELS[c1.selectbox("모델", list(AERIAL_MODELS),
                                                    help="항공사진(DOTA)으로 학습된 모델입니다. YOLO26x 가 더 정확하지만 약 2~3배 느립니다.")]
        else:
            options = dict(GROUND_MODELS)
            if (WEIGHTS_DIR / APRON_MODEL).exists():   # 계류장 학습 모델은 파일이 있을 때만 (git 에 없음). 넓은 전경 사진엔 안 맞아 기본값은 아님
                first = next(iter(options.items()))
                options = {first[0]: first[1], APRON_LABEL: APRON_MODEL, **options}
            if (WEIGHTS_DIR / AUTO_MODEL).exists():
                # 2026-11-26 테스트 사진 5장 점검: 기본 YOLO26x 가 놓친 카트·토잉카 같은 작은 지상 차량을 확신 0.8 안팎으로 찾고
                # (터미널 근접 사진 차량 36 → 159대, 인천 전경 비행기 27 → 83대) 위성 사진에서도 비행기 16대를 찾았다(기본 0대).
                # 그래서 파일이 있으면 맨 앞(기본값)에 둔다. 약 2배 느리다. 파일이 없는 팀원은 기본 YOLO26x 그대로.
                options = {AUTO_LABEL: AUTO_MODEL, **options}
            model_name = options[c1.selectbox("모델", list(options),
                                              help="넓은 공항 전경용: 높은 곳에서 찍은 공항 사진으로 추가 학습해 멀리 있는 작은 비행기·지상 차량을 더 많이 찾지만 "
                                                   "잘못 찾는 것도 늘어납니다(사람은 기본 YOLO26x 로 찾음). 계류장 학습 모델: 벨트 로더·급유차·토잉카·탑승교 등 지상조업 장비까지 찾고, "
                                                   "사람·승용차·버스는 기본 YOLO26x 로 함께 찾습니다(두 모델이라 약 2배 느림). "
                                                   "YOLO26 은 Ultralytics 최신 모델로, 클수록 정확하지만 느립니다.")]
        # 기본값 (2026-11-19 확인): 위성 두바이는 진짜 비행기 18대가 모두 0.87 이상, 오탐 1개가 0.24 -> 0.3.
        # 지상 뮌헨은 진짜 7대가 0.51 이상, 탑승교 오탐이 0.21 -> 0.25. 사진 종류마다 따로 기억하도록 key 를 나눈다.
        conf = c2.slider("탐지 신뢰도 기준", 0.05, 0.9, 0.3 if aerial else 0.25, 0.05, key=f"apron_conf_{int(aerial)}",
                         help="낮추면 더 많이 찾지만 잘못 찾는 것도 늘어납니다.")
        tile_mode = c3.radio("조각내서 찾기", list(TILE_MODES), horizontal=True,
                             help="넓은 사진을 겹치게 잘라 조각마다 따로 찾은 뒤 합칩니다. 멀리 있는 작은 비행기·차량을 훨씬 잘 찾습니다. "
                                  "'세밀하게'는 더 작게 잘라 2배로 키워 찾아 게이트에 붙은 작은 비행기·차량까지 찾지만 약 5배 느립니다.")
        w1, w2, w3 = st.columns(3)
        weights = {"aircraft": w1.number_input("비행기 1대 점수", 0.0, 20.0, DEFAULT_WEIGHTS["aircraft"], 0.5),
                   "vehicle": w2.number_input("차량 1대 점수", 0.0, 10.0, DEFAULT_WEIGHTS["vehicle"], 0.5),
                   "person": w3.number_input("사람 1명 점수", 0.0, 10.0, DEFAULT_WEIGHTS["person"], 0.5, disabled=aerial,
                                             help="위성·항공사진 모델은 사람을 찾지 않습니다." if aerial else None)}
        st.caption("주기장 수를 입력하면: 점유율 = 비행기 수 ÷ 주기장 수  ·  "
                   + "  ".join(f"{icon} {name} {'<' + format(limit, '.0%') if limit != float('inf') else '그 이상'}" for limit, name, icon in OCC_GRADES))
        st.caption("주기장 수가 0이면: 혼잡 점수 = 비행기 수 × 비행기 점수 + 차량 수 × 차량 점수 + 사람 수 × 사람 점수  ·  "
                   + "  ".join(f"{icon} {name} {'<' + str(limit) if limit != float('inf') else '그 이상'}" for limit, name, icon in GRADES))

    files = st.file_uploader("사진 올리기 (여러 장 가능)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    if not files:
        st.info("공항 계류장·활주로 사진을 올려주세요. 위에서 사진 종류(지상·CCTV / 위성·항공)를 맞게 고르면 더 정확합니다.",
                icon=":material/photo_camera:")
        with st.expander("이 기능은 이렇게 동작해요"):
            st.markdown(
                "- **지상·CCTV 모델**: YOLO26(x/m/s, Ultralytics 최신). 일상 사진 80종(COCO)으로 학습되어 비행기·버스·트럭·승용차·사람을 찾습니다.\n"
                "- **위성·항공 모델**: YOLO26x-OBB(또는 빠른 YOLO11m-OBB). 항공사진 데이터(DOTA)로 학습되어 위에서 본 비행기·헬리콥터·대형/소형 차량을 기울어진 박스로 찾습니다.\n"
                "- **조각내서 찾기**: 넓은 사진을 겹치게 잘라 따로 찾은 뒤, 같은 물체를 여러 번 센 박스는 하나로 합칩니다.\n"
                "- **지상조업 장비 모델**(파일이 있을 때만): 공개 계류장 사진 516장으로 YOLO26x 를 추가 학습해 수하물 트럭·벨트 로더·급유차·토잉카·"
                "계단차·케이터링 트럭·컨테이너 로더·지상전원장치·탑승교를 찾습니다(테스트 mAP50 장비 0.77~0.99). 비행기·사람·승용차·버스는 기본 YOLO26x 가 찾습니다. "
                "가까이서 찍은 CCTV 사진으로 배워서, 높은 곳에서 찍은 공항 전경에서는 장비를 잘못 찾을 수 있습니다.\n"
                "- **넓은 공항 전경용 모델**(파일이 있을 때만, 지상 사진 기본값): 공항 전경 사진에 여러 모델이 동의한 박스로 YOLO26x 를 추가 학습했습니다. "
                "2026-11-26 테스트 사진 6장 점검에서 기본 YOLO26x 가 놓친 카트·토잉카 같은 작은 지상 차량을 훨씬 많이 찾았고"
                "(터미널 근접 차량 36 → 159대, 인천 전경 비행기 27 → 83대) 위성 사진에서도 비행기를 찾았습니다(기본 0대 → 16대). 약 2배 느립니다.\n"
                "- **한계**: 기본 모델은 공항 사진만으로 학습한 것은 아니라서, 토잉카·급유차 같은 지상조업 차량은 '트럭'이나 '차량'으로만 잡히고 "
                "가려지거나 아주 작은 대상은 놓칠 수 있습니다. 위성 사진은 위에서 '위성·항공사진'을 골라야 정확합니다.\n"
                "- **🚚 차량 종류 맞히기**(지상 사진, 모델 파일이 있을 때만): 찾은 차량마다 13종류(급유차·토잉카·버스·계단차·케이터링·벨트 로더·"
                "수하물 차량·컨테이너 로더·지상전원장치·트럭·승용차·탑승교·차량 아님) 중 무엇인지 맞힙니다. "
                "'빠르게'(약 60%, 그림 모델 5개 + 로지스틱 회귀)와 '정확하게'(70.6%, 코랩에서 파인튜닝한 EVA02-L CLIP 과 섞기, 느림) 중 고릅니다. "
                "자세한 정확도는 아래 '차량 종류 모델' 칸에 있습니다.\n"
                "- **🌉 탑승교로 자리 수 채우기**: '지상조업 장비까지' 모델이 탑승교를 찾으면 탑승교 1개 = 붙는 자리 1개로 보고 자리 수 칸을 미리 채웁니다. "
                "원격 주기장은 못 세므로, 비행기가 탑승교보다 많으면 채우지 않고 참고로만 알려 줍니다. 가까운 CCTV 사진에서 잘 되고, 넓은 전경에서는 탑승교를 일부만 찾습니다.\n"
                "- **AI 자리 수 추정**: YOLO 는 '주기장'을 찾지 못하므로, Gemini 가 탑승교·바닥 표시선·줄지은 비행기 사이 빈틈을 보고 "
                "사진에 보이는 자리 수(빈자리 포함)와 장소, 그 공항 전체 주기장 수(참고)를 알려줍니다. "
                f"모델은 {' → '.join(AI_STAND_MODELS)} 순서로 시도합니다(무료 등급 flash 는 하루 20번). 같은 사진은 저장된 결과를 다시 씁니다.\n"
                "- **등급**: 자리 수가 있으면 비행기가 자리를 얼마나 채웠는지(점유율)로 매깁니다. 자리 수가 없으면 탐지 수에 가중치를 곱한 점수로 매기는데, "
                "이 경우 큰 공항일수록 원래 비행기가 많아 혼잡하게 나오기 쉽습니다.\n"
                "- **AI 한계**: 공항 전체처럼 아주 넓은 사진은 자리 수가 물을 때마다 달라질 수 있습니다. 터미널 하나 정도가 보이는 사진이 가장 정확합니다. "
                "틀렸으면 자리 수 칸을 직접 고치면 됩니다.")
        vehicle_model_info()
        return

    summary = []
    for i, f in enumerate(files):
        with st.spinner(f"{f.name} 분석 중..."):
            try:
                plotted, dets, n_tiles, width, height, elapsed = detect_cached(f.getvalue(), model_name, aerial, conf, tile_mode)
            except Exception:
                st.warning(f"{f.name} 은(는) 열 수 없는 사진입니다.")
                continue
        counts = counts_of(dets)
        photo_id = hashlib.md5(f.getvalue()).hexdigest()[:12]
        ai_key = f"ai_stands_{photo_id}"

        st.markdown(f"#### {f.name}")
        left, right = st.columns([3, 2])
        with right:
            ai = st.session_state.get(ai_key)
            if not isinstance(ai, dict):
                btn_slot = st.empty()
                if btn_slot.button("🤖 AI로 자리 수·장소 알아보기", key=f"ai_btn_{photo_id}", use_container_width=True,
                                   help="Gemini 가 사진에 보이는 주기장(빈자리 포함) 수와 장소를 추정합니다. 20~60초 걸리고, 같은 사진은 저장된 결과를 씁니다."):
                    with st.spinner("Gemini 가 사진을 보고 자리를 세는 중... (20~60초)"):
                        try:
                            st.session_state[ai_key] = ai = ai_stands(f.getvalue())
                            btn_slot.empty()
                        except Exception as e:
                            st.session_state[ai_key] = str(e)
                if isinstance(st.session_state.get(ai_key), str):
                    st.warning(st.session_state[ai_key], icon=":material/error:")
            if isinstance(ai, dict):
                place = ai.get("location") or "알 수 없음"
                total = ai.get("airport_total_stands")
                st.markdown(f"📍 **{place}** (확신 {CONF_KO.get(ai.get('location_confidence'), '-')})"
                            + (f"  \n공항 전체 주기장 약 **{total}개** (AI 지식 기반 참고값)" if total else ""))
                st.caption(f"AI가 본 이 사진: 서 있는 비행기 {ai['aircraft_parked']}대 · 보이는 자리 {ai['stands_total']}개 "
                           f"(빈자리 {ai['stands_empty']}) · 신뢰도 {CONF_KO.get(ai.get('confidence'), '-')} · {ai['model']}, {ai['seconds']}초"
                           + (f"  \n{ai['reason']}" if ai.get("reason") else ""))
                if ai["model"] in AI_LOW_ACCURACY:
                    st.caption("⚠️ 정확한 모델의 하루 한도를 다 써서 가벼운 모델로 추정했습니다. 비행기가 빽빽한 사진에서는 많이 셀 수 있습니다.")
                elif ai.get("confidence") == "low":
                    st.caption("⚠️ 넓은 사진이라 추정이 부정확할 수 있습니다. 터미널 하나 정도가 보이는 사진이 가장 정확합니다.")
            ai_total = ai["stands_total"] if isinstance(ai, dict) else 0
            # 탑승교 1개 ≈ 붙는 자리 1개 (2026-11-26). 계류장 학습 모델만 탑승교를 찾는다(가까운 CCTV 사진에서 mAP50 0.99).
            # 원격 주기장은 못 세고 큰 비행기 자리는 탑승교가 2~3개라, AI 추정이 없을 때만 '최소 자리 수'로 미리 채운다.
            bridges = int((dets["종류"] == "탑승교").sum()) if len(dets) else 0
            # 시험 CCTV 사진 3장: 탑승교 1개를 정확히 찾았지만 뒤쪽 원격·유도로 비행기가 7~9대라, 1로 채우면 점유율이 수백 %가 된다.
            # 그래서 비행기 수가 탑승교 수 이하일 때만 채운다(사진 속 비행기가 모두 탑승교에 붙어 있을 수 있는 경우).
            fill = bridges if bridges and counts["aircraft"] <= bridges else 0
            start_val = int(ai_total) or fill
            if bridges:
                st.caption(f"🌉 탑승교 {bridges}개 찾음 → 탑승교에 붙는 자리 약 {bridges}개"
                           + (" (AI 추정이 있어 자리 수는 AI 값을 씀)" if ai_total else
                              " (자리 수 칸에 미리 채움. 원격 주기장이 있으면 더하면 됩니다)" if fill else
                              f" (비행기 {counts['aircraft']}대가 더 많아 원격 주기장·유도로 비행기가 있는 것 같아 자리 수에는 채우지 않음)"))
            # 사진마다 찍힌 구역이 달라 자리 수도 다르다. 바꿔도 탐지는 다시 돌지 않는다(저장된 결과 사용)
            stands = st.number_input("이 사진에 보이는 자리 수 (원격 주기장 포함)", 0, 500, start_val, 1,
                                     key=f"stands_{photo_id}_{ai_total}_{bridges}",
                                     help="사진에 보이는 비행기 세울 자리 수(빈자리 포함). AI 추정값이 처음 채워지고, AI 추정이 없으면 찾은 탑승교 수가 채워집니다. "
                                          "틀렸으면 고치면 됩니다. 0이면 탐지 대수에 가중치를 곱한 점수로 판단합니다.")
        parked = ai["aircraft_parked"] if isinstance(ai, dict) else None
        name, icon, value, detail_text = judge(counts, weights, stands, parked)
        summary.append({"사진": f.name, "장소": ai.get("location") if isinstance(ai, dict) else "-",
                        **{GROUP_LABELS[g]: counts[g] for g in GROUP_LABELS},
                        "주기장": str(stands) if stands else "-", "판단 방식": "점유율" if stands else "점수",
                        "값": f"{value:.0%}" if stands else f"{value:.1f}점", "등급": f"{icon} {name}",
                        "_rank": (OCC_GRADES if stands else GRADES).index(next(g for g in (OCC_GRADES if stands else GRADES) if g[1] == name))})

        left.image(plotted, use_container_width=True)
        left.caption(f"{model_name.replace('.pt', '')} · {width}×{height}"
                     + (f" · 조각 {n_tiles}개" if n_tiles else "") + f" · {elapsed:.1f}초")
        with right:
            st.metric("혼잡 등급", f"{icon} {name}", detail_text, delta_color="off", border=True)
            if stands and (counts["aircraft"] if parked is None else parked) > stands:
                st.caption("비행기가 자리 수보다 많습니다. 유도로·활주로의 비행기까지 셌거나 자리 수를 적게 넣었을 수 있습니다.")
            if parked is not None:
                st.caption(f"점유율은 AI가 센 '서 있는 비행기' {parked}대로 계산했습니다 (YOLO 탐지 {counts['aircraft']}대는 유도로의 비행기도 포함할 수 있음).")
            if counts["aircraft"] and counts["vehicle"] / counts["aircraft"] >= BUSY_VEHICLES_PER_AIRCRAFT:
                st.caption(f"🚚 지상조업 활발: 비행기 1대당 차량 {counts['vehicle'] / counts['aircraft']:.1f}대")
            st.markdown("  \n".join(f"{GROUP_LABELS[g]} **{counts[g]}**{'명' if g == 'person' else '대'}" for g in GROUP_LABELS
                                   if not (aerial and g == "person")))
            if len(dets):
                detail = dets.groupby("종류").agg(개수=("신뢰도", "size"), 평균_신뢰도=("신뢰도", "mean")).round(2)
                st.dataframe(detail.sort_values("개수", ascending=False), use_container_width=True)
            else:
                st.caption("찾은 대상이 없습니다. 사진 종류가 맞는지 확인하고, 신뢰도 기준을 낮춰보세요.")
        if not aerial and counts["vehicle"]:
            vehicle_types_section(f.getvalue(), dets, photo_id)
        if i < len(files) - 1:
            st.divider()

    if not aerial and summary:
        vehicle_model_info()   # 차량 종류 모델 설명·정확도표 (사진 분석 뒤에도 볼 수 있게)

    if len(summary) > 1:
        st.markdown("### 사진별 비교")
        table = pd.DataFrame(summary)
        st.dataframe(table.drop(columns="_rank"), use_container_width=True, hide_index=True)
        if (table["판단 방식"] == "점유율").all():
            occ = table.set_index("사진")["값"].str.rstrip("%").astype(float).rename("주기장 점유율(%)")
            st.bar_chart(occ, horizontal=True, height=60 + 40 * len(table))
        elif table["판단 방식"].nunique() > 1:
            st.caption("사진마다 판단 방식(점유율/점수)이 달라 값을 직접 비교할 수 없어 등급 단계로 비교합니다. 모든 사진에 주기장 수를 넣으면 점유율로 비교합니다.")
        busiest = table.loc[table["_rank"].idxmax()]
        st.success(f"가장 혼잡한 사진은 **{busiest['사진']}** ({busiest['등급']}, {busiest['판단 방식']} {busiest['값']}) 입니다.")
