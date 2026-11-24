"""활주로·계류장 혼잡도: 공항 CCTV·사진에서 YOLO 로 비행기·차량·사람을 세어 혼잡 등급을 매긴다.

사진 종류에 따라 다른 모델을 쓴다 (둘 다 Ultralytics 사전 학습 모델, 추가 학습 없음):
  - 지상·CCTV 사진 : YOLO11 (COCO 80종) - 비행기·버스·트럭·승용차·사람. 크기 n/s/m 중 선택
  - 위성·항공사진  : YOLO11-OBB (DOTA 항공사진 15종) - 비행기·헬리콥터·대형/소형 차량을 회전 박스로 찾는다
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

GROUND_MODELS = {"정확하게 (YOLO11m)": "yolo11m.pt", "균형 (YOLO11s)": "yolo11s.pt", "빠르게 (YOLO11n)": "yolo11n.pt"}
AERIAL_MODEL = "yolo11m-obb.pt"
PHOTO_TYPES = ["지상·CCTV 사진", "위성·항공사진 (위에서 내려다본 사진)"]

# COCO 클래스 번호 -> (화면 이름, 묶음)
COCO_CLASSES = {4: ("비행기", "aircraft"), 2: ("승용차", "vehicle"), 5: ("버스", "vehicle"), 7: ("트럭·지상조업차", "vehicle"),
                3: ("오토바이", "vehicle"), 0: ("사람", "person")}
# DOTA 클래스 이름 -> (화면 이름, 묶음)
DOTA_CLASSES = {"plane": ("비행기", "aircraft"), "helicopter": ("헬리콥터", "aircraft"),
                "large vehicle": ("대형 차량", "vehicle"), "small vehicle": ("소형 차량", "vehicle")}
GROUP_LABELS = {"aircraft": "✈️ 비행기", "vehicle": "🚌 차량", "person": "🧍 사람"}
GROUP_INDEX = {"aircraft": 0, "vehicle": 1, "person": 2}
DEFAULT_WEIGHTS = {"aircraft": 5.0, "vehicle": 1.0, "person": 0.5}
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


def detect_ground(model, image, conf, tile_mode="보통"):
    """지상·CCTV 사진: 전체 사진(640·1280) + 조각별 탐지를 합친다. (박스 그린 RGB 이미지, 탐지 표)"""
    import torch
    from torchvision.ops import batched_nms

    image = _shrink(image)
    w, h = image.size
    classes = list(COCO_CLASSES)
    whole = [model.predict(image, conf=conf, imgsz=s, classes=classes, verbose=False)[0] for s in (640, 1280)]
    datas = [r.boxes.data for r in whole]
    tile_cfg = TILE_MODES.get(tile_mode)
    pieces = tiles_of(image, tile=tile_cfg[0]) if tile_cfg else []
    if pieces:
        for (x0, y0, piece), r in zip(pieces, model.predict([p for _, _, p in pieces], conf=conf, imgsz=tile_cfg[1], classes=classes, verbose=False)):
            datas.append(_tile_boxes(r.boxes.data, x0, y0, piece.width, piece.height, w, h))
    data = torch.cat(datas)
    if len(data):
        # 같은 물체를 승용차/트럭처럼 다른 이름으로 겹쳐 잡은 경우도 하나만 남기도록 '묶음' 단위로 겹침 제거
        groups = _group_ids(data[:, 5], COCO_CLASSES.__getitem__)
        keep = batched_nms(data[:, :4], data[:, 4], groups, iou_threshold=0.45)
        data, groups = data[keep], groups[keep]
        data = data[_drop_contained(data[:, :4], data[:, 4], groups)]
    result = whole[0]
    result.update(boxes=data)
    rows = [{"종류": COCO_CLASSES[int(c)][0], "묶음": COCO_CLASSES[int(c)][1], "신뢰도": round(s, 2)}
            for c, s in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())]
    return result.plot(line_width=2, font_size=12)[:, :, ::-1], pd.DataFrame(rows, columns=["종류", "묶음", "신뢰도"]), len(pieces)


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
    rows = [{"종류": mapping(int(c))[0], "묶음": mapping(int(c))[1], "신뢰도": round(s, 2)}
            for c, s in zip(whole.obb.cls.tolist(), whole.obb.conf.tolist())]
    return whole.plot(line_width=2, font_size=12)[:, :, ::-1], pd.DataFrame(rows, columns=["종류", "묶음", "신뢰도"]), len(pieces)


@st.cache_data(max_entries=24, show_spinner=False)
def detect_cached(data, model_name, aerial, conf, tile_mode):
    """사진 내용·설정이 같으면 저장된 탐지 결과를 돌려준다. (박스 그린 이미지, 탐지 표, 조각 수, 너비, 높이, 걸린 초)"""
    import io
    import time

    from PIL import Image

    image = Image.open(io.BytesIO(data)).convert("RGB")
    start = time.time()
    plotted, dets, n_tiles = (detect_aerial if aerial else detect_ground)(load_model(model_name), image, conf, tile_mode)
    return plotted, dets, n_tiles, image.width, image.height, time.time() - start


def counts_of(dets):
    return {g: int((dets["묶음"] == g).sum()) for g in GROUP_LABELS}


def score_of(counts, weights):
    return sum(counts[g] * weights[g] for g in GROUP_LABELS)


def show_apron_congestion():
    try:
        import ultralytics  # noqa: F401
    except ImportError as e:
        # 패키지가 없을 때뿐 아니라 cv2 가 시스템 라이브러리(libGL 등)를 못 찾을 때도 ImportError 라 원인을 함께 보인다
        print(f"[계류장] ultralytics 불러오기 실패: {e}")
        st.error("ultralytics 를 불러오지 못했습니다. 저장소 최상위에서 `pip install -r requirements.txt` 를 실행해주세요.")
        st.caption(f"원인: {e}")
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
            c1.markdown(f"**모델**: YOLO11m-OBB  \n항공사진(DOTA) 학습 모델")
            model_name = AERIAL_MODEL
        else:
            model_name = GROUND_MODELS[c1.selectbox("모델 크기", list(GROUND_MODELS),
                                                    help="클수록 정확하지만 느립니다. m 은 n 보다 약 3배 느립니다.")]
        conf = c2.slider("탐지 신뢰도 기준", 0.05, 0.9, 0.2, 0.05, help="낮추면 더 많이 찾지만 잘못 찾는 것도 늘어납니다.")
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
                "- **지상·CCTV 모델**: YOLO11(n/s/m). 일상 사진 80종(COCO)으로 학습되어 비행기·버스·트럭·승용차·사람을 찾습니다.\n"
                "- **위성·항공 모델**: YOLO11m-OBB. 항공사진 데이터(DOTA)로 학습되어 위에서 본 비행기·헬리콥터·대형/소형 차량을 기울어진 박스로 찾습니다.\n"
                "- **조각내서 찾기**: 넓은 사진을 겹치게 잘라 따로 찾은 뒤, 같은 물체를 여러 번 센 박스는 하나로 합칩니다.\n"
                "- **한계**: 두 모델 모두 공항 사진만으로 학습한 것은 아니라서, 토잉카·급유차 같은 지상조업 차량은 '트럭'이나 '차량'으로만 잡히고 "
                "가려지거나 아주 작은 대상은 놓칠 수 있습니다. 더 정확하게 하려면 공항 사진에 직접 표시한 데이터로 추가 학습(파인튜닝)해야 합니다.\n"
                "- **AI 자리 수 추정**: YOLO 는 '주기장'을 찾지 못하므로, Gemini 가 탑승교·바닥 표시선·줄지은 비행기 사이 빈틈을 보고 "
                "사진에 보이는 자리 수(빈자리 포함)와 장소, 그 공항 전체 주기장 수(참고)를 알려줍니다. "
                f"모델은 {' → '.join(AI_STAND_MODELS)} 순서로 시도합니다(무료 등급 flash 는 하루 20번). 같은 사진은 저장된 결과를 다시 씁니다.\n"
                "- **등급**: 자리 수가 있으면 비행기가 자리를 얼마나 채웠는지(점유율)로 매깁니다. 자리 수가 없으면 탐지 수에 가중치를 곱한 점수로 매기는데, "
                "이 경우 큰 공항일수록 원래 비행기가 많아 혼잡하게 나오기 쉽습니다.\n"
                "- **AI 한계**: 공항 전체처럼 아주 넓은 사진은 자리 수가 물을 때마다 달라질 수 있습니다. 터미널 하나 정도가 보이는 사진이 가장 정확합니다. "
                "틀렸으면 자리 수 칸을 직접 고치면 됩니다.")
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
                if btn_slot.button("🤖 AI로 자리 수·장소 알아보기", key=f"ai_btn_{photo_id}", width="stretch",
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
            # 사진마다 찍힌 구역이 달라 자리 수도 다르다. 바꿔도 탐지는 다시 돌지 않는다(저장된 결과 사용)
            stands = st.number_input("이 사진에 보이는 자리 수 (원격 주기장 포함)", 0, 500, int(ai_total), 1,
                                     key=f"stands_{photo_id}_{ai_total}",
                                     help="사진에 보이는 비행기 세울 자리 수(빈자리 포함). AI 추정값이 처음 채워지고, 틀렸으면 고치면 됩니다. "
                                          "0이면 탐지 대수에 가중치를 곱한 점수로 판단합니다.")
        parked = ai["aircraft_parked"] if isinstance(ai, dict) else None
        name, icon, value, detail_text = judge(counts, weights, stands, parked)
        summary.append({"사진": f.name, "장소": ai.get("location") if isinstance(ai, dict) else "-",
                        **{GROUP_LABELS[g]: counts[g] for g in GROUP_LABELS},
                        "주기장": str(stands) if stands else "-", "판단 방식": "점유율" if stands else "점수",
                        "값": f"{value:.0%}" if stands else f"{value:.1f}점", "등급": f"{icon} {name}",
                        "_rank": (OCC_GRADES if stands else GRADES).index(next(g for g in (OCC_GRADES if stands else GRADES) if g[1] == name))})

        left.image(plotted, width="stretch")
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
                st.dataframe(detail.sort_values("개수", ascending=False), width="stretch")
            else:
                st.caption("찾은 대상이 없습니다. 사진 종류가 맞는지 확인하고, 신뢰도 기준을 낮춰보세요.")
        if i < len(files) - 1:
            st.divider()

    if len(summary) > 1:
        st.markdown("### 사진별 비교")
        table = pd.DataFrame(summary)
        st.dataframe(table.drop(columns="_rank"), width="stretch", hide_index=True)
        if (table["판단 방식"] == "점유율").all():
            occ = table.set_index("사진")["값"].str.rstrip("%").astype(float).rename("주기장 점유율(%)")
            st.bar_chart(occ, horizontal=True, height=60 + 40 * len(table))
        elif table["판단 방식"].nunique() > 1:
            st.caption("사진마다 판단 방식(점유율/점수)이 달라 값을 직접 비교할 수 없어 등급 단계로 비교합니다. 모든 사진에 주기장 수를 넣으면 점유율로 비교합니다.")
        busiest = table.loc[table["_rank"].idxmax()]
        st.success(f"가장 혼잡한 사진은 **{busiest['사진']}** ({busiest['등급']}, {busiest['판단 방식']} {busiest['값']}) 입니다.")
