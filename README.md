# AERO INSIGHT · 항공 통합 대시보드

한국·미국·유럽·전 세계 항공 자료로 운항 지연, 수요, 혼잡, 안전을 분석하고 예측한 결과를 한 화면에서 보는 Streamlit 앱입니다.
4IR 팀 프로젝트(TEAM 4IR-AIR4)의 결과물입니다.

- 저장소만 받아도 모든 화면이 나오도록 화면용 결과와 모델을 git에 함께 올렸습니다.
- 실시간 지도와 AI 기능(AI 도우미, 계류장 자리 수 추정)만 각자의 키가 필요합니다(아래 [키 넣기](#키-넣기)).

## 팀

<table>
  <tr>
    <td align="center" width="25%"><img src="ldh/docs/img/team_male.png" width="120" alt="이동형 캐릭터"></td>
    <td align="center" width="25%"><img src="https://github.com/user-attachments/assets/07bed60f-1430-4a04-b302-a0f21698b94f" width="120" alt="오정현 캐릭터"></td>
    <td align="center" width="25%"><img src="https://github.com/user-attachments/assets/f3b5dea4-ce18-4db0-ba03-35590f22bda4" width="120" alt="신정욱 캐릭터"></td>
    <td align="center" width="25%"><img src="ldh/docs/img/team_female.png" width="120" alt="최효선 캐릭터"></td>
  </tr>
  <tr>
    <td align="center"><b>이동형</b></td>
    <td align="center"><b>오정현</b></td>
    <td align="center"><b>신정욱</b></td>
    <td align="center"><b>최효선</b></td>
  </tr>
  <tr>
    <td align="center">한국·미국·유럽·안전 분석,<br>공통 학습 엔진, 인천 예보</td>
    <td align="center">실시간 관제, 교통량·공항 수요 예측,<br>계류장 혼잡도, AI 도우미, 통합 앱 구성</td>
    <td align="center">국내 공항 여객·기상·운항 수집,<br>혼잡 예상 자료</td>
    <td align="center">결항 요인,<br>국내 여객 수요 분석</td>
  </tr>
</table>

브랜치는 `develop`에서 작업하고 팀원 PR도 `develop`으로 받습니다. `main`에는 `develop`을 PR로 합칩니다.

## 빠른 시작

Python 3.12 기준입니다.

```bash
git clone https://github.com/4ir-air4/4ir_air4.git
```

```bash
cd 4ir_air4
```

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

```bash
.venv/Scripts/python.exe -m streamlit run app.py
```

macOS·Linux에서는 `.venv/Scripts/python.exe` 대신 `.venv/bin/python`을 씁니다. 브라우저에서 http://localhost:8501 이 열립니다.

계류장 혼잡도는 PyTorch를 씁니다. GPU가 없으면 위 설치 전에 CPU 판을 먼저 설치하는 편이 가볍습니다.

```bash
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

YOLO 기본 가중치는 처음 쓸 때 `ojh/models/saved/yolo/`로 내려받습니다(git에는 없음).<br>
계류장 추가 학습 모델과 차량 종류 모델은 git에 없어서 팀 공유 드라이브에서 받아야 합니다(아래 [git에 없는 것](#git에-없는-것)).

### 키 넣기

`.streamlit/secrets.toml.example`을 같은 폴더에 `secrets.toml`로 복사하고 필요한 키만 넣습니다.

| 키 | 쓰는 곳 | 받는 곳 |
|---|---|---|
| `OPENSKY_CLIENT_ID`, `OPENSKY_CLIENT_SECRET` | 실시간 지도 | [OpenSky Network](https://opensky-network.org) API 클라이언트 |
| `GEMINI_API_KEY` | AI 도우미, 계류장 자리 수 추정 | [Google AI Studio](https://aistudio.google.com/apikey) (무료 등급) |
| `TAVILY_API_KEY`, `SERPAPI_API_KEY` | AI 도우미의 최신 소식·항공권 검색(선택) | Tavily, SerpApi (무료 등급) |

`secrets.toml`은 `.gitignore`에 들어 있습니다. 키를 커밋하지 마세요.<br>
키가 없으면 해당 기능에만 안내 문구가 나오고 다른 화면은 그대로 동작합니다. 자세한 사용법은 [ojh/README.md](ojh/README.md)에 있습니다.


## 기술 스택

<p>
  <img src="https://skillicons.dev/icons?i=py,tensorflow,pytorch,sklearn,git,github,pycharm,vscode,powershell" alt="Python, TensorFlow, PyTorch, scikit-learn, Git, GitHub, PyCharm, VS Code, PowerShell">
</p>

**ML 모델**

![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD?style=flat-square)
![Random Forest](https://img.shields.io/badge/Random%20Forest-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Linear Regression](https://img.shields.io/badge/Linear%20Regression-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![LSTM](https://img.shields.io/badge/LSTM-D00000?style=flat-square&logo=keras&logoColor=white)
![GRU](https://img.shields.io/badge/GRU-D00000?style=flat-square&logo=keras&logoColor=white)
![SARIMAX](https://img.shields.io/badge/SARIMAX%20%28statsmodels%29-4051B5?style=flat-square)
![YOLO26](https://img.shields.io/badge/YOLO26%20%28Ultralytics%29-111F68?style=flat-square)
![EVA02-L CLIP](https://img.shields.io/badge/EVA02--L%20CLIP%20%28timm%29-6E6E6E?style=flat-square)
![DINOv2](https://img.shields.io/badge/DINOv2-0467DF?style=flat-square&logo=meta&logoColor=white)
![Grounding DINO](https://img.shields.io/badge/Grounding%20DINO-6E6E6E?style=flat-square)
![Ensemble](https://img.shields.io/badge/Ensemble-6E6E6E?style=flat-square)

**튜닝·설명**

![Optuna](https://img.shields.io/badge/Optuna-1F5E96?style=flat-square)
![TreeSHAP](https://img.shields.io/badge/TreeSHAP-FF0D57?style=flat-square)
![SMOTE](https://img.shields.io/badge/SMOTE-6E6E6E?style=flat-square)

**데이터·화면**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![Apache Arrow](https://img.shields.io/badge/PyArrow-D22128?style=flat-square&logo=apachearrow&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Altair](https://img.shields.io/badge/Altair-1F77B4?style=flat-square)
![Folium](https://img.shields.io/badge/Folium-199900?style=flat-square&logo=leaflet&logoColor=white)

**AI·협업**

![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=flat-square&logo=googlegemini&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-D97757?style=flat-square&logo=claude&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)
![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=flat-square&logo=googledrive&logoColor=white)
![Confluence](https://img.shields.io/badge/Confluence-172B4D?style=flat-square&logo=confluence&logoColor=white)

## 화면 구성

왼쪽 메뉴는 그룹별로 나뉘어 있고, 화면은 모두 17개입니다. 모든 화면 오른쪽 아래에는 AI 항공 도우미 '에어로'가 있습니다.

| 그룹 | 화면 | 내용 |
|---|---|---|
| 실시간 | 실시간 지도 | OpenSky 실시간 항공기 위치·고도·기종과 비행 경로 |
| | 교통량 예측 | 5분 단위 항공기 수 예측과 모델 비교 |
| | 공항 수요 예측 | 공항별 월 승객과 GDP 성장률로 수요 예측 |
| | 계류장 혼잡도 | 공항 사진에서 YOLO로 비행기·차량·사람을 세어 혼잡 등급 산정, 지상 차량 13종류 맞히기(5겹 채점 70.6%), 탑승교로 자리 수 채우기 |
| 한국 | 인천 지연 위험 예보 | 인천공항 도착·출발편 지연 위험 예보와 10일 실전 채점 |
| | 한국 운항·지연 | 국내 공항별 운항 실적, 지연 패턴, 지연·결항 원인 |
| | 국내 공항 혼잡 예상 | 공항별 시간대 예상 여객과 혼잡 구분, 최신 기상·운항 현황 |
| | 한국 수요·수송 | 국내 항공 여객·화물 수송 추이와 예측 |
| | 한국 국내선 항공료 | 노선별 항공료 흐름 |
| 미국 | 미국 운항 | BTS 운항 실적으로 지연·결항 예측 |
| | 미국 기상과 운항 | 기상 조건이 공항 지연에 주는 영향 |
| 유럽·세계 | 전 세계 항공 | OpenSky 비행 기록으로 본 지역별 교통 흐름 |
| | 유럽 공항 | EUROCONTROL 공항 운항·지연 분석 |
| 항공 안전 | 항공 안전 | NTSB 사고·사건 기록으로 본 안전 동향 |
| 종합·모델 | 통합 요약 | 카테고리별 핵심 지표와 모델 성능 |
| | 모델 비교 | 같은 기준(시간 순 미래 구간)으로 모델 비교, SHAP 설명 |
| | 2026년 학습 결과 | 2026년 자료로만 학습한 구동 확인 결과 |


## 폴더 구조

```
app.py                        | 통합 앱 메뉴(AERO INSIGHT)
requirements.txt              | 화면 실행용 패키지(리눅스 배포용 torch CPU 휠 포함)
packages.txt                  | Streamlit Cloud 시스템 패키지(OpenCV용)
AERO_INSIGHT_서비스기획_발표.pdf | 서비스 기획 발표 자료
.streamlit/                   | 밝은 테마 설정, 키 예시 파일
ojh/                          | 오정현 : 실시간 지도, 교통량·공항 수요 예측, 계류장 혼잡도, AI 도우미(설명서 README.md)
├─ views/                     | 통합 앱에서 여는 화면 래퍼
└─ models/                    | 학습 코드(train.py, train2.py), 계류장 학습 노트북, 차량 종류 맞히기(veh_*.py), 저장 모델(saved/)
ldh/                          | 이동형 : 카테고리 분석·예측
├─ categories/                | 카테고리 8개(폴더 하나가 메뉴 하나, 추가 방법은 categories/README.md)
├─ core/                      | 공통 학습 엔진, 화면(통합 요약, 모델 비교, 인천 예보, 혼잡 예상 등)
├─ scripts/                   | 자료 준비, 학습, 평가
├─ collect/                   | 인천공항 운항 수집기
├─ results/                   | 화면용 결과(요약·지표, 집계, 예측 표본, SHAP)
├─ tests/                     | 점검 스크립트
├─ docs/                      | 자료 안내(DATA_GUIDE.md), 출처(DATA_SOURCES.md), 작업 기록(WORK_LOG.md)
└─ requirements-train.txt     | 학습·자료 준비용 패키지(화면용 포함)
chs/notebooks/                | 최효선 : 결항·지연 예측, 월별 여객 수요 예측 Colab 노트북(설명 chs/README.md)
sjw/outputs/                  | 신정욱 : 국내 공항 여객·기상·운항 수집 결과(정적 내보내기, 설명 sjw/README.md)
```

## 학습 다시 돌리기

화면만 볼 때는 필요 없습니다. 학습용 패키지(LightGBM, XGBoost, TensorFlow 등)를 설치합니다.

```bash
.venv/Scripts/python.exe -m pip install -r ldh/requirements-train.txt
```

원본 자료는 용량과 재배포 제한 때문에 git에 없습니다. 메뉴별로 필요한 자료와 넣을 위치는 <br>
[ldh/docs/DATA_GUIDE.md](ldh/docs/DATA_GUIDE.md)에 있습니다. 자료를 넣은 뒤 프로젝트 루트에서 모듈로 실행합니다.

```bash
.venv/Scripts/python.exe -m ldh.scripts.run_models korea_ops:kac_delay_hour
```

학습 원칙은 다음과 같습니다.

- 학습·검증·평가 구간은 시간 순으로 나눕니다.
- 모델·설정·앙상블은 검증 성적으로만 고릅니다. 평가 성적을 보고 바꾸지 않습니다.
- 입력은 예측 시점(예정 출발 2시간 전)에 알 수 있는 값만 씁니다.
- 기준선을 못 이긴 과제는 화면에 그대로 표시합니다.

## 점검

git에 올라간 파일만으로 모든 화면을 열어 오류, "파일 없음" 안내, 100MB 넘는 파일을 찾습니다. PR을 합친 뒤에 돌립니다.

```bash
.venv/Scripts/python.exe -m ldh.tests.test_fresh_clone
```

커밋 전 변경까지 포함하려면 `--include-new`를 붙입니다. 모델을 바꾼 뒤에는 `python -m ldh.tests.test_saved_models <카테고리>`로 다시 불러오기를 확인합니다.

## git에 없는 것

| 항목 | 이유 | 받는 곳 |
|---|---|---|
| 원본 자료(`ldh/datasets/`) | 용량, 재배포 제한 | 각 기관 공개 자료(DATA_GUIDE.md) |
| 저장 모델·원 예측값(`saved_models`, `member_preds`) | 용량 | 팀 공유 드라이브 |
| 미국 운항·기상, OpenSky 건별 예측 전체 | 약 2.7GB, OpenSky 재배포 제한 | 팀 공유 드라이브(화면은 5만 건 표본을 씀) |
| ojh 계류장·차량 종류 모델 5개, `btstranstats_counts.csv` | 100MB 넘음 | 팀 공유 드라이브(파일 목록은 아래 표) |
| `.streamlit/secrets.toml`, `.env` | 개인 키 | 각자 만듦 |

재학습한 모델은 `joblib.dump(..., compress=("zlib", 6))`로 저장해야 GitHub 100MB 한도를 넘지 않습니다.

### ojh 계류장·차량 종류 모델 받기

팀 공유 드라이브에서 아래 파일을 받아 **같은 경로**에 넣으면 계류장 메뉴와 차량 종류 맞히기가 나옵니다.

| 파일 (같은 경로에 넣기) | 크기 | 없으면 |
|---|---|---|
| `ojh/models/saved/yolo/apron_cons_yolo26x.pt` | 114MB | 지상 사진 기본 모델이 일반 YOLO26x로 바뀜(넓은 사진에서 찾는 수가 크게 줆) |
| `ojh/models/saved/yolo/apron_yolo26x.pt` | 113MB | '지상조업 장비까지' 메뉴와 탑승교로 자리 수 채우기가 없음 |
| `ojh/models/saved/yolo/yolo26x-cls.pt` | 57MB | 차량 종류 '빠르게'에 필요(공식 모델이라 처음 쓸 때 내려받아질 수도 있음) |
| `ojh/models/saved/yolo/veh_types/combo_model.joblib` | 0.7MB | 차량 종류 맞히기가 아예 안 나옴 |
| `ojh/models/saved/yolo/veh_types/veh_eva02l_final2.pt` | 579MB | 차량 종류 '정확하게(70.6%)'가 없고 '빠르게'만 남음 |
| `ojh/data/raw/btstranstats_counts.csv` | 353MB | 교통량 예측의 과거 화면이 테스트 구간만 그림 |

없어도 앱은 멈추지 않고, 그 기능만 빠지거나 안내 문구가 나옵니다.

## 프로젝트 진행 하면서 느낀점.
<table>
  <tr>
    <td align="center"><b>이동형</b></td>
    <td align="left">전체적으로 자료를 수집 하는데 있어 여러 API 제공 업체 및 자료를 찾는 재미가 있었고, <br>
                     전체적인 작업 진행은 비록 Ai를 이용해 Ai협업 형식으로 진행을 했지만<br>
                     진행을 함에 있어서 구상이나 진행 방향을 직접적으로 수정 및 정리를 하면서 하는<br>
                     방식으로 사용을 함에 있어서 다른 의미로 진행함에 유종의 미를 거두게 되었습니다,<br>
                     한편으로는 대부분의 작업을 진행 함에 있어, 그리고 모델을 학습하는 과정에 있어<br>
                     실제적으로는 비는 시간들이 많아 그 시간에 팀원들과 좀 더 소통을 하는 방향을 이루었으면<br>
                     좋았을 텐데 그러지 못한 부분이 아쉬움으로 남아 있습니다.<br>
                     그럼에도 팀원 분들께서 자발적으로 상황 공유 및 추가 변경 사항에 대해 말씀을 해주셔서<br>
                     진행을 함에 있어 보다 편안하게 진행을 했습니다, 다들 감사합니다.
    <br></td>
  </tr>
  <tr>
    <td align="center"><b>오정현</b></td>
    <td align="left">
                     처음에는 "지금 하늘에 떠 있는 비행기를 지도에 띄워 보자"는 작은 생각에서 시작했는데, 하다 보니 실시간 수집기, 교통량 예측, 공항 수요 예측, 계류장 혼잡도, AI 도우미까지 이어졌고 마지막에는 팀원들의 화면을 모두 모아 AERO INSIGHT라는 하나의 앱으로 묶게 되었습니다. 기능 하나를 만들면 그다음에 필요한 것이 자연스럽게 보였고, 그걸 따라가다 보니 프로젝트가 커졌습니다.<br><br>
                     가장 기억에 남는 건 계류장 차량 종류 맞히기입니다. 처음 모델은 49%밖에 맞히지 못했고, 정답이 없으니 팀 고르기 화면을 직접 만들어 공항 사진 속 차량 1,616대를 하나하나 골랐습니다. 여러 그림 모델을 묶어 60%, 큰 모델(EVA02-L CLIP)을 코랩에서 파인튜닝해 68%, 둘을 섞어 70.6%까지 올렸을 때는 정말 뿌듯했습니다. 숫자 1%를 올리는 게 이렇게 어렵다는 걸 몸으로 배웠습니다. GPU 시간이 모자라 계정을 바꿔 가며 학습을 돌리고, 메모리가 부족해 컴퓨터가 멈추는 일도 여러 번 겪었습니다.<br><br>
                     또 하나 배운 건 "점수를 부풀리지 않는 법"입니다. 모델이 공부할 때 본 사진으로 시험을 보면 점수가 높게 나오기 마련이라, 항상 모델이 처음 보는 사진으로만 채점했습니다. 다른 모델을 섞거나 공개 데이터를 더 넣는 등 여러 방법을 시도했는데, 오히려 정확도가 떨어진 방법도 있었고 그런 결과도 숨기지 않고 README에 그대로 남겼습니다.<br><br>
                     AI와 함께 작업하면서 코드를 짜는 속도는 정말 빨라졌습니다. 하지만 무엇을 만들지, 결과가 진짜 맞는지, 화면에 보이는 숫자가 사용자에게 오해를 주지 않는지는 결국 제가 직접 사진을 보고 확인하고 판단해야 했습니다. AI가 "다 됐다"고 해도 직접 열어 보면 빠진 부분이 있었고, 그걸 찾아 다시 묻는 과정이 오히려 공부가 되었습니다.<br><br>
                     아쉬운 점도 많습니다. 고르기 화면에 쌓인 차량 사진은 거의 3만 장에 달했는데, 하나하나 사람이 직접 보고 골라야 해서 그걸 전부 고르는 건 시간상 불가능했고, 프로젝트 기간 안에는 1,616대까지 고른 것이 최대였습니다. 주어진 시간 안에서 할 수 있는 만큼은 끝까지 해 봤지만, 그래서 컨테이너 로더처럼 정답이 적은 차량은 아직 잘 못 맞히고 넓은 공항 전경에서 탑승교도 일부만 찾는 점이 아쉽습니다. 시간이 더 있다면 정답을 훨씬 더 모아 80% 이상까지 올려 보고 싶습니다. 그리고 각자 맡은 부분에 집중하다 보니 팀원들과 더 자주 이야기하지 못한 것도 아쉬움으로 남습니다.<br><br>
                     마지막에 팀장님과 함께 모든 팀원의 결과를 AERO INSIGHT 하나로 모았는데, 각자 따로 만든 결과가 한 화면에 모이는 걸 보니 정말 뿌듯했습니다. 한국·미국·유럽 데이터 분석과 공통 학습 엔진, 인천 예보까지 프로젝트의 큰 부분을 맡아 이끌어 주시고 마지막 통합까지 함께해 주신 팀장님, 차량 정답 고르기를 함께 도와준 박기흠 님, 그리고 각자 자리에서 최선을 다한 팀원들 모두 고생 많으셨습니다. 이번에 배운 것을 바탕으로, 다음에는 더 많은 데이터로 더 정확한 모델을 만들어 보고 싶습니다.
    </td>
  </tr>
  <tr>
    <td align="center"><b>신정욱</b></td>
    <td align="left">호기롭게 시작한 제 마음가짐과는 달리 AI를 다루고 학습하는 것이 상당히 복잡하고 어렵다는 것을 느꼈습니다. 팀장님이 잘 이끌어주시고 팀원분들도 다들잘 해주셔서 마칠 수 있었지만 제 자신에게 조금 부족함을 느끼고 많이 배운 시간이였습니다. 부족한 부분을 좀 더 배우고 학습해서 더 나은 제 자신으로 다음 프로젝트를 진행하고 싶습니다. </td>
  </tr>
  <tr>
    <td align="center"><b>최효선</b></td>
    <td align="left">팀원들이 모두다 각자도생이네요..기술좋은 ai덕분에 ai가 다해줘서 프로젝트 하는느낌도 안났네요..<br>자료 좀더 찾을수 있었으면 도움이 좀 더 됐을꺼같은데..새로만난 팀원분들 모두 좋으신분들 같네요<br>얘기는 많이 안해봤지만 각자 장점이 많으신분들 같아요..고생하신 팀원들 남은 기간도 더 열심히 하세요.
    </td>
  </tr>  
</table>

## 자료 출처

- 한국: 공공데이터포털(인천국제공항공사, 한국공항공사, 국토교통부), KOSIS, 공항 기상(ASOS), 기상청 공항 기상(AMOS)
- 미국: BTS(교통통계국) 운항 실적, 공항 기상(ASOS), NTSB 사고 DB
- 유럽·세계: EUROCONTROL, Eurostat, OpenSky Network, OurAirports, OpenFlights, World Bank

출처별 이용 조건은 [ldh/docs/DATA_SOURCES.md](ldh/docs/DATA_SOURCES.md)에 있습니다.<br>
EUROCONTROL 자료는 비상업 용도로만 쓸 수 있고, OpenSky 자료는 재배포가 제한됩니다.

