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

YOLO 가중치는 처음 쓸 때 `ojh/models/saved/yolo/`로 내려받습니다(git에는 없음).

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
![YOLO11](https://img.shields.io/badge/YOLO11%20%28Ultralytics%29-111F68?style=flat-square)
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
| | 계류장 혼잡도 | 공항 사진에서 YOLO로 비행기·차량·사람을 세어 혼잡 등급 산정 |
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
app.py                    | 통합 앱 메뉴(AERO INSIGHT)
requirements.txt          | 화면 실행용 패키지 17개
.streamlit/               | 밝은 테마 설정, 키 예시 파일
ojh/                      | 오정현 : 실시간 지도, 교통량·공항 수요 예측, 계류장 혼잡도, AI 도우미(설명서 README.md)
views/                    | 통합 앱에서 여는 화면 래퍼
models/                   | 학습 코드(train.py, train2.py)와 저장 모델(saved/)
ldh/                      | 이동형 : 카테고리 분석·예측
categories/               | 카테고리 8개(폴더 하나가 메뉴 하나, 추가 방법은 categories/README.md)
core/                     | 공통 학습 엔진, 화면(통합 요약, 모델 비교, 인천 예보, 혼잡 예상 등)
scripts/                  | 자료 준비, 학습, 평가
collect/                  | 인천공항 운항 수집기
results/                  | 화면용 결과(요약·지표, 집계, 예측 표본, SHAP)
tests/                    | 점검 스크립트
docs/                     | 자료 안내(DATA_GUIDE.md), 출처(DATA_SOURCES.md), 작업 기록(WORK_LOG.md)
requirements-train.txt    | 학습·자료 준비용 패키지(화면용 포함)
chs/notebooks/            | 최효선 : 결항·지연 예측, 월별 여객 수요 예측 Colab 노트북(설명 chs/README.md)
sjw/outputs/              | 신정욱 : 국내 공항 여객·기상·운항 수집 결과(정적 내보내기, 설명 sjw/README.md)
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
| `ojh/data/raw/btstranstats_counts.csv` | 353MB | 팀 공유 드라이브(없으면 과거 화면은 테스트 구간만 그림) |
| `.streamlit/secrets.toml`, `.env` | 개인 키 | 각자 만듦 |

재학습한 모델은 `joblib.dump(..., compress=("zlib", 6))`로 저장해야 GitHub 100MB 한도를 넘지 않습니다.

## 프로젝트 진행 하면서 느낀점.
<table>
  <tr>
    <td align="center"><b>이동형</b></td>
    <td align="center">여기에 내용을 입력하세요.</td>
  </tr>
  <tr>
    <td align="center"><b>오정현</b></td>
    <td align="center">여기에 내용을 입력하세요.</td>
  </tr>
  <tr>
    <td align="center"><b>신정욱</b></td>
    <td align="center">여기에 내용을 입력하세요.</td>
  </tr>
  <tr>
    <td align="center"><b>최효선</b></td>
    <td align="center">여기에 내용을 입력하세요.</td>
  </tr>  
</table>

## 자료 출처

- 한국: 공공데이터포털(인천국제공항공사, 한국공항공사, 국토교통부), KOSIS, 공항 기상(ASOS), 기상청 공항 기상(AMOS)
- 미국: BTS(교통통계국) 운항 실적, 공항 기상(ASOS), NTSB 사고 DB
- 유럽·세계: EUROCONTROL, Eurostat, OpenSky Network, OurAirports, OpenFlights, World Bank

출처별 이용 조건은 [ldh/docs/DATA_SOURCES.md](ldh/docs/DATA_SOURCES.md)에 있습니다.<br>
EUROCONTROL 자료는 비상업 용도로만 쓸 수 있고, OpenSky 자료는 재배포가 제한됩니다.

