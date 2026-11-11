# 카테고리 추가 방법

`categories/` 아래 폴더 하나가 카테고리 하나입니다. **폴더를 추가하면 메뉴, 전체 요약, `run_all.py`에 자동으로 나타납니다.** 다른 파일은 고칠 필요가 없고, 폴더를 지우면 깔끔하게 빠집니다.

## 시작하기

1. `_template` 폴더를 복사해 이름을 바꿉니다. 영문 소문자와 밑줄만 씁니다 (예: `hotel_demand`).
2. `__init__.py`의 `META`에 제목과 아이콘을 적습니다.
3. `pipeline.py`의 `run(store)`에 통계와 학습 코드를 씁니다.
4. `page.py`에 화면을 씁니다.
5. `python run_all.py <폴더이름>` 으로 실행한 뒤 `streamlit run app.py`로 확인합니다.

## 폴더 구성

| 파일 | 역할 |
|---|---|
| `__init__.py` | `META = {"title", "icon", "group", "order"}`. `group`이 같으면 메뉴에서 묶입니다. |
| `pipeline.py` | `run(store)` 함수. 원본 데이터를 읽어 결과를 `outputs/<폴더이름>/`에 저장합니다. |
| `page.py` | 화면. `outputs/`만 읽고 원본 데이터는 읽지 않습니다. |

## 지킬 약속

- **화면은 결과 파일만 읽습니다.** 그래야 원본 데이터가 없는 팀원 컴퓨터에서도 `outputs/`만 받으면 화면이 열립니다. (예외: `us_flights`의 예측 폼은 `models/`의 모델을 읽습니다.)
- **`store.save_summary()`를 마지막에 호출합니다.** `headline`(한 줄 요약), `kpis`, `models`, `notes`, `sources`가 전체 요약 화면에 그대로 쓰입니다.
- **모델 결과는 `core.ml.model_row()`로 만듭니다.** 그래야 전체 모델 표에 같은 형식으로 합쳐집니다.
- **평가는 시간순으로 나눕니다.** 무작위 분할은 쓰지 않고, 예측 시점 이후에 알 수 있는 값은 특성에서 뺍니다. 단순 기준선(평균, 과거 값 그대로)과 함께 보고합니다.
- **다른 카테고리 폴더를 import하지 않습니다.** 같이 쓸 코드는 `core/`로 옮깁니다.

## 공통 도구

| 위치 | 내용 |
|---|---|
| `core/store.py` | `Store`: `save_table`, `table`, `save_summary`, `summary` |
| `core/ml.py` | `encode`, `binary_result`, `regression_result`, `lag_regression`, `model_row` |
| `core/data.py` | `REF`(reference_data 경로), `CACHE`, `read_csv_any`(CP949·UTF-8 자동 처리), `to_number` |
| `core/ui.py` | `store_for`, `header`, `notes`, `load`, `model_tables` |

## 데이터 공유 주의

`bts_data/`, `reference_data/`, `data/`, `models/`, `outputs/`는 저장소에 올리지 않습니다. 팀 저장소가 공개 상태이고, EUROCONTROL과 OpenSky에 이용 조건이 있기 때문입니다 (`DATA_SOURCES.md` 참고).
