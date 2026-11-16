# 4ir 항공 데이터 프로젝트

- 실행: `.venv/Scripts/python.exe -m streamlit run app.py` (시스템 파이썬에는 lightgbm·xgboost·tensorflow가 없다)
- 내 작업은 모두 `ldh/` 안에 둔다(팀원 파일과 섞지 않는다). `core/`, `categories/`, `scripts/`(학습·평가·전처리),
  `collect/`(인천 수집기), `tests/`, `docs/`, `notebooks/`, `config/`(tuned_params.json), `datasets/`·`results/`·`logs/`(아래, git 제외).
  프로젝트 루트에서 모듈로 실행한다(예: `python -m ldh.scripts.run_models korea_ops:kac_delay_hour`, `python -m ldh.tests.test_collect_icn`).
- 루트에는 `app.py`(메뉴), `requirements.txt`, `CLAUDE.md`, `.streamlit/config.toml`(밝은 테마, PR #9)과 사람별 폴더(`ldh/`, `ojh/`)만 둔다.
- 팀원 JhonOh(ojh) 작업은 `ojh/`이고 배치는 JhonOh가 정한다(PR #6): `live_app.py`(실시간 항공기 화면)와 탭 파일, `fetch_*.py`, `data_collector.py`, `check_airport_data.py`가 `ojh/` 바로 아래, `models/`(train.py, train2.py, saved), `data/`. 팀원 파일은 옮기거나 이름을 바꾸지 않는다(다음 PR과 충돌). 루트 `app.py`는 JhonOh가 PR #9에서 통합 앱 AERO INSIGHT(상단·사이드 그룹 메뉴, 디자인 CSS)로 바꿨다. `ojh/`를 import 경로에 넣고, ojh 화면은 `ojh/views/`의 래퍼(실시간 지도·교통량 예측·공항 수요 예측)로 연다. ldh 카테고리는 `META["group"]`에 맞는 메뉴 그룹에 자동으로 붙는다. app.py 화면 구성은 JhonOh 쪽을 기준으로 한다.
- 데이터 폴더(git 제외):
  - `ldh/datasets/` 내 입력 자료: `raw/bts_zip`(BTS 원본 zip), `raw/reference`(받은 참고 자료, 재배포 제한 있음), `raw/icn_live`(인천 수집),
    `processed/bts_parquet`(prepare_data 결과, 2020년 이후), `processed/cache`(학습용 캐시)
  - `ldh/results/` 내 결과: `full`(전체 학습), `2026`(2026년 학습), `legacy_models`(예전 scripts/train.py 모델)
- 공개하면 안 되는 것: `.streamlit/secrets.toml`과 `.env`(키)는 읽거나 출력하거나 커밋하지 않는다. `ldh/datasets/raw/reference`는 커밋하지 않는다(EUROCONTROL 비상업 한정, OpenSky 재배포 제한).
- 인천 수집기는 `ldh/collect/collect_icn.py`다. 윈도우 예약 작업 3개(`4ir_icn_status`, `4ir_icn_closing`, `4ir_icn_backup`)가 이 경로로 실행하므로, 옮기면 예약 작업 인자도 바꿔야 한다.
  수집 3개는 사용자 요청으로 2026-11-16에 멈추도록 끝 시각을 걸었다(수집·마감 08:59, 백업 09:30). 다시 모으려면 트리거의 끝 시각을 지운다.
  수집은 2026-11-16 08:50에 멈췄고, 10일치 실전 채점(`ldh.scripts.predict_icn_live`, 도착 PR-AUC 0.318·출발 0.278)을 마친 뒤 일회성 예약 작업 `4ir_icn_live_eval`은 지웠다. 결과는 `korea_ops/icn_live_eval.json`이고 인천 예보 화면에 나온다.
- 인천 수집 자료: API 시각은 게이트 도착·출발이라 `korea_ops/pipeline.py`의 `LIVE_OFFSET`(수집 10일치로 정한 값, 도착 -12분·출발 +15분)으로 학습 기준(착륙·이륙)에 맞춘다. 학습 자료에 섞지 않고 실전 채점에만 쓴다.
- 내려받기 스크립트와 tune_small.py는 2026-11-11에 지웠다. 필요하면 커밋 `2d31ba4e`에서 꺼낸다(예: `git show 2d31ba4e:download_kosis.py`). 로그는 `ldh/logs/`.
- 구글 드라이브(G:)는 팀원과 모델·데이터를 공유하는 곳이다. **사용자가 올려 달라고 할 때만** 올리고, 그 외 작업에서는 읽기·동기화·정리·삭제를 하지 않는다. 예외: 예약 작업 `4ir_icn_backup`이 인천 수집 자료만 매시간 자동 백업한다(사용자가 유지하기로 함). 내 자리는 `G:\내 드라이브\4ir\ldh`(프로젝트 루트 구조), 다른 팀원 폴더(`chs`, `ojh`, `sjw`)는 건드리지 않는다.
- 컨플루언스 스페이스 `4IR(에어포)`(키 MFS)가 팀 문서다. 올리거나 고치기 전에 내용을 보여 주고 확인받는다. 팀 페이지(`4IR Team Project`, `4IR 프로젝트 계획`)에서는 이동형 칸만 고치고 표·이모지 서식을 그대로 둔다(로컬 ID로 해당 칸만 교체).
- 작업 기록은 `ldh/docs/WORK_LOG.md`에 남긴다. 결과 보고서 파일은 2026-11-11에 지웠고, 결과가 완성되면 사용자가 새로 쓴다(그전에는 만들지 않는다). 컨플루언스 정리 페이지는 요청받았을 때만 만들거나 갱신한다.
- 브랜치: 작업은 `develop`, 팀원 PR도 `develop`으로 받는다. `main`은 develop을 PR로 머지한다.
- 화면의 시계열 차트는 `ldh/core/ui.py`의 `time_chart()`로 그린다(코로나 시기 표시).

## 학습 원칙

- 학습·검증·평가 구간은 시간 순으로 나눈다. 모델·설정·앙상블 조합은 **검증 성적으로만** 고르고, 평가 성적을 보고 바꾸지 않는다(그것도 누수).
- 입력은 예측 시점(예정 출발 2시간 전)에 알 수 있는 값만 쓴다. 기준선을 못 이기면 화면에 그대로 표시한다.
- 모델을 바꾼 뒤에는 `python -m ldh.tests.test_saved_models <카테고리>`로 다시 불러오기 점검을 한다.
- 긴 학습은 따로 띄운 PowerShell 창에서 돌리고 `ldh/logs/progress.log`에 단계를 남긴다. 진행은 `ldh/scripts/watch_training.ps1 <로그 이름>`으로 본다.
  - 실행 창을 닫거나 Ctrl+C를 누르면 멈춘다. 끊기면 `run_models --resave`(모델 파일이 없는 것만)로 이어 가거나 남은 과제만 다시 돌린다.
  - 로그는 `cmd /c "... > ldh\logs\x.log 2>&1"`와 `PYTHONIOENCODING=utf-8`로 남긴다. PowerShell의 `>`는 UTF-16이라 한글이 깨진다.

## 작업별로 먼저 불러올 스킬

| 작업 | 스킬 |
|---|---|
| LightGBM·XGBoost 설정 반영, 튜닝, 드문 사건(결항·회항) 불균형 처리 | `xgboost-lightgbm`, `scikit-learn` |
| 기준선을 못 이긴 시계열 과제(시간대별 지연 편수, 기종별 월 여객, 유럽 택시인) | `statsmodels` |
| 모델 설명(SHAP) | `shap` |
| 인천 수집 자료 시각 보정, 자정 경계 처리, 표 가공 | `pandas-pro` |
| 학습 재현성, 저장·평가 파이프라인 점검 (LSTM 포함) | `ecc:mle-workflow` (검토는 `ecc:mle-reviewer` 에이전트) |
| Streamlit 화면 수정, 페이지·탭 구성 | `developing-with-streamlit` |
| 차트 새로 만들기 | `dataviz` |
| 테스트 추가 (`test_*.py`) | `ecc:python-testing` |
| 팀원 PR 검토, 충돌 해결 | `code-review`, `ecc:git-workflow` |
| 발표 자료 | Artifact 슬라이드(`quickstart` intent `slides`) 또는 `anthropic-skills:pptx` |
