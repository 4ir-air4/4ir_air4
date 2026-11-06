# 자료 안내

이 PC 기준, 2026-11-06 확인. 출처와 이용 조건은 [DATA_SOURCES.md](DATA_SOURCES.md)에 있습니다.

```text
원본 자료                    변환 자료               결과                화면
bts_data/          ->  data/ontime_*.parquet  ->  outputs/        ->  app.py
reference_data/    ->  data/cache/            ->  outputs_2026/
인천공항 API       ->  data/icn_live/             (아직 모델과 연결 안 됨)
```

## 1. 폴더 한눈에 보기

| 폴더 | 내용 | 크기 | 지워지면 |
|---|---|---|---|
| `bts_data/` | 미국 운항 원본 zip | 7.9GB | 내려받기 스크립트로 다시 받음 |
| `reference_data/` | 그 외 모든 원본 (아래 2번) | 9.4GB | 대부분 다시 받음 |
| `data/ontime_*.parquet` | `bts_data`에서 필요한 열만 뽑은 월별 파일 | 2.2GB | `prepare_data.py`로 다시 만듦 |
| `data/cache/` | 학습용으로 변환한 파일 | 22.2GB | 다시 만들 수 있지만 오래 걸림 |
| `data/icn_live/` | 인천공항 API 수집 자료 | 계속 늘어남 | 드라이브 백업에서 복원. **백업 전 1시간 이내 자료는 되찾을 수 없음** |
| `outputs/` | 통계, 학습 결과, 저장 모델 | 357MB | 다시 학습해야 함 |
| `outputs_2026/` | 2026년 자료만 쓴 학습 결과 | 0.2MB | 다시 학습해야 함 |
| `models/` | 예전 `train.py` 모델과 지표. 미국 화면이 읽음 | 56MB | 미국 화면 일부가 비거나 다시 학습해야 함 |
| `_archive_before_2020/` | 2020년으로 범위를 줄이기 전의 옛 결과 | 234MB | 현재 코드에 영향 없음 |
| `categories/`, `core/` | 코드 | 1MB 미만 | Git에서 복구 (커밋된 부분만) |
| `.venv/` | 파이썬 환경 | 2.3GB | `requirements.txt`로 다시 설치 |

위 자료 폴더는 코드를 제외하고 모두 Git에 올라가지 않습니다. 그래서 팀원이나 다른 PC에는 따로 전달해야 합니다.

## 2. 원본 자료

`reference_data/` 안의 폴더입니다.

| 폴더 | 내용 | 크기 |
|---|---|---|
| `korea/` | 공공데이터포털 파일 25종. 파일명 끝에 날짜가 있으면 과거 판 (133개) | 37MB |
| `kosis/` | 국내 항공료, 노선별 운항과 여객, 환율, 유가, 물가 | 1MB |
| `weather_asos/` | 공항 기상. `all_2023~2026`은 미국 분석용, `asos_14airports`는 인천 등 14개 공항 | 827MB |
| `eurocontrol/` | 유럽 공항 교통량과 지연. **비상업 한정** | 355MB |
| `opensky_flightlist/` | 전 세계 항공편 목록 2019~2022. **공유 범위 제한** | 7.8GB |
| `opensky/` | 항공기 등록 정보. **OpenSky 약관 적용으로 간주** | 90MB |
| `ntsb/` | 미국 항공 사고 DB | 278MB |
| `ourairports/`, `openflights/`, `worldbank/` | 공항, 항공사, 노선, 국가별 항공 통계 | 22MB |

`bts_data/`는 1987-10~1989-12와 2000-01~2026-08만 있습니다. **1990~1999년은 없습니다.**

## 3. 화면 메뉴별 사용 자료

| 메뉴 | 폴더 | 원본 | 변환 자료 | 이 PC의 모델 결과 |
|---|---|---|---|---|
| 미국 운항 | `us_flights` | `bts_data`, `openflights` | `data/ontime_2020~`, `cache/bts_2020` | 8개 |
| 미국 기상과 운항 | `us_weather` | `bts_data`, `weather_asos/all_*` | `cache/us_weather_full` | 6개 |
| 한국 운항·지연 | `korea_ops` | `korea/인천공항_환승승객현황*`, `korea/한국공항공사_지연시간_학습데이터`, `weather_asos/asos_14airports` | 없음 | 12개 (저장 모델 8개) |
| 한국 수요·수송 | `korea_demand` | `korea/`의 운송실적, 제주 여객, 수송실적 | 없음 | 12개 |
| 한국 국내선 항공료 | `korea_airfare` | `kosis` | 없음 | 없음 |
| 유럽 공항 | `europe` | `eurocontrol` | `cache/euro_*` | 32개 |
| 전 세계 항공 | `world_traffic` | `opensky_flightlist` 2020~, `opensky`, `ourairports`, `openflights`, `worldbank` | `cache/opensky_2020` | 없음 |
| 항공 안전 | `safety` | `ntsb`, `korea/국토교통부_*사고*` | `cache/ntsb_*` | 4개 |

## 4. 인천공항 수집 자료

`data/icn_live/<날짜>/`에 쌓입니다. 학원 PC의 작업 스케줄러가 `collect_icn.py`를 현황은 10분마다, 운항마감은 1시간마다 실행하고, 1시간마다 `G:\내 드라이브\4ir\data\icn_live`로 백업합니다.

| 파일 | 내용 | 주의 |
|---|---|---|
| `arrivals.jsonl`, `departures.jsonl` | 도착, 출발 현황. 값이 바뀔 때만 한 줄 추가 | 시각에 날짜가 없고 공동운항편이 섞여 있음 |
| `closing.jsonl` | 운항마감과 지연 표시 | 실제 시각은 없음 |
| `runs.jsonl` (날짜 폴더 밖) | 모든 수집 실행 기록 | `--status`로 공백 확인 |

학습 자료와 지연 기준이 달라 보여서 아직 모델 입력으로 쓰지 않습니다. 자세한 내용은 [WORK_LOG.md](WORK_LOG.md)에 있습니다.

## 5. 최상위 파일

| 용도 | 파일 |
|---|---|
| 화면 | `app.py` |
| 자료 받기 | `download_bts.py`, `download_refs.py`, `download_kosis.py`, `download_weather.py`, `collect_icn.py` |
| 변환 | `prepare_data.py`, `preprocess.ipynb` |
| 통계와 학습 | `run_all.py`, `run_models.py`, `trial_model.py`, `colab_run.ipynb` |
| 평가와 추정 | `evaluate_icn.py`, `evaluate_icn_inputs.py`, `forecast_icn_monthly.py` |
| 점검 | `test_collect_icn.py`, `test_forecast_icn.py`, `test_saved_models.py`, `test_download_kosis.py` |
| 드라이브와 진행 확인 | `upload_to_drive.ps1`, `watch_training.ps1` |
| 예전 학습 | `train.py`, `train_enhanced.py` |
| 문서 | `DATA_GUIDE.md`, `DATA_SOURCES.md`, `WORK_LOG.md` |

## 6. 현재 코드가 쓰지 않는 자료

학습 범위가 2020년 이후라서 아래 자료는 지금 코드가 읽지 않습니다. 지우지 않았습니다. 디스크 여유가 882GB라 지울 필요도 없습니다.

| 자료 | 크기 | 비고 |
|---|---|---|
| `bts_data/ontime_1987~2019` | 5.9GB | `START_YEAR`를 낮추면 다시 필요 |
| `data/ontime_1987~2019` | 1.6GB | `START_YEAR`를 낮추면 다시 필요 |
| `data/cache/opensky_full/` | 4.7GB | `opensky_2020`으로 대체됨 |
| `reference_data/opensky_flightlist/flightlist_2019*` | 2.1GB | `START_YEAR`를 낮추면 다시 필요 |
| `_archive_before_2020/` | 234MB | 옛 결과 보관용 |
| `models/pilot/`, `models/recent_2023_2025/` | 8MB | 코드에서 읽는 곳을 찾지 못함 |
