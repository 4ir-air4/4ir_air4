# 최효선 (chs) 작업

최효선님이 Google Colab에서 작성한 분석 노트북입니다. 팀 공유 드라이브 `4ir\chs\4ir_air4`에 있던 파일 가운데 본인이 작성한 것만 옮겼습니다(2026-11-17).
같은 폴더에 있던 팀 저장소 사본(`app.py`, `fetch_*.py`, `data/` 등), 가상환경, `.git`은 넣지 않았습니다.

| 파일 | 내용 |
|---|---|
| `notebooks/Untitled6.ipynb`, `Untitled6 (1).ipynb` | 결항 예측 연습용 자료 만들기(코드 3셀). 드라이브에 두 판이 있어 둘 다 두었습니다 |
| `notebooks/Untitled7.ipynb`, `Untitled7 (1).ipynb` | 공공데이터 오픈 API 조회 틀과 운항·기상 자료 결합 후 분류 모델(코드 3셀). 두 판 모두 둠 |
| `notebooks/Untitled8.ipynb` | 결항 예측: 자료 탐색, 전처리, 머신러닝 모델, 과거 기상 평년값 탐색 |
| `notebooks/Untitled9.ipynb` | 운항 지연 예측과 동적 임계값 제어(LightGBM, SMOTE, 비용 민감 학습, Optuna, ROI 계산) |
| `notebooks/untitled9 (1).py` | Untitled9를 Colab에서 내려받은 파이썬 파일(조금 더 긴 판) |
| `notebooks/Untitled10.ipynb` | 월별 국내 항공 여객 수요 예측(GRU, Stacked LSTM, BiLSTM-Attention, Optuna), 2026~2030 시나리오, 기상·요인별 지연·결항 원인 통계 |

## 실행

- Colab에서 만든 노트북이라 입력 파일은 Colab에 직접 올려 읽습니다. 자료 파일은 저장소에 없습니다. 출처는 공공데이터포털(한국공항공사 결항·지연 통계 등), 기상청, KOSIS입니다.
- 공공데이터포털 인증키는 Colab 비밀값(`PUBLIC_DATA_KEY`)으로 읽습니다. 노트북에 키 값은 들어 있지 않습니다.
- `untitled9 (1).py` 머리말의 Colab 원본 주소 한 줄은 공개 저장소라서 지웠습니다. 그 밖의 내용은 원본과 같습니다.

## 팀 앱 반영

- 한국 운항·지연 화면의 지연·결항 원인 그래프는 이 작업의 아이디어를 팀 자료로 다시 만든 것입니다.
- Untitled6~9의 결항 예측 정답은 연습용으로 만든 값이라, 노트북의 성능 숫자는 실제 성능으로 쓰지 않습니다.
