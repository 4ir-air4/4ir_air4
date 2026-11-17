# 공항 안내 outputs 전달 자료

이 폴더는 기존 조회·정제·여객 스냅샷 선택 함수를 실행한 저장 결과입니다. 원본 CSV/수집기/AI 모델을 읽거나 실행할 필요 없이 팀 화면에서 이 결과만 읽습니다. ZIP 내부 outputs/airport_guidance/<생성번호>/ 구조를 유지해 별도 경로에 풀어 주세요. 기존 결과를 덮어쓰지 마세요.

## 파일·컬럼
- passengers.json: metadata와 records. ARP/airport_code 공항, airport_name 이름, SDT 자료 날짜(YYYYMMDD), HH 원본 시간 구분(문자열), AOD D출발/A도착, TOF D국내/I국제, PCT 예상 인원, PCG 별도 원본값, CONGEST_YN 제공기관 구분값. airport_verified/operation_date_verified/semantics_verified 원래 검증 상태, collected_at 수집 시각, snapshot_sha256/sources 출처, completeness API 응답 건수 검증. 업무 키 ARP+SDT+HH+AOD+TOF.
- passengers.csv: 같은 여객 행의 편의용 표. sources는 JSON 문자열. JSON과 CSV를 함께 합산하지 마세요.
- weather.json: airports[공항코드]가 기존 공개 조회 응답. records에 observed_at 실제 관측 시각, collected_at 수집 시각, temperature/humidity/wind_speed_10m/wind_dir_10m/l_vis/r_vis/precipitation. 재보정하지 않았으며 결측은 null. available/status/reasons/quality와 mapping_verified 등 반환 플래그 유지.
- flights.json: airports[공항코드]가 기존 공개 조회 응답. records에 airFln 편명, io O출발/I도착, std 예정·etd 변경 HHMM 문자열, rmkKor/rmkEng 제공기관 운항 상태, collected_at 수집 시각 및 운항일자·매핑 검증 상태. 여객 AOD 코드와 운항 io 코드는 다릅니다.
- quality.json: 공항/방향/국내외별 자료 미제공·검증 제외 사유, 이력 검증 결과. unavailable 공항은 빈 결과와 사유를 표시하고 다른 공항 값으로 대체하지 마세요.
- manifest.json: 파일 SHA-256/크기, 자료 건수. manifest 자신은 해시 목록에서 제외합니다.

## 팀 화면에서 읽기
```python
import json
from pathlib import Path
import streamlit as st
# 아래 경로의 생성번호를 전달받은 폴더명으로 지정 (사용자 입력 경로 금지)
base = Path(__file__).resolve().parents[1] / 'outputs' / 'airport_guidance' / '20261117_122311'
try:
    data = json.loads((base/'passengers.json').read_text(encoding='utf-8'))
    rows = [r for r in data['records'] if r['ARP']=='GMP' and r['AOD']=='D' and r['TOF']=='D']
    # 날짜/HH 선택지는 rows에 실제 존재하는 값으로 구성
    st.dataframe(rows)
    st.caption('PCT는 예상 여객 수, Y/N은 제공기관 구분값입니다.')
    weather = json.loads((base/'weather.json').read_text(encoding='utf-8'))['airports']['GMP']
    if weather['available']:
        st.write(weather['records'])
    else:
        st.info('기상 자료 없음 또는 조회 실패: '+', '.join(weather['reasons']))
except (OSError, ValueError, KeyError):
    st.error('결과 파일을 읽을 수 없습니다. 이전 결과를 최신 자료로 표시하지 않습니다.')
```
실제 적용 시 manifest의 해시와 전체 파일 존재를 확인한 뒤 화면에 게시하세요. 위 예시는 읽기 방법이며 운영 어댑터 구현/테스트를 뜻하지 않습니다. CSV를 읽으면 SDT/HH/std/etd는 문자열로 지정하세요.

## 의미·갱신 주의
전체 지원 공항을 조회했으며 8507이 제공하는 국내·국제(D/I), 출발·도착 범위를 사용했습니다. 공항별 적격 스냅샷 하나를 날짜별 선택하고 그 스냅샷의 원본 행만 사용했습니다. 미검증 공항과 미제공 날짜/HH는 임의로 채우지 않았습니다. 과거조회 분리 폴더의 자료는 8507 기본 이력 범위가 아니므로 추가하지 않았습니다.
PCT는 실제 탑승 인원·실측 혼잡·대기시간이 아닙니다. PCG와 합산하지 않습니다. CONGEST_YN은 제공기관 혼잡 구분값이며 AI 예측이 아닙니다. HH의 KST 시간 구간 의미는 미검증입니다. API 전체 응답 건수 검증은 하루 24시간 완전성을 보장하지 않습니다.
기상·운항은 여객 SDT에 대응하는 과거 자료가 아니라 별도 최신 조회 결과입니다. 관측 시각/수집 시각/여객 SDT/내보내기 시각을 따로 표시하세요. 운항 수집일을 운항일자로 쓰거나 std-etd 차이로 실제 지연을 만들지 마세요. 세 자료의 원자적 동일 시점 스냅샷은 보장하지 않습니다.
정적 내보내기이므로 파일을 새로 읽어도 API나 원본 데이터가 갱신되지 않습니다. data_age는 조회 당시 초 단위 값이므로 현재 경과 시간은 기준 시각에서 다시 계산하세요. 새 결과는 별도 생성번호 폴더 전체를 전달하고 누락/손상 시 이전 결과를 정상 최신 결과로 대체하지 마세요.
service_validated=false, reliable_generalization=false, temporal_availability=NOT_VERIFIED 유지. 이번 패키지에는 AI 모델/학습/평가 결과가 없습니다.
