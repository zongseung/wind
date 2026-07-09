# preprocessed/ 검증 리포트 — 원본(open/) 대조

검증일: 2026-07-09 · 원본: `/Users/ijongseung/Downloads/open` · 대상: `preprocessed/*.parquet`

## 결론 한 줄
**누설 없음·값 정확 (모든 항목 오차 0)** — 단, 격자와 변수를 상당히 버린 "최소" 전처리라서 개선 여지가 큼.

## 역추적된 전처리 레시피 (오차 1e-6 미만으로 정확 재현)

| 항목 | 방식 |
|---|---|
| 시간 정렬 | `kst_dtm` = `forecast_kst_dtm`. 원본은 전일 13:00 공개(`data_available`) 예보만 포함, 리드 12~35h. **누설 불가능 구조** |
| GFS 집계 | **최근접 1개 격자(grid 5)** 값 그대로. 3그룹 모두 동일. `_mean` 접미사는 이름만 mean |
| LDAPS 집계 | group1 = 3격자 (5,6,10) / group2 = 2격자 (6,11) / group3 = 2격자 (6,12). 통계 = mean·std(ddof=1)·min·max |
| 풍속 파생 | ws = √(u²+v²) 후 격자 집계 |
| 라벨 | `train_labels.csv`와 완전 일치 (그룹3 2022 공백 포함) |
| test | `sample_submission.csv`의 forecast_id·시각과 1:1 일치 (8,760행) |

## 검증 항목별 결과

| 검증 | 결과 |
|---|---|
| 예보시각당 availability 고유값 | 1 (혼합 없음) ✅ |
| 리드타임 범위 (train/test 동일) | 12~35h ✅ |
| GFS ws100 재현 (3그룹, train+test) | 오차 0 ✅ |
| LDAPS ws10 mean/std 재현 (3그룹) | 오차 0 ✅ |
| 라벨 3그룹 | 오차 0 ✅ |
| test 행 정합 | 8,760행, 시각 일치 ✅ |

## 버려진 정보 (개선 기회)

1. **GFS 9격자 중 8개 미사용** — 공간 gradient·풍상(upwind) 정보 소실. GFS는 공간통계 자체가 없음.
2. **LDAPS 16격자 중 13~14개 미사용**.
3. **원본 변수 누락**: GFS 500hPa 전체(u·v·gh·t), 850/700hPa 기온(→대기 안정도·감률 계산 불가), 850hPa 습도, LDAPS 5m 경계층풍(XBLWS/YBLWS). *(복사·적설·lsm·지형고도는 풍력에 무관하니 버린 게 타당)*
4. SCADA(VESTAS 12기·UNISON 5기 10분 데이터)는 파이프라인에서 전혀 미사용 (파워커브·라벨정제 잠재 자산).

## 시사점
- 기존 preprocessed로 만든 모든 결과(모델·CV·FICR)는 **누설 걱정 없이 유효**.
- 상위팀 1-NMAE가 ~0.87에서 포화 → 남은 차이는 이런 **버려진 공간·상층 정보**에서 나올 가능성. 전처리 보강(격자 확장 + 누락 변수 복구)이 다음 우선순위.

## 참고: `wind_lib.lead_h` 주석
리드타임 feature는 09:00 초기화 기준 16~39h로 명명했으나 실제 availability(13:00) 기준으론 12~35h. **hour와 1:1 단조 대응이라 모델에는 동일** — 이름 차이일 뿐 버그 아님.
