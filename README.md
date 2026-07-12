# BARAM 2026 — 풍력 발전량 예측 (DACON 제3회)

[제3회 풍력발전량 예측 AI 경진대회 - BARAM 2026](https://dacon.io/competitions/official/236727) 참가 프로젝트.

태백 풍력단지 3개 KPX 그룹의 **2025년 시간별 발전량(kWh)** 을 기상 수치예보(NWP: GFS·LDAPS)만으로 day-ahead 예측한다.
예보는 전일 13:00 KST에 공개되며 리드타임 12~35시간. 설비용량은 그룹1/2 = 21,600, 그룹3 = 21,000 kWh/h.

---

## 1. 결과 요약

2024 holdout(자가검증) 공식 총점 궤적:

| 단계 | 총점 | 무엇을 더했나 |
|---|---:|---|
| GBM 기준선 (86 feats) | 0.6008 | LightGBM + 물리 파생 + 파워커브 |
| v2 | 0.6013 | 격자 공간 feature + 감률, 변수 86→65 |
| v2 + FICR 후처리 | 0.6266 | debias + FICR nudge |
| v3 | 0.6383 | + MLP 블렌드 (50:50) |
| v4 | 0.6463 | MLP random search 튜닝 (w=0.6) |
| v5 | 0.6461 | MLP **시드 5개 앙상블** (w=0.7) — **실제 제출: LB 0.61206** |
| v6 | 0.6273* | 후처리를 **보수적 nudge(P3c)** 로 교체 — **실제 제출: LB 0.6292 (212위)**, holdout↔LB 편차 +0.002로 캘리브레이션 확립 |
| v7 | 0.6351 | + **SCADA 라벨정제**(stuck 가중 0.5) + **GBM 튜닝** — **실제 제출: LB 0.63497 (139위)**, 캘리브레이션 재확인(편차 −0.0001) |
| v8 | 0.6438 | + FICR-정렬 손실 가중(α=5) + 시드 10 — **실제 제출: LB 0.62614 ❌ 실패**. 상향 편향(평균 예측 CF 41~43%)이 2025에서 처벌됨 — 2폴드 통과에도 LB 기각 |
| v9 | 0.6364 | v7 구성 + 시드 10 — **실제 제출: LB 0.63157** (1-nMAE 0.86958, FiCR 0.39356), 평균 확대 실패 |
| v10 | 0.6370 | v7 구조 고정 + OOF forecast-combination — **실제 제출: LB 0.63158** (1-nMAE 0.87056, FiCR 0.39260), v9와 사실상 동일 |
| v11 | 0.6395 | 평가 제외 행(`y<10%`)을 GBM 학습에서도 제외 + 기존 MLP 유지. 2023/2024 raw 모두 개선, 그룹별 LB probe 대기 |
| v12 | **0.6410** | v11 + 수정된 공식 기대효용 argmax — **실제 제출: LB 0.63858 (95위)**, 1-nMAE 0.87309·FiCR 0.40408 |
| v13 | **0.6418** | v12의 그룹3 LightGBM만 엄격 검증된 pooled LightGBM으로 교체 — **제출 전 후보**, 2024 v12 대비 +0.00089 |
| v14 | **0.6433** | 그룹3의 약한 HistGBM도 제거해 pooled LightGBM 40% + MLP 60% — **새 제출 후보**, v12 대비 +0.00239 |
| v16 | **0.6442 환산** | 그룹3 LightGBM 비중을 날씨별 25~55%로 정하는 제한형 gate — **새 제출 후보**, v14 대비 +0.00088 |

(*v6 holdout이 v5보다 낮은 건 정상 — v5의 홀드아웃 0.6461은 2024에 과적합된 낙관치였음이 실전에서 판명. 단계별 정량 로그와 노트북은 각 `submission/ver_{num}` 폴더에 보존)

**⚠️ v5 실전 결과와 교훈 (2026-07-11)**: v5 제출 → **LB 0.61206** (1-NMAE 0.85656, FICR 0.36756). 홀드아웃 대비 −0.034 하락. 원인 확정(로컬 `DIAGNOSIS_LB.ipynb`): 후처리(debias+nudge scale≤1.15)를 **2024 holdout 한 해만 보고 선택**한 것이 연도 과적합 — 같은 후처리가 2023 폴드에선 총점을 −0.07 붕괴시키는 도박이었음(예측을 실측 32%인데 47~50%로 상향). 모델 자체(GBM⊕MLP)의 원본 예측은 잘 캘리브레이션돼 있었고, CatBoost 추가·교체는 이득 없음(Δ≈0).

**현재 챔피언**: **[`submission/ver_12/submission.csv`](submission/ver_12/submission.csv)** (LB 0.63858, 95위). v7 대비 총점 +0.00360, 1-nMAE +0.00217, FiCR +0.00505.

**제출 권장 후보**: **[`submission/ver_16/submission.csv`](submission/ver_16/submission.csv)**. 2023 blocked OOF로만 학습한 제한형 gate가 완전히 미래인 2024 그룹3에서 v14 대비 +0.00265를 기록해 전체 환산 +0.00088을 더했다. 1-NMAE·FiCR가 동시에 개선됐고 분기 3/4·월 8/12에서 우위였다. 그룹1/2 제출 열은 Public 검증된 v12와 정확히 동일하다. LB 확인 전까지 챔피언 표기는 v12로 유지한다.

**⚠️ v8 교훈 (2026-07-12)**: FICR-정렬 가중은 2폴드를 모두 통과하고도 LB에서 −0.009. v5와 동일 패턴 — **평균 예측 CF를 실측(~30%)보다 크게 올리는 변경은 미래 연도의 예측 불가 정지·제한에 처벌당한다**. 이후 채택 규율에 **편향 가드(평균 예측 CF ≤ 39.5%)** 추가.

참고: 리더보드 1등 0.669, 상위권 1-NMAE ~0.87 포화 → **순위는 FICR에서 갈린다**. 우리 팀 "짱승": v5 328위 → v6 212위 → v7 139위 → v12 95위.

## 2. 평가 지표

```
총점 = 0.5 × (1 − NMAE) + 0.5 × FICR
```

- **NMAE**: 그룹별 유효시간 절대오차율(|오차|/설비용량) 평균의 3그룹 평균
- **FICR**: 그룹별 `Σ(실측×단가) / Σ(실측×4)`의 3그룹 평균 — **발전량 가중 2티어**. 단가: 오차율 ≤6% → 4.0, ≤8% → 3.0, 초과 → 0
- **유효시간**: 실측 발전량 ≥ 설비용량의 10%인 시간만 채점

구현은 [`official_metric.py`](official_metric.py) (대회 제공 코드 그대로 보존). FICR이 발전량 가중이므로 **출력이 큰 시간을 6% 오차 안에 넣는 것**이 점수의 핵심 지렛대다.

## 3. 데이터

```
~/Downloads/open/          # 대회 원본 (repo 밖, gitignore) — CSV: GFS/LDAPS train·test, 라벨, SCADA, info.xlsx
repo/preprocessed/         # 시간별 wide 테이블 (그룹별 train/test parquet) + spatial_v2 feature
```

- `preprocessed/train_*.parquet`, `test_*.parquet` — 기존 전처리. **원본 대조 검증 완료(누설 없음, 모든 값 오차 0 재현)**. 역추적된 레시피: GFS = 최근접 1격자(grid 5), LDAPS = 그룹별 2~3격자 mean/std/min/max. 상세: [`document/PREPROCESSING_VERIFICATION.md`](document/PREPROCESSING_VERIFICATION.md)
- `preprocessed/spatial_v2_{train,test}.parquet` — v2에서 추가한 격자 공간 feature(GFS 9격자·LDAPS 16격자의 mean/std/동서·남북 gradient) + 기온 감률 2종(t850−t700, t2m−t850). 생성: [`scripts/build_spatial_v2.py`](scripts/build_spatial_v2.py) (원본 CSV 필요)
- 그룹3 라벨은 2023~2024만 존재(2022 공백). 그룹3 타깃은 parquet에 문자열로 저장돼 있어 로더에서 숫자 변환(처리는 `wind_lib.load_train`)

## 4. 환경 & 실행

Python 3.11 + [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra notebook                   # 실행 + 노트북 의존성 설치 (.venv)
# 노트북 실행 (torch를 쓰는 노트북은 --with torch + OMP 가드)
OMP_NUM_THREADS=1 uv run --with nbconvert --with ipykernel --with torch \
  jupyter nbconvert --to notebook --execute --inplace submission/ver_9/pipeline.ipynb
```

⚠️ **macOS 필수 주의 — torch + LightGBM 세그폴트**: `import torch` 후 `lightgbm.fit()`을 같은 프로세스에서 호출하면 libomp 중복 로드로 segfault(exit 139)가 난다. 해당 노트북들은 첫 셀에서 다음 3종 가드를 건다 (절대 제거 금지):
`os.environ["OMP_NUM_THREADS"]="1"` · `torch.set_num_threads(1)` · LightGBM `n_jobs=1`. MLP 학습은 MPS(Apple GPU)를 사용한다.

## 5. Repo 구조

```text
wind_lib.py                 # 데이터 로더·물리/공간 feature
wind_pipeline.py            # v10+ 공용 GBM/MLP 학습 기반(지연 로딩)
official_metric.py          # 대회 공식 지표
submission_validation.py    # 제출 스키마·범위 검증 및 atomic 저장
scripts/build_spatial_v2.py # 원본 NWP → spatial parquet
submission/
  registry.json             # 버전별 상태·점수·제출 경로
  ver_1/ ... ver_16/        # 해당 버전의 CSV·JSON·코드·노트북·리서치
tests/                       # 지표·데이터 join·제출 구조 회귀 테스트
```

버전별 상세 파일과 판정은 [`submission/README.md`](submission/README.md)와
[`submission/registry.json`](submission/registry.json)을 기준으로 한다. 기각된 HMM과 시간
이웃 feature도 공용 코드에서 제거해 각각 `ver_1`, `ver_8`의 legacy 모듈에 보존했다.

## 6. 현재 파이프라인 (v12 챔피언 + v16 후보)

```
feature 65개 = 원 NWP lean(트리무의미·죽은변수·중복 31개 제거)
             + 물리 파생(전단, 파워로 α, 공기밀도, ρv³, gust비, GFS·LDAPS 차이, 풍향 sin/cos, 리드타임)
             + 파워커브 pc_pred_cf (isotonic, 학습구간 fit)
             + 격자 공간 mean/std/gradient 8개 + 감률 2개   ← wind_lib.lean_features() + SPATIAL_COLS

학습 가중 = MLP는 stuck(SCADA 식별 고장·제한 시간 0.5)
           GBM은 공식 평가 제외 행(CF<10%) 가중 0

v12 = 0.4 × GBM(LightGBM + HistGBM) + 0.6 × pooled MLP(시드 3)
      + 조건부 quantile 기반 공식 기대효용 action

v16 = 그룹1/2는 v12 바이트 그대로 유지
      그룹3은 pooled LightGBM과 pooled MLP를 날씨별 constrained gate로 결합

후처리 = 두 연도 모두 손해 없는 보수적 nudge + 평균 CF 39.5% guard
```

## 7. 검증 규율 & 누설 방지

- **랜덤 분할 금지** (강한 계절성). expanding-window CV: 그룹1/2 = [2022→2023, 2022-23→2024], 그룹3 = [2023→2024]
- 기법 채택 기준: **2023·2024 두 폴드 모두 우위**일 때만 (단일 holdout의 노이즈 차단)
- 파워커브·nudge·MLP 표준화 — 전부 **해당 폴드 학습구간에서만 fit**
- 테스트 입력은 NWP뿐. SCADA·재분석·사후보정 금지. 상세: [`document/CONSTRAINTS_주의점.md`](document/CONSTRAINTS_주의점.md)

## 8. 실험 이력 — 채택/기각과 근거

| 실험 | 판정 | 근거 |
|---|---|---|
| 격자 공간 feature (9+16격자 통계·gradient) | ✅ 채택 | 문헌(Andrade & Bessa 2017, −12.85%)과 방향 일치, CV 두 해 모두 우위 |
| 기온 감률 2종 (안정도) | ✅ 채택 | feature 중요도 전체 4·5위 |
| 변수 축소 86→65 | ✅ 채택 | 축소분(−31)이 죽은 변수임을 CV로 확인 |
| MLP 블렌드 | ✅ 채택 | 20 trial 중 17개가 두 해 모두 GBM 우위 |
| FICR 후처리 — 기존(debias+nudge≤1.15, 2024만 검증) | ❌ **실전 실패** | LB에서 FICR −0.053. 2023 폴드 −0.07 붕괴 확인 → v6에서 교체 |
| FICR 후처리 — **보수적 nudge(P3c, ≤1.05)** | ✅ 채택(v6) | 2023·2024 두 해 모두 우위, worst-year 최강. **LB 실측 +0.017 회복(0.612→0.629)** |
| 구간별 nudge | ❌ 기각 | 연도 부호 뒤집힘 |
| quantile FICR 점추정(C2) | ❌ **실험 무효** | capacity 중복 나눗셈으로 FiCR 항이 약 21,000배 축소됨. v12에서 수정 재실험 |
| **SCADA 라벨 정제** (stuck 시간 가중 0.5) | ✅ 채택(v7) | 두 폴드 +0.008~0.011 — 전 변형이 두 해 모두 우위(`submission/ver_7/scada_clean_result.json`). g1/g2의 18%가 stuck |
| **GBM 하이퍼파라미터 튜닝** (lr0.021·트리2000·mcs300) | ✅ 채택(v7) | 두 폴드 +0.003~0.004 (`submission/ver_7/gbm_hpo_result.json`) |
| FICR-정렬 손실 가중 (α=5) | ❌ **LB 기각**(v8) | 2폴드 단조 개선했으나 LB −0.009 — 상향 편향의 연도 이전 실패. 편향 가드 신설 계기 |
| OOF forecast-combination(v10) | ❌ **LB 무효** | v9 대비 총점 +0.000013. 1-nMAE 이득을 FiCR 하락이 정확히 상쇄 |
| **10% 유효시간 정렬(v11)** | △ LB 대기 | GBM 저발전 행 가중 0. LightGBM 두 해 단조 개선, 전체 raw 2023 +0.0058·2024 +0.0082 |
| **수정 기대효용 점추정(v12)** | ✅ **Public 채택** | LB 0.63858(95위). v7 대비 1-nMAE +0.00217·FiCR +0.00505로 두 항 모두 전이 |
| **압축 공간 feature(IDW/PCA)** | ❌ 기각 | 연도 안전 조건 실패. 그룹1/2는 연도별 부호가 뒤집히거나 악화, 그룹3도 pooled보다 약함 (`submission/ver_13/spatial_backtest.json`) |
| **그룹3 pooled LightGBM(v13)** | ✅ **제출 후보** | 6/6 순차 분기 개선(평균 +0.01423), 2024 월 9/12 개선. 최종 v12 결합 후 전체 +0.00089, 1-nMAE·FiCR 동시 개선 |
| **그룹3 HistGBM 제거(v14)** | ✅ **새 제출 후보** | pooled-only GBM이 순차 분기 5/6·월 10/12 개선. 최종 v12 대비 +0.00239, v13 대비 +0.00150 |
| 그룹3 MLP 60%→50%(v15) | ❌ 기각 | 점예측 검증은 통과했지만 기대효용 후 v14 대비 +0.000004뿐. 1-nMAE 이득을 FiCR 하락이 상쇄 |
| **그룹3 제한형 gating(v16)** | ✅ **새 제출 후보** | 2023 OOF 학습→2024 완전 홀드아웃. 그룹3 +0.00265, 전체 환산 +0.00088, 1-nMAE·FiCR 동시 개선 |
| 신경망 bake-off (ResNet-MLP·1D-CNN·혼합) | ❌ 기각 | 6변형 전부 현행 MLP에 패배 (`submission/ver_8/bakeoff_result.json`) |
| 시간 이웃 NWP feature (배치 내 lag/lead 12개) | ❌ 기각 | 두 폴드 모두 악화 (`submission/ver_8/tempfeat_result.json`) |
| SCADA 임계·가중 미세화 / 블렌드 재스캔 | ❌ 기각 | v7 설정이 평탄 최적점 (`submission/ver_7/tune2_result.json`) |
| MLP 재튜닝(α 가중 하) | ❌ 기각 | 12 trial 전패 — 현행 설정 유지 |
| CatBoost 추가/교체 | ❌ 기각 | 두 해 모두 Δ≈0 — GBM 계열 내 교체는 무효 |
| 시드 앙상블 | ✅ 채택(v5) | 단일시드 CV는 시드운 포함 낙관치 — 앙상블이 실전 기대값 우위 |
| HistGBM 단순 추가 | △ 유지 | 이득 미미하나 전 폴드 일관 (−0.02~−0.06%p) |
| GRU 잔차보정 | ❌ 기각 | +0.55~0.74%p **악화** — 시퀀스 1,400개는 부족 |
| HMM 국면(regime) feature | ❌ 기각 | 폴드 간 부호 뒤집힘 = 노이즈 |
| FiLM / 계층 헤드 조건화 | ❌ 기각 | 두 해 모두 concat 임베딩에 패배 |
| MOS(NWP 보정) | ❌ 미사용 | 규칙 해석 리스크 + GBM이 흡수 (팀 결정, CONSTRAINTS §3) |

(각 판정의 상세 수치와 노트북은 해당 `submission/ver_{num}` 폴더에 기록)

> 교훈: 이 데이터 규모(그룹당 1.7~2.6만 행)에서는 **단순한 것(물리 feature·격자 통계·소형 MLP·후처리)이 이기고, 구조적 복잡성(GRU·FiLM·attention류)은 일관되게 진다.**

## 9. 재현 순서

```bash
# 1. 대회 원본을 ~/Downloads/open 에 배치
#    다른 위치라면 WIND_RAW_DIR=/path/to/open 설정
# 2. 의존성
uv sync --extra notebook
# 3. (spatial parquet 재생성이 필요할 때만)
uv run python scripts/build_spatial_v2.py
# 4. v9 제출 재현 → submission/ver_9/submission.csv
OMP_NUM_THREADS=1 uv run --with nbconvert --with ipykernel --with torch \
  jupyter nbconvert --to notebook --execute --inplace submission/ver_9/pipeline.ipynb

# 5. v10 이력 재현 → submission/ver_10/submission.csv
uv run python -m submission.ver_10.pipeline

# 6. v11 anchor 생성 → submission/ver_11/submission.csv
uv run python -m submission.ver_11.pipeline

# 7. v12 챔피언 생성 → submission/ver_12/submission.csv
uv run python -m submission.ver_12.pipeline

# 8. v13 후보 검증 및 생성
uv run python -m submission.ver_13.backtest
uv run python -m submission.ver_13.pipeline

# 9. v14 후보 검증 및 생성
uv run python -m submission.ver_14.backtest
uv run python -m submission.ver_14.pipeline

# 10. v16 제한형 gate 검증 및 생성 → submission/ver_16/submission.csv
uv run python -m submission.ver_16.backtest
uv run python -m submission.ver_16.pipeline

# 11. 구조·지표·제출물 회귀 테스트
uv run python -m unittest discover -s tests -v
```

## 10. 남은 작업

1. v12를 Public 챔피언으로 유지한 채 v16을 제출하고 실제 LB 전이를 확인
2. action strength/nudge의 선택 점수와 최종 점수를 분리하는 strict/nested 검증 추가
3. 문서상 SCADA 파워커브와 현행 NWP→KPX isotonic feature의 불일치 검증
