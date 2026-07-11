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
| v9 | 0.6364 | v7 구성 그대로 + 시드 10 (α 없음, 편향 프로파일 v7과 동일). holdout +0.0013 — 미세·무위험 후보 |

(*v6 holdout이 v5보다 낮은 건 정상 — v5의 홀드아웃 0.6461은 2024에 과적합된 낙관치였음이 실전에서 판명. 단계별 정량 로그 `*_summary.json`·실험 노트북은 로컬 전용)

**⚠️ v5 실전 결과와 교훈 (2026-07-11)**: v5 제출 → **LB 0.61206** (1-NMAE 0.85656, FICR 0.36756). 홀드아웃 대비 −0.034 하락. 원인 확정(로컬 `DIAGNOSIS_LB.ipynb`): 후처리(debias+nudge scale≤1.15)를 **2024 holdout 한 해만 보고 선택**한 것이 연도 과적합 — 같은 후처리가 2023 폴드에선 총점을 −0.07 붕괴시키는 도박이었음(예측을 실측 32%인데 47~50%로 상향). 모델 자체(GBM⊕MLP)의 원본 예측은 잘 캘리브레이션돼 있었고, CatBoost 추가·교체는 이득 없음(Δ≈0).

**최종 제출 후보(챔피언)**: **`submission_v7.csv`** (LB 실측 0.63497, 139위 — DACON 제출선택은 v7로). v8은 실패 기록으로 보존.

**⚠️ v8 교훈 (2026-07-12)**: FICR-정렬 가중은 2폴드를 모두 통과하고도 LB에서 −0.009. v5와 동일 패턴 — **평균 예측 CF를 실측(~30%)보다 크게 올리는 변경은 미래 연도의 예측 불가 정지·제한에 처벌당한다**. 이후 채택 규율에 **편향 가드(평균 예측 CF ≤ 38%)** 추가.

참고: 리더보드 1등 0.669, 상위권 1-NMAE ~0.87 포화 → **순위는 FICR에서 갈린다**. 우리 팀 "짱씅": v5 328위 → v6 212위 → v7 139위 (1,332팀).

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
uv sync                                    # 의존성 설치 (.venv)
# 노트북 실행 (torch를 쓰는 노트북은 --with torch + OMP 가드)
OMP_NUM_THREADS=1 uv run --with nbconvert --with ipykernel --with torch \
  jupyter nbconvert --to notebook --execute --inplace PIPELINE_FINAL.ipynb
```

⚠️ **macOS 필수 주의 — torch + LightGBM 세그폴트**: `import torch` 후 `lightgbm.fit()`을 같은 프로세스에서 호출하면 libomp 중복 로드로 segfault(exit 139)가 난다. 해당 노트북들은 첫 셀에서 다음 3종 가드를 건다 (절대 제거 금지):
`os.environ["OMP_NUM_THREADS"]="1"` · `torch.set_num_threads(1)` · LightGBM `n_jobs=1`. MLP 학습은 MPS(Apple GPU)를 사용한다.

## 5. Repo 구조

### 추적되는 핵심 파일
| 파일 | 역할 |
|---|---|
| **`PIPELINE_FINAL.ipynb`** | **최종(현재 v8) 파이프라인 end-to-end 재현 (유일하게 추적되는 노트북)** |
| `wind_lib.py` | 로더·물리 파생·파워커브·유효구간 NMAE·spatial v2 조인·lean feature 세트·HMM(기각됨) |
| `official_metric.py` | 대회 공식 채점 코드 (총점/1-NMAE/FICR) |
| `scripts/build_spatial_v2.py` | 원본 CSV → spatial_v2 parquet 생성 |
| `submission_v4~v8.csv` | 제출 이력·후보 (그 외 산출물은 미추적) |

### 실험 노트북 (로컬 전용, git 미추적 — 실험 이력 기록)
| 순서 | 노트북 | 내용 | 결론 |
|---|---|---|---|
| 1 | `EDA_MAIN.ipynb` | 분포·상관·VIF·파워커브 EDA | 풍속이 지배변수, 트리모델 적합 |
| 2 | `CORE_FEATURE_EDA.ipynb` | 70→40 feature 축소 실험 | wind_uv 40개로 성능 유지 |
| 3 | `EDA_SUPPLEMENT.ipynb` | 리드타임·연도 분포이동·무발전 하한·풍향 교란 | 유효구간 지표 정합 확인 |
| 4 | `MODELING.ipynb` | LightGBM 기준선 + 첫 제출 | pooled가 그룹3에 유효 |
| 5 | `MODELING_ADVANCED.ipynb` | HistGBM 앙상블·HMM 국면·GRU 잔차 ablation | GRU 명확 기각 |
| 6 | `MODELING_CV.ipynb` | expanding-window CV로 판정 견고화 | 앙상블 채택, HMM 기각 |
| 7 | `MODELING_FICR.ipynb` | 공식지표 기반 FICR 후처리 4종 비교 | debias+nudge 채택 |
| 8 | `FEATURE_REDUCTION.ipynb` | 변수 축소 ablation (86→55) | 축소만으론 손해 |
| 9 | `MODELING_V2.ipynb` | 격자 공간 feature + 감률 (65개) | **v2 채택** — 두 해 모두 우위 |
| 10 | `MODELING_MLP.ipynb` | MLP 앙상블 멤버 실험 | 블렌드 채택 (B50) |
| 11 | `MODELING_MLP_TUNED.ipynb` | MLP random search 20 trial | trial 15, w=0.6 → **v4** |
| 12 | `MODELING_V3_FINAL.ipynb` | v3 최종 파이프라인 | `submission_v3.csv` |
| 13 | `MODELING_FILM.ipynb` | FiLM·계층헤드 조건화 실험 | **기각** — concat 유지 |
| 14 | `MODELING_SEED_ENS.ipynb` / `_FINAL.ipynb` | 시드 5개 앙상블 → 최종 | **v5** = 권장 제출 |

### 문서
| 파일 | 내용 |
|---|---|
| `document/baram2026_project_plan.md` | 초기 기획서 (데이터 현황·단계별 전략) |
| `document/CONSTRAINTS_주의점.md` | 규칙·누설 체크리스트 (테스트 입력 = NWP only) |
| `document/PREPROCESSING_VERIFICATION.md` | 전처리 원본 대조 검증 리포트 |
| `document/research_conditioning_layers_2026-07-08.md` | 조건화 계층(FiLM/AdaLN) 리서치 |
| `claudedocs/research_nwp_features_2026-07-09.md` | NWP feature 근거 리서치 (격자 ★★★·감률 ★★☆·500hPa 제외) |

실험별 정량 결과는 `*_summary.json`(로컬 전용)에 기록되어 있으며, 핵심 수치는 이 README의 §1·§8 표에 옮겨져 있다.

## 6. 최종 파이프라인 (v8)

```
feature 65개 = 원 NWP lean(트리무의미·죽은변수·중복 31개 제거)
             + 물리 파생(전단, 파워로 α, 공기밀도, ρv³, gust비, GFS·LDAPS 차이, 풍향 sin/cos, 리드타임)
             + 파워커브 pc_pred_cf (isotonic, 학습구간 fit)
             + 격자 공간 mean/std/gradient 8개 + 감률 2개   ← wind_lib.lean_features() + SPATIAL_COLS

학습 가중 = stuck(SCADA 식별 고장·제한 시간 0.5) × FICR-정렬 (1 + 5·y/용량)

모델 = 0.3 × GBM(튜닝 LightGBM + HistGBM 평균, MAE 손실, 가중)
     + 0.7 × MLP(256×3 GELU, 그룹 concat 임베딩, 3그룹 pooled, 가중 L1, 시드 10 평균, MPS)

후처리(그룹별, 학습기간 OOF로만 fit) = 보수적 nudge(scale≤1.05·shift≤2%) → [0, 설비용량] 클리핑
```

## 7. 검증 규율 & 누설 방지

- **랜덤 분할 금지** (강한 계절성). expanding-window CV: 그룹1/2 = [2022→2023, 2022-23→2024], 그룹3 = [2023→2024]
- 기법 채택 기준: **2023·2024 두 폴드 모두 우위**일 때만 (단일 holdout의 노이즈 차단)
- 파워커브·HMM·debias·nudge·MLP 표준화 — 전부 **해당 폴드 학습구간에서만 fit**
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
| 구간별 nudge / quantile FICR-지향 점추정 | ❌ 기각 | 연도 부호 뒤집힘 / 두 해 모두 악화 — 후처리 레버 소진 확인 |
| **SCADA 라벨 정제** (stuck 시간 가중 0.5) | ✅ 채택(v7) | 두 폴드 +0.008~0.011 — 전 변형이 두 해 모두 우위(`scada_clean_result.json`). g1/g2의 18%가 stuck |
| **GBM 하이퍼파라미터 튜닝** (lr0.021·트리2000·mcs300) | ✅ 채택(v7) | 두 폴드 +0.003~0.004 (`gbm_hpo_result.json`) |
| FICR-정렬 손실 가중 (α=5) | ❌ **LB 기각**(v8) | 2폴드 단조 개선했으나 LB −0.009 — 상향 편향의 연도 이전 실패. 편향 가드 신설 계기 |
| 신경망 bake-off (ResNet-MLP·1D-CNN·혼합) | ❌ 기각 | 6변형 전부 현행 MLP에 패배 (`bakeoff_result.json`) |
| 시간 이웃 NWP feature (배치 내 lag/lead 12개) | ❌ 기각 | 두 폴드 모두 악화 (`tempfeat_result.json`) |
| SCADA 임계·가중 미세화 / 블렌드 재스캔 | ❌ 기각 | v7 설정이 평탄 최적점 (`tune2_result.json`) |
| MLP 재튜닝(α 가중 하) | ❌ 기각 | 12 trial 전패 — 현행 설정 유지 |
| CatBoost 추가/교체 | ❌ 기각 | 두 해 모두 Δ≈0 — GBM 계열 내 교체는 무효 |
| 시드 앙상블 | ✅ 채택(v5) | 단일시드 CV는 시드운 포함 낙관치 — 앙상블이 실전 기대값 우위 |
| HistGBM 단순 추가 | △ 유지 | 이득 미미하나 전 폴드 일관 (−0.02~−0.06%p) |
| GRU 잔차보정 | ❌ 기각 | +0.55~0.74%p **악화** — 시퀀스 1,400개는 부족 |
| HMM 국면(regime) feature | ❌ 기각 | 폴드 간 부호 뒤집힘 = 노이즈 |
| FiLM / 계층 헤드 조건화 | ❌ 기각 | 두 해 모두 concat 임베딩에 패배 |
| MOS(NWP 보정) | ❌ 미사용 | 규칙 해석 리스크 + GBM이 흡수 (팀 결정, CONSTRAINTS §3) |

(각 판정의 상세 수치는 로컬 `*_summary.json`·실험 노트북에 기록)

> 교훈: 이 데이터 규모(그룹당 1.7~2.6만 행)에서는 **단순한 것(물리 feature·격자 통계·소형 MLP·후처리)이 이기고, 구조적 복잡성(GRU·FiLM·attention류)은 일관되게 진다.**

## 9. 재현 순서

```bash
# 1. 대회 원본을 ~/Downloads/open 에 배치 (또는 scripts/build_spatial_v2.py의 RAW 경로 수정)
# 2. 의존성
uv sync
# 3. (spatial parquet 재생성이 필요할 때만)
uv run python scripts/build_spatial_v2.py
# 4. 최종 제출 재현 → submission_v5.csv
OMP_NUM_THREADS=1 uv run --with nbconvert --with ipykernel --with torch \
  jupyter nbconvert --to notebook --execute --inplace PIPELINE_FINAL.ipynb
```

## 10. 남은 작업

1. **실제 리더보드 제출** (v5 → v4 순) — 홀드아웃(≈0.646) 대비 실전 편차 캘리브레이션이 이후 모든 개선의 기준
2. FICR 후처리 세분화 — 출력구간별 nudge (FICR이 발전량 가중이므로 고출력 구간 집중)
3. group3 개선 — FICR 최약체(0.33대), pooled 재검토·전용 보정
