# 풍력 발전량 예측 딥러닝 — 조건화(conditioning) 계층 설계 리서치 리포트

작성일: 2026-07-08 · 대상 프로젝트: `/root/wind` (BARAM 2026, NWP→시간별 풍력 발전량, 평가 NMAE)
질문: **"HMM regime + GRU에 FiLM / AdaLN-Zero 같은 조건화 계층을 붙이는 게 딥러닝쪽에서 맞는 접근인가? 다른 계층 대안은?"**

---

## 요약 (한 문단)

방향 자체(단순 concat보다 **조건화 변조**가 낫다)는 옳습니다. 다만 **AdaLN-Zero는 GRU에 안 맞습니다**(LayerNorm이 전제인 Transformer용). GRU를 유지한다면 정답은 **FiLM**이고, AdaLN-Zero의 진짜 알맹이(zero-init 항등 초기화)만 FiLM에 이식하면 됩니다("FiLM-Zero"). 그리고 **가장 중요한 두 가지**는 따로 있습니다: ① **"바람 있는데 발전 0" 신호는 조건 신호로 못 씁니다(테스트 누설)** — HMM regime은 반드시 **NWP 변수로만** 정의해야 합니다. ② 조건화 메커니즘보다 **RevIN(분포이동 정규화)** 이 비용대비 실익 1위입니다. 결론적으로 **소량·정형 데이터에서 conditioning 메커니즘 선택은 2차 효과**이고, regime은 우선 **LightGBM feature로** 넣어 이득을 확인한 뒤, 딥 보완층을 붙일 때만 **RevIN + FiLM-Zero**를 쓰는 순서가 맞습니다.

---

## 0. 대전제 — "바람 있는데 발전 0"은 조건 신호로 쓸 수 없다 (누설)

이게 모든 설계의 출발점입니다. 국면(regime)을 두 종류로 반드시 구분해야 합니다.

| 국면 종류 | 예시 | 출처 | 테스트(2025)에서 쓸 수 있나 |
|---|---|---|---|
| **(A) 날씨 국면** | 잔잔 / 돌풍 / 전선통과 / 대기 안정도 | **NWP에서 계산 가능** | ✅ 재현됨 → 조건화 OK |
| **(B) 고장·출력제한 국면** | "바람 있는데 발전 0" | **SCADA(train)에만 있음** | ❌ NWP로 예측 불가 → 조건화 불가 |

당신이 주목한 **"바람 있는데 발전 0"은 정확히 (B)** 입니다. 이건 SCADA에만 있고 2025 테스트엔 없으며 NWP로 예측할 수도 없습니다. 그래서:

- **HMM regime은 반드시 NWP 파생 변수(풍속·spread·전단·안정도)로만 정의**해야 테스트에서 재현됩니다. 실측 발전량으로 regime을 학습하면 학습 땐 좋아 보여도 테스트에서 무너집니다.
- (B) 신호의 유일한 합법적 용도는 **학습 라벨 정제**(정지 구간 마스킹·가중치 축소)와 **파워커브 적합**뿐입니다. → 이전 EDA 분석과 계획서의 "SCADA는 파워커브/정제용, 테스트 입력 금지" 방침과 일치.
- **줄일 수 없는 한계**: 2025년 정지는 예측 불가이므로, 실측 발전량의 약 6~9%(EDA에서 확인)는 원리적 오차 하한으로 남습니다.

> 요컨대: 아키텍처를 (B) 신호 중심으로 설계하지 말 것. HMM+conditioning으로 얻을 수 있는 이득은 **(A) 날씨 국면**에서만 나옵니다.

---

## 1. 질문에 대한 직답

**"HMM regime + GRU에 FiLM/AdaLN-Zero"** — 방향은 맞지만 조합에 두 가지 수정이 필요합니다.

1. **AdaLN-Zero ↔ GRU는 구조적으로 안 맞습니다.** AdaLN-Zero는 정의상 *LayerNorm의 scale/shift를 조건으로 생성 + 잔차 게이트 zero-init*입니다. 바닐라 GRU엔 LayerNorm이 없어 변조할 대상 자체가 없습니다.
2. **GRU를 유지하면 FiLM이 정답**입니다(FiLM 원논문의 조건 생성기 자체가 GRU였을 만큼 RNN 조건화의 표준). AdaLN-Zero에서 가져올 진짜 가치는 아키텍처가 아니라 **zero-init 항등 초기화**이며, 이건 FiLM에 그대로 이식됩니다.

즉 선택지는: **(a) GRU 유지 → FiLM(-Zero)** 또는 **(b) AdaLN-Zero를 제대로 쓰려면 Transformer로 전환**. 이 문제 규모(정형·소량·NMAE)에선 **(a)** 가 맞고, (b)로 아키텍처를 바꾸는 것은 본말전도입니다(3·6장).

---

## 2. FiLM vs AdaLN-Zero — 원리와 GRU 적합성

### FiLM (Feature-wise Linear Modulation, Perez et al. 2018)
- **원리**: 조건 `z`에서 작은 생성기가 채널별 `γ(z), β(z)`를 만들어 활성값을 `h' = γ(z)⊙h + β(z)`로 affine 변조. 핵심은 **곱셈적 상호작용**을 명시적으로 주입.
- **concat보다 나은 이유**: concat은 조건을 입력에 한 번 덧붙일 뿐, 곱셈적 상호작용을 뒤 층이 스스로 풀어야 함(소량에선 잘 안 됨). FiLM은 gating의 일반화(scale+shift 무제약)라 표현력이 크고, **저차원·전역 조건(regime 확률/그룹 임베딩)에 특히 효율적**.
- **시계열 선례**: **TFT**(1912.09363)가 사실상 FiLM 계열 — static covariate로 문맥 벡터를 만들어 GRN을 통해 시간축 처리를 조건화(static enrichment). "regime/그룹이 시간 동역학을 변조"라는 당신 문제의 정확한 선행 사례이고 **정형 multi-horizon 예측용으로 설계**됨.

### AdaLN-Zero (DiT, Peebles & Xie 2023)
- **원리**: Transformer 블록의 LayerNorm scale/shift를 조건으로 생성 + 잔차 직전 스케일 α를 **0으로 초기화** → 각 블록이 학습 시작 시 **항등함수**가 되어 깊은 트랜스포머가 안정적으로 학습됨.
- **DiT에서 우위였던 이유**: 조건이 **단일 전역 벡터**(class+timestep)라 전역 affine 변조가 값싸고 강력했고, cross-attention(+15% FLOPs)보다 품질·연산 모두 우월. **핵심은 zero-init(항등 초기화)**.
- **GRU 급소**: LayerNorm이 전제라 바닐라 GRU엔 그대로 못 붙음. 억지로 LayerNorm-GRU를 만들어 조건화하는 건 검증된 레시피가 없는 비표준 조합.

### 실전 트릭 — "FiLM-Zero"
AdaLN-Zero의 알맹이인 **항등 초기화를 FiLM에 이식**하세요: γ 생성기를 `γ=1, β=0`(또는 잔차 게이트 α=0)으로 초기화하면, 학습 시작 시 GRU가 **무조건화 baseline과 동일**하게 출발합니다. 소량 데이터에서 "조건화가 도움될 때만 자라고, 최소한 plain GRU보다 나빠지지 않는다"는 안전장치를 얻습니다.

---

## 3. 조건화 신호(z)를 무엇으로 구성할까

권장 조건 벡터 `z = concat[...]`:
- **HMM regime posterior (soft, K차원)** — filtered/smoothed 확률. **하드 argmax 금지**(경계 불연속·gradient 단절 유발, soft가 부드러운 혼합을 줌). **단, NWP 변수로만 학습**(0장).
- **그룹 임베딩** — 학습 가능한 entity embedding(TFT의 static entity embedding과 동일 발상, 그룹3 소량 보완).
- **NWP 통계** — 풍속 앙상블 spread(예보 불확실성 대리), 예측 horizon, 시각/계절.

**이산 regime엔 per-regime 테이블이 깔끔**: regime k마다 `(γ_k, β_k)`를 학습하고 soft posterior로 볼록결합 `(γ,β)=Σ_k p_k(γ_k,β_k)`. class-conditional BN과 동일한 검증된 패턴이며 싸고 해석 가능(각 국면의 변조가 눈에 보임).

---

## 4. 조건화 계층 전체 메뉴 — 소량·정형 풍력 기준 랭킹

두 번째 리서치가 후보군 전체를 "비용 대비 실익" 순으로 정리했습니다.

| 순위 | 후보 | 비용 | 실익 | 판정 |
|---|---|---|---|---|
| **1** | **RevIN** (+ 인스턴스/rolling 표준화) | 극저 | 높음 (non-stationarity 직접 대응) | **필수 채택** |
| **2** | **FiLM** (regime확률+NWP통계 → per-feature scale/shift) | 저 | 높음 (곱셈 변조, 파라미터가 조건차원에만 비례) | **1순위 conditioning** |
| **3** | **GRN/GLU 게이팅** (TFT 블록 차용) | 저~중 | 중상 (게이트 정규화 + 변수선택) | 강력한 2순위 / FiLM과 병용 |
| **4** | **Conditional LayerNorm / AdaLN-Zero** | 저 | 중상 (FiLM의 정규화판) | FiLM과 A/B할 근사 동급 |
| 5 | **DAIN** | 중 | 중 (RevIN 부족 시) | 조건부 2순위 |
| 6 | **Squeeze-Excitation (self-gating)** | 저 | 중하 (외부조건 주입 아님) | 보조용 |
| 7 | **MoE / RSMoE (regime experts)** | 중고 | 중 (표본분할로 소량 악화) | 대체로 불필요 (LightGBM 분기+FiLM으로 흡수) |
| 8 | **HyperNetworks** | 고 | 중 (FiLM이 안전한 축소판) | **과설계 — 지양** |
| 9 | **Cross-attention conditioning** | 고 | 저 (저차원 조건엔 부적합) | **과설계 — 지양** |
| 10 | **Mamba / Selective SSM** | 고 | 저 (장시퀀스용, 명시 regime 있으면 불필요) | **과설계 — 지양** |
| 11 | **MDN** | 중 | 저 (NMAE와 지표 불일치) | 탈락 (점추정 지표) |
| 12 | **SLDS** | 고 | 저 (HMM 단계서 이미 취함) | 탈락 |
| 13 | **gMLP** | 중 | 저 (시퀀스 mixing용) | 탈락 |

### 관통하는 원리 (스펙트럼)
조건 신호가 네트워크를 **얼마나 크게 바꾸도록 허용하느냐**의 스펙트럼입니다:

```
concat(상호작용 없음) → SE/게이트(자기 게이팅) → FiLM/CondLN/AdaLN(층별 affine 변조)
   → GRN(변조+게이트+skip) → MoE(모듈 분기) → HyperNet(가중치 전체 생성) → Cross-attn(가변조건 attend)
   왼쪽 ─────────────────── 표현력↑ 파라미터↑ 과적합위험↑ ─────────────────→ 오른쪽
```

**소량·정형의 최적점은 왼쪽~중앙(FiLM/게이팅/조건정규화)** 이고, RevIN은 여기에 얹히는 **직교적(비조건) 정규화**로 분포이동을 별도 처리합니다. 오른쪽(HyperNet/MoE/cross-attn/Mamba)은 표현력은 크지만 이 데이터 규모에서 과적합·불안정·불필요.

---

## 5. RevIN — 놓치면 안 되는 진짜 1순위

두 리서치가 공통으로 강조한, 조건화보다 먼저 챙길 것:

- **RevIN**(Reversible Instance Normalization, ICLR 2022): 입력을 인스턴스별로 learnable affine 정규화하고 출력에서 역정규화. **인스턴스별 레벨/스케일(비정상성)을 모델에서 분리**.
- 풍력은 계절·기단에 따라 레벨·분산 이동이 큰 전형적 non-stationary 신호 → RevIN이 정확히 겨냥. 파라미터는 채널당 affine 2개 수준으로 **위험 사실상 0**.
- **정형 wide 회귀에선** 타깃/피처의 인스턴스 또는 rolling-window 표준화로 변형 적용. RevIN으로 부족하면 DAIN(적응형 입력 정규화)을 2순위로.

---

## 6. GRU 유지 vs Transformer vs TFT — 명확한 권고

**GRU를 유지하세요 (또는 TFT로 확장).** AdaLN-Zero를 쓰려고 플레인 Transformer로 바꾸는 건 over-engineering입니다:
- AdaLN-Zero의 우위는 **대규모 diffusion transformer에서만** 입증됐고, ~17k행 정형 예측엔 전이 안 됨. 트랜스포머는 이 데이터량에서 GRU/GBM보다 나빠지기 쉬움.
- conditioning 기법 하나를 정당화하려고 아키텍처를 바꾸는 건 순서가 거꾸로.
- **아키텍처를 정말 바꿀 거라면 플레인 Transformer+AdaLN이 아니라 TFT.** TFT는 이미 static-covariate GRN enrichment + GLU gating으로 **FiLM식 조건화를 정형 multi-horizon 예측용으로 구현**해 둔 기성품. 경량·정형·소량에서 균형이 좋은 대안은 **TiDE**(경량 MLP 인코더-디코더).
- PatchTST / iTransformer / Mamba류는 "긴 시퀀스·다변량 상관"이 강점인데, 이 문제의 병목이 아님.

---

## 7. 최종 권장 설계

### 1순위 — regime은 먼저 GBM feature로 (가장 강한 반론이자 첫 수순)
- LightGBM은 트리 분기로 **regime 상호작용을 native·데이터효율적으로 처리**(사실상 gradient-boosted MoE). FiLM/AdaLN이 학습으로 얻으려는 걸 공짜로 줌.
- **HMM soft posterior + 상호작용(spread×regime, horizon×regime)을 LightGBM에 직접 투입**하고 이걸 "이겨야 할 baseline"으로 확정. 대부분의 실익을 여기서, 최저 비용·최소 과적합으로 얻음.

### 2순위 — 딥 보완층을 정말 붙일 때: RevIN + GRU + FiLM-Zero
- **RevIN**을 기본 정규화로 무조건 채택.
- 조건 `z = concat[soft HMM posterior(NWP-only, K), 그룹 임베딩, NWP spread/horizon/시간]`.
- **이산 regime은 per-regime (γ_k, β_k) 테이블 + soft posterior 볼록결합**.
- GRU **은닉상태(또는 게이트 pre-activation)를 FiLM 변조**, 주입점 1곳, **γ=1/β=0 zero-init**로 plain GRU와 동일 출발.
- 추가로 **h₀를 static 문맥에서 생성**(가장 싼 강한 조건화).
- 강한 정규화(weight decay) + **GBM과 앙상블(잔차 스태킹)**.

### 3순위 대안 (A/B로만)
- **GRN(TFT 블록) 차용** — 곱셈 게이트 + LayerNorm skip + static context. wide feature 노이즈에 강함.
- **Conditional LayerNorm / AdaLN-Zero** — residual 스택이 깊어지면 FiLM보다 안정. 거의 동급이라 실측 A/B로 결정.
- **단순 concat + 강한 정규화** — 반드시 대조군으로 두고 FiLM이 실제로 이기는지 검증(소량에선 통계적으로 구분 안 되는 경우 많음).

### 피할 것 (과설계 / 지표 불일치)
- **HyperNetworks, Cross-attention conditioning, Mamba/Selective SSM** — 소량·저차원조건·정형에서 과적합·불안정·불필요.
- **순진한 regime-MoE** — 전문가로 데이터를 쪼개면 소량 문제 악화. 분할은 LightGBM 트리가, DL은 FiLM 변조(파라미터 공유)가 담당.
- **MDN**(NMAE는 점추정 L1 지표라 분포 표현력이 값이 안 남), **SLDS**(HMM 단계서 이미 취함), **gMLP**(시퀀스 mixing용).

---

## 8. 당신 프로젝트에 맞춘 실행 순서

1. **HMM을 NWP 변수로만** 학습(풍속·spread·전단·안정도) → soft regime posterior 산출. (실측 발전량 사용 금지 = 누설 방지)
2. regime posterior + 상호작용을 **LightGBM에 feature로 추가** → 2024 holdout에서 NMAE 개선폭 측정. **개선 없으면 여기서 중단**(딥 보완층 불필요).
3. 개선이 있으면 **RevIN + GRU + FiLM-Zero** 잔차 보정 모델을 시계열 CV로 실험. plain GRU / concat GRU를 대조군으로.
4. 최종 **LightGBM ⊕ (GRU 잔차)** 앙상블. FICR 후처리는 마지막.
5. 여유 시 **Conditional LayerNorm / GRN 차용**을 A/B, 아키텍처 확장이 필요하면 TFT 검토.

> 한 줄 결론: **방향(조건화>concat)은 맞다. 단 AdaLN-Zero는 GRU에 안 맞으니 FiLM-Zero로, regime은 반드시 NWP-only로, 그리고 conditioning보다 RevIN과 "먼저 GBM feature로 넣기"가 실익이 크다.**

---

## 부록 A — 용어 함정 (반드시 팀 공유)
"FiLM"은 시계열 문헌에서 **두 개의 완전히 다른 것**을 가리킵니다:
- **FiLM = Feature-wise Linear Modulation** (Perez et al. 2018, arXiv 1709.07871) — 당신이 쓰려는 조건화 계층.
- **FiLM = Frequency improved Legendre Memory** (Zhou et al. NeurIPS 2022, arXiv 2205.08897) — conditioning과 무관한 **예측 백본**. 검색 시 반드시 걸리므로 혼동 주의.

## 부록 B — 주요 출처
- FiLM: [arXiv 1709.07871](https://arxiv.org/abs/1709.07871)
- DiT / AdaLN-Zero: [arXiv 2212.09748](https://arxiv.org/abs/2212.09748)
- TFT (GRN/GLU/VSN/static enrichment): [arXiv 1912.09363](https://arxiv.org/abs/1912.09363)
- RevIN: [ICLR 2022, Kim et al.](https://openreview.net/forum?id=cGDAkQo1C0p)
- DAIN: [arXiv 1902.07892](https://arxiv.org/abs/1902.07892)
- Mamba (Selective SSM): [arXiv 2312.00752](https://arxiv.org/abs/2312.00752)
- HyperNetworks: [arXiv 1609.09106](https://arxiv.org/abs/1609.09106)
- 시계열 diffusion conditioning (TimeGrad/CSDI/SSSD): TimeGrad(ICML 2021), CSDI(NeurIPS 2021)
- TiDE: Das et al. 2023 · 백본 비교(PatchTST/iTransformer/TSMixer/N-HiTS)
- Regime-switching 풍력: MS-AR forecast errors [S0378779620304442](https://www.sciencedirect.com/science/article/am/pii/S0378779620304442)
