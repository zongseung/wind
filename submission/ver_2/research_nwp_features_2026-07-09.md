# 리서치: NWP feature 추가 — 무엇이 근거 있고 무엇을 빼야 하나

작성: 2026-07-09 · 목적: 전처리 v2에서 **변수 수를 늘리지 않고(오히려 줄이며)** 실증 근거 있는 것만 추가
대상: BARAM 2026 (day-ahead 리드 12~35h, 한국 태백 복잡 산악지형, 허브 117m, GBM 기반)

---

## 요약 (한 문단)

근거의 강도가 뚜렷하게 갈립니다. **① 다중 격자 공간 feature는 강한 실증 근거**(단일 격자 대비 풍력 MAE **−12.85%**, Andrade & Bessa 2017; GEFCom2014 우승팀들도 NWP 격자 공간 feature 사용)가 있어 최우선 추가 대상입니다 — 새 원본 변수 없이 **이미 있는 풍속의 집계만 바꾸면** 됩니다. **② 수직층 바람**은 한국 복잡지형 day-ahead 연구(Energy 2024)가 "상위 수직층이 더 강한 상관을 보일 수 있으며 층 선택+PCA로 NMAE 감소"를 보고해 중간 근거 — 우리는 850/700hPa 바람을 이미 보유하므로 **추가 불필요**. **③ 대기 안정도(기온 감률)**는 전단·허브고도 외삽·파워커브에 영향을 준다는 물리·자원평가 근거가 일관되나 day-ahead ML feature로서의 직접 실증은 약함 — **파생 2개만** 소규모 추가 후 CV 판정 권고. **④ 500hPa 지위고도·기온·바람, 850hPa 습도**는 풍력 예측 feature로서의 실증 근거를 찾지 못함 — **제외 권고**.

---

## 질문별 판정

### Q1. 500hPa 지위고도/바람, 850·700hPa 기온이 정확도를 올리는가?

| 변수 | 판정 | 근거 |
|---|---|---|
| **500hPa gh·t·u·v** | ❌ **제외** | 종관 패턴 차트용(z500/t850)으로는 표준이나, 풍력 예측 feature importance로 유효했다는 실증 문헌 미발견. 허브 117m 대비 ~5.5km 상공은 850hPa(~1.5km)이 이미 대변 |
| **850·700hPa 기온** | ⚠️ **감률 파생으로만** | 직접 근거는 약하나 안정도 계산의 재료(Q3). 원변수 2개 → 파생 2개로 압축 투입 |
| **850hPa 습도** | ❌ 제외 | 근거 없음. EDA에서도 습도류 상관 ~0 |
| GEFCom2014 우승 접근 | 참고 | GBM + **과감한 feature selection** + 격자 공간 feature가 우승 공식. "많이 넣기"가 아니라 "잘 고르기"가 반복 확인됨 |

### Q2. 다중 격자(공간 이웃·gradient)가 최근접 단일 격자보다 나은가?

**예 — 이번 리서치에서 가장 강한 근거.**
- Andrade & Bessa (2017, IEEE Trans. Sustainable Energy): NWP **격자** feature(공간 평균·분산·smoothing·PCA) + GBT로 단일 지점 대비 **풍력 MAE −12.85%, CRPS −12.06%**.
- GEFCom2014 풍력 트랙: 우승팀들이 격자 기반 시공간 feature 사용, 타 지점 정보 활용으로 pinball loss −2.5%.
- 우리 현황: GFS는 9격자 중 **1개만**, LDAPS는 16격자 중 2~3개만 사용 → 문헌이 지목하는 "location error"에 그대로 노출.
- **비용 0**: 새 원본 변수 없이 기존 풍속의 격자 집계(공간 mean/std/gradient)만 추가.

### Q3. 안정도(감률·Richardson 대용)가 허브고도 풍속 추정에 유효한가?

**물리적으로 예, 실증은 중간.**
- 전단지수 α는 안정도에 민감하고 주야로 변함(야간 안정 시 α↑) — 고정 α 대신 안정도 반영 시 자원평가·외삽 정확도 개선 (Renewable Energy 2016; IOP ERL 2012; Int J Energy Res 2021).
- "day-ahead 예측도 안정도 반영 파워커브로 개선될 것"이라는 서술은 있으나 ML feature로서의 정량 실증은 부족.
- 권고: **감률 2개만** 추가(`t850−t700`, `t2m−t850` 역전 프록시) 후 CV로 판정. 우리가 이미 가진 α(풍속비 기반)와 상보적(기온 기반은 예보 풍속 오차와 독립).

---

## 최종 권고: v2 변수 계획 (총량은 오히려 감소)

**빼기** (기존 축소 ablation F3 채택: 86 → 55):
- hub_v/v²/v³(트리 불변), 구름·강수·습도·이슬점 19개(EDA 상관~0), 풍속 요약통계 중복 9개

**더하기** (+10, 모두 근거 기반):
| 추가 feature | 개수 | 근거 강도 |
|---|---|---|
| GFS ws100 9격자 공간 mean·std | 2 | ★★★ (−12.85% 문헌) |
| GFS ws100 동서·남북 gradient | 2 | ★★★ 동일 |
| LDAPS ws 16격자 공간 mean·std | 2 | ★★★ 동일 |
| LDAPS ws 동서·남북 gradient | 2 | ★★☆ |
| 감률 t850−t700, 역전 t2m−t850 | 2 | ★★☆ (물리+자원평가) |

**추가 안 함**: 500hPa 전체, 850hPa 습도, LDAPS 5m BLWS(10m와 중복), 복사·적설·lsm·지형고도.

**결과: 86 → 약 65개** (축소 −31, 근거 있는 추가 +10). 각 추가군은 expanding-window CV에서 공식 총점 개선 시에만 최종 채택.

---

## 출처

- [Andrade & Bessa 2017, Improving Renewable Energy Forecasting With a Grid of NWP (IEEE)](https://ieeexplore.ieee.org/document/7903735/) · [INESC TEC 리포지토리 PDF](https://repositorio.inesctec.pt/handle/123456789/5297)
- [GEFCom2014 총설 (Hong et al., IJF 2016)](https://www.sciencedirect.com/science/article/abs/pii/S0169207016000133) · [GEFCom2014 GBM 우승 접근·격자 feature 논의 (arXiv 2404.17276)](https://arxiv.org/pdf/2404.17276)
- [한국 복잡지형 day-ahead: 수직층 바람 특성 통합 (Energy 2024)](https://www.sciencedirect.com/science/article/pii/S0360544223031080)
- [복잡지형 CNN+DeepSHAP (Frontiers in Energy Research 2023)](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2023.1328899/full)
- [안정도-가변 전단계수로 외삽 개선 (Renewable Energy 2016)](https://www.sciencedirect.com/science/article/abs/pii/S0960148115303876)
- [안정도의 터빈 출력 영향 (IOP ERL 2012)](https://iopscience.iop.org/article/10.1088/1748-9326/7/1/014005)
- [전단지수 글로벌 분석 (Int J Energy Res 2021)](https://onlinelibrary.wiley.com/doi/full/10.1002/er.6382)
- [ECMWF z500/t850 표준 차트 (종관 패턴 참고용)](https://charts.ecmwf.int/products/medium-z500-t850)

신뢰도: Q2 높음(복수 독립 출처·정량 수치) · Q3 중간(물리 근거 강, ML 실증 약) · Q1 제외판정 중간(부재 증명은 불가능하나 탐색 범위 내 근거 없음)
