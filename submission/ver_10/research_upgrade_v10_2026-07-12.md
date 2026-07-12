# BARAM 2026 v10 개선 조사 요약

## 대회 신호

- DACON BARAM 2026은 GFS/LDAPS 예보와 학습 구간 SCADA를 활용해 3개 KPX 그룹의 시간별 발전량을 예측한다.
- 평가는 `Score = 0.5 * (1 - NMAE) + 0.5 * FICR`이며, 실제 발전량이 설비용량 10% 이상인 시간만 채점된다.
- 현재 공개 리더보드는 1-NMAE가 상위권에서 0.87대에 몰려 있고 FICR 차이가 순위를 크게 가른다.

## 조사에서 얻은 구현 포인트

1. **NWP forecast combination**
   - 최신 풍력 예측 연구는 단일 예보보다 여러 기상 예보/모델의 조합이 유효하다고 보고한다.
   - 이 저장소에서는 GFS와 LDAPS를 이미 함께 쓰고 있으므로, 다음 개선은 모델 출력 조합을 OOF 기반으로 고르는 쪽이 맞다.

2. **Tree model + neural model ensemble**
   - KDD Cup 2022 풍력 솔루션들은 GBDT가 기본 패턴을 잡고 RNN/그래프 계열이 시공간 의존성을 보완하는 식의 앙상블을 사용했다.
   - 우리 데이터는 그룹 3개, 연 2~3년 규모라 복잡한 그래프/시퀀스 모델은 기존 실험에서 불안정했다. 따라서 v10은 새 구조를 크게 키우지 않고, 기존 GBM/MLP 후보의 조합을 더 견고하게 선택한다.

3. **Power-curve anchoring**
   - 물리식/경험적 power curve는 단독 모델보다는 앵커나 약한 보정 후보로 쓰는 편이 안정적이다.
   - v10은 isotonic power curve 예측을 0/3/6% 범위에서만 섞어, 과한 평균 CF 상승을 방지한다.

4. **Metric-aware but conservative postprocess**
   - FICR는 6%, 8% 오차율 임계값을 갖는 비연속 지표라 nudge가 효과적일 수 있다.
   - v5/v8 실패에서 확인했듯 평균 CF를 크게 올리는 후처리는 2025 분포에서 처벌된다. v10은 기존 `scale <= 1.05`, `shift <= 2% capacity`와 최종 평균 CF 가드를 유지한다.

## 구현

- 실행 모듈: `submission.ver_10.pipeline`
- 출력:
  - `submission/ver_10/result.json`: OOF/holdout 조합 탐색 결과, 선택 조합, nudge 값, 최종 평균 CF
  - `submission/ver_10/submission.csv`: DACON 제출 파일
- 후보 조합:
  - LightGBM/HGB 비율: 0.35, 0.50, 0.65
  - MLP 비율: 0.60, 0.65, 0.70, 0.75
  - power-curve 약한 앵커: 0.00, 0.03, 0.06
- 선택 기준:
  - v7식 고정 조합 대비 2023 raw, 2024 post 점수의 worst delta를 우선한다.
  - 평균 CF 가드 위반 조합은 후순위로 둔다.
