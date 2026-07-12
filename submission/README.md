# Versioned submissions

각 `ver_{num}` 폴더는 해당 버전의 제출 파일, 결과, 실행 코드 및 관련 노트북을 함께 보관한다.

| 폴더 | 핵심 변경 | 상태 |
|---|---|---|
| `ver_1` | EDA, GBM 기준선, 초기 FICR 후처리 | 이력 |
| `ver_2` | 공간 feature와 감률 | 이력 |
| `ver_3` | GBM + MLP 블렌드 | 이력 |
| `ver_4` | MLP 튜닝 | 이력 |
| `ver_5` | seed ensemble | LB 기각 |
| `ver_6` | 보수적 nudge | 이력 |
| `ver_7` | SCADA 라벨 정제와 GBM 튜닝 | 이력 |
| `ver_8` | FICR 정렬 가중 | LB 기각 |
| `ver_9` | 10-seed v7 구성 | 이력 |
| `ver_10` | OOF forecast combination | LB 무효 |
| `ver_11` | 10% 유효시간 정렬 | 중간 단계 |
| `ver_12` | 수정 기대효용 점추정 | Public 챔피언 |
| `ver_13` | 그룹3 pooled LightGBM | 후보 이력 |
| `ver_14` | 그룹3 HistGBM 제거 | 후보 이력 |
| `ver_15` | 그룹3 50:50 blend | 기각 |
| `ver_16` | 그룹3 constrained gate | 제출 후보 |

v10 이후 파이프라인은 저장소 루트에서 모듈로 실행한다.

```bash
uv run python -m submission.ver_12.pipeline
uv run python -m submission.ver_16.backtest
uv run python -m submission.ver_16.pipeline
```

각 파이프라인은 `submission.csv`, `result.json`, `backtest.json`을 자신의 버전 폴더에 기록한다.
