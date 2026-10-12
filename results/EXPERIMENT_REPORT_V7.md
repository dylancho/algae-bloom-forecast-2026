# 개선 실험 보고서 V7 — 외부 신규 관측(영양염·자동측정망) 수집과 검증

작성일: 2026-10-09. V6 보고서가 남긴 "실제 신규 관측 확보" 항목을 실행했다. **결론: 미채택.** 두 종류의 신규 자료를 확보해 피처로 넣었지만 과거 CV 개선폭이 표준오차 1배 수준이고 NSE는 나빠졌다. 2025년 참고 검증에서도 고정 후보가 기존 모델과 같았다. 기존 선택·제출 파일은 바꾸지 않았다.

## 1. 수집한 자료

| 자료 | 출처 | 범위 | 2026 평가 구간 사용 가능성 |
|---|---|---|---|
| 수질측정망 채수 자료 (TN, TP, PO4-P, NO3-N, NH3-N, DTP, Chl-a 등) | 물환경정보시스템 파일셋 다운로드, 연도별 xlsx 7개 | 2011-01 ~ 2026-07-30, 낙동강·금강 21만 행 | 2026 파일이 7/30까지라 8~9월 기준일은 최대 45일 묵은 값 |
| 수질자동측정망 일자료 (수온, pH, DO, EC, 탁도, TOC, TN, TP, Chl-a) | 물환경정보시스템 확정·미확정 자료 조회 | 15개 측정소, 2012-07 ~ 2026-10-08 | 2026 5~9월도 가능. 미확정 자료는 4월 이후 제공 |
| 보 수위 | 물환경정보시스템 보 운영 현황 | 2017 ~ 2026-08 | **미사용**: 수위만 있고 방류량이 없어 기존 WAMIS 보 상·하류 수위계와 중복 |
| 과거 발표 기상예보 | 기상청 API허브 / 기상자료개방포털 | — | **미수집**: 인증키 또는 회원 로그인이 필요해 확보하지 못함 |

- 원자료와 수집 기록은 `data_external/nier_wq/`(manifest.json에 fileId·SHA-256), `data_external/nier_auto/`(manifest.json)에 있다. 수집·변환 코드는 `scripts/convert_nier_wq.py`, `scripts/fetch_nier_auto.py`다.
- 자동측정망 Chl-a는 측정소마다 2026년 가동 여부가 다르다. 칠서·해평은 2026년 Chl-a가 없어서, 2026년에도 Chl-a를 재는 인접 측정소를 골랐다: 해평→칠곡, 강정고령→강창, 칠서→적포, 물금매리→상동, 회남→대청호.
- 외부 자료 사용이 대회 규정상 허용되는지는 여전히 확인되지 않았다(V5와 같은 상태).

## 2. 피처

- **영양염 22개** (`algae/nutrients.py`): 근접·상류 수질측정망 지점별 최신 TN·TP·PO4-P·NO3-N·NH3-N·Chl-a(log), TN:TP, 직전 대비 TP·TN 변화, 45일 평균 TP, 경과일. 채수일 ≤ 기준일−lag만 사용한다. lag 1일(기본)과 14일(분석 결과 공개 지연 가정) 두 가지를 비교했다.
- **자동측정 18개** (`algae/autowq.py`): Chl-a 최근값·3/7일 평균·7일 최대·추세·30일 대비, 7일 TN·TP, pH·DO 3일 평균과 추세(광합성 신호), 탁도, EC 3/21일 비(희석 신호), TOC, 수온. 기준일 전날까지 자료만 사용한다.
- 누수 테스트 4개를 추가했다(`tests/test_nutrients_v7.py`, `tests/test_autowq_v7.py`): 기준일(또는 기준일−lag) 이후 값을 크게 바꿔도 피처가 바뀌지 않음을 확인했다.

## 3. 과거 CV (2018~2024, 5~10월, 7개 연도 동일 가중 평균)

V6와 같은 fold·모델·하이퍼파라미터를 썼다. `blend_tab_*`는 v1 50/50 블렌드에서 TabPFN만, `blend_lgb_*`는 LightGBM만, `blend_both_*`는 둘 다 교체한 것이다. 음수 RMSE 차이가 개선이다.

| 설정 | log RMSE | NSE | v1 대비 RMSE | 표준오차 | 개선 연도 | v1 대비 NSE |
|---|---:|---:|---:|---:|---:|---:|
| v1_selected_5050 | 0.79733 | 0.20913 | 0 | — | — | 0 |
| **blend_tab_nutr** | 0.79224 | 0.19499 | −0.00509 | 0.00499 | 5/7 | −0.0141 |
| blend_tab_nutr_lag14 | 0.79318 | 0.20456 | −0.00416 | 0.00542 | 5/7 | −0.0046 |
| blend_tab_auto_bio | 0.79518 | 0.20214 | −0.00215 | 0.00501 | 4/7 | −0.0070 |
| v6_blend_tabpfn_environment | 0.79579 | 0.21012 | −0.00154 | 0.00491 | 3/7 | +0.0010 |
| blend_tab_auto | 0.79611 | 0.18552 | −0.00122 | 0.00552 | 5/7 | −0.0236 |
| blend_tab_auto_nutr | 0.79702 | 0.20531 | −0.00031 | 0.00647 | 3/7 | −0.0038 |
| blend_lgb_auto_bio | 0.80122 | 0.19676 | +0.00389 | 0.00487 | 3/7 | −0.0124 |
| blend_lgb_nutr_lag14 | 0.80202 | 0.20399 | +0.00469 | 0.00242 | 2/7 | −0.0051 |
| blend_lgb_nutr | 0.80473 | 0.17859 | +0.00740 | 0.00479 | 2/7 | −0.0305 |

전체 결과는 `artifacts/cv7/summary.csv`, `scores.csv`(지점별 포함)에 있다.

해석:

- **LightGBM은 어떤 신규 피처를 넣어도 나빠졌다.** V2·V6와 같은 패턴이다.
- **TabPFN을 교체하면 약간 좋아지지만, 모든 후보가 같은 연도 패턴을 보인다**: 2019년 −0.02~−0.03, 2022·2024년 +0.01~+0.02. V6의 environment 피처도 같은 모양이다. 영양염·Chl-a 고유의 정보라기보다 TabPFN에 피처를 추가할 때 생기는 공통 효과로 보는 편이 맞다.
- NSE(원 단위 고농도 정확도)는 거의 모든 후보에서 나빠졌다. 순위 지표에 R²·NSE가 포함되므로 로그 RMSE만으로 채택할 수 없다.
- 지점별로도 표준오차 2배를 넘는 개선은 없었다. 가장 큰 것은 회남(대청호) 자동측정 −0.005±0.003이지만, 여러 지점·변형을 비교한 결과라 우연 범위로 본다.
- 공개 지연을 반영한 lag 14일 영양염도 lag 1일과 거의 같았다. 영양염 신호가 1~2주 시차에 민감하지 않다는 뜻이다.

## 4. 2025년 참고 검증

2025년 예측 전에 `artifacts/cv7/frozen_candidate.json`에 고정했다. 규칙은 V6와 같은 "과거 평균 log RMSE 최소 블렌드"다.

| 설정 | log RMSE | MAE | NSE |
|---|---:|---:|---:|
| v1_selected_5050 | 0.63350 | 0.47691 | 0.00599 |
| **blend_tab_nutr (고정 후보)** | 0.63363 | 0.48388 | −0.00298 |
| blend_tab_nutr_lag14 (참고) | 0.62883 | 0.48482 | 0.00036 |
| blend_tab_auto_bio (참고) | 0.62686 | 0.47726 | −0.00207 |
| v6_blend_tabpfn_environment (V6 기록) | 0.61969 | 0.46859 | 0.01643 |

고정 후보는 기존 모델과 같았다(차이 +0.0001). 참고 후보 두 개는 RMSE가 약간 낮지만 NSE는 기존보다 낮다. 2025는 여러 버전에서 이미 본 자료이므로 이 결과로 후보를 바꾸지 않았다.

## 5. 판정과 남은 방향

**미채택.** 채택 기준(표준오차 1배를 넘는 RMSE 개선 + NSE 손실 없음)을 어떤 후보도 넘지 못했다. `artifacts/selection.json`, `results/submission_draft.csv`, `results/submission_refit2025.csv`는 그대로이고, `python -m algae.exp6 verify`로 해시 불변을 확인했다.

이번 결과로 확인된 점:

1. 주 1회 채수 영양염과 일별 Chl-a 자동측정 모두 기존 세포수·수온·유량 이력이 담은 정보를 크게 넘지 못했다. 7일 뒤 세포수를 정하는 것은 과거 상태보다 **앞으로 7일의 날씨·유량**일 가능성이 크다.
2. 따라서 남은 가장 유력한 정보는 **과거 발표 기상예보**(예보 시점 ≤ 기준일)다. 기상청 API허브 인증키가 있으면 수집할 수 있다. 회원가입·키 발급은 사용자가 직접 해야 한다.
3. 자동측정 Chl-a는 2026년 평가 구간에서도 쓸 수 있는 유일한 고빈도 생물량 자료다. 쓸 경우 회남(대청호)에 한정하는 지점별 모델을 별도 가설로 사전 등록해 검증해야 한다.

## 6. 재현

```powershell
& .venv\Scripts\python.exe -m scripts.convert_nier_wq        # xlsx -> data_external/nier_wq/wq_nakdong_geum.csv
& .venv\Scripts\python.exe -m scripts.fetch_nier_auto        # 자동측정망 스냅샷 (네트워크 필요)
& .venv\Scripts\python.exe -m unittest discover -s tests     # 21개 통과
& .venv\Scripts\python.exe -m algae.exp7 run --model lightgbm
& .venv\Scripts\python.exe -m algae.exp7 run --model tabpfn --variants nutr nutr_lag14 nutr_near auto_bio auto auto_nutr
& .venv\Scripts\python.exe -m algae.exp7 summary
& .venv\Scripts\python.exe -m algae.exp7 run --model tabpfn --variants nutr nutr_lag14 auto_bio --years 2025
& .venv\Scripts\python.exe -m algae.exp7 summary --years 2025
```

환경 변경: xlsx 읽기를 위해 `.venv`에 `openpyxl 3.1.5`(+`et-xmlfile 2.0.0`)를 설치했다. 모델 코드에는 영향이 없다.
