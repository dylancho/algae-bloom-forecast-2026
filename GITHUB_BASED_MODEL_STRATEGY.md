# GitHub 자료 기반 모델 전략 보완 (2026-10-12)

입력 자료
- `유해남조류_예측_관련논문_검토.md` (문헌 검토)
- 실험 기록 V1~V9 (`results/EXPERIMENT_REPORT*.md`, `CLAUDE_HANDOFF.md`)
- GitHub 저장소 6개를 얕게 복제해 코드를 직접 읽었다(실행하지 않음). 확인한 커밋은 각 절에 적었다.
- 이번 작업에서 새로 돌린 데이터 점검 2가지(2절)

이번 문서는 전략과 데이터 점검까지만 다룬다. 모델 선택과 제출 파일은 바꾸지 않았다.

---

## 0. 결론 요약

1. **참고 저장소 대부분은 우리 과제와 문제 유형이 다르다.** 6개 중 미래를 예측하는 것은 CyanoLakes(1주, 위성 Chl-a)와 ACES(1일, 자동측정 Chl-a) 두 개뿐이다.
   - **유해남조류 세포수를 7일 앞서 예측하는 저장소는 없다.**
   - Tick Tick Bloom 수상작 3개와 Nasios는 채수 당일의 심각도를 위성으로 맞히는 **현재 농도 탐지**다.
   - EPA LakeTrophicModelling은 호수 한 번 방문 자료로 영양 상태를 분류하는 **횡단면 모델**이다.
   - EPA provisional_habs는 모델이 아니라 **관측 자료 품질관리** 사례다.

   따라서 가져올 것은 성능 수치가 아니라 **설계 방식**이다. 지역별 모델, 과거로만 계산한 공간 지연, 계절 기준값, 인과 전처리 규칙, 품질 플래그가 여기에 해당한다.
2. **데이터 점검 결과, 여러 변수를 제외하거나 범위를 좁혀야 한다.**
   - 낙동강 보 운영 자료는 없다.
   - 저수량은 2026년에만 있다.
   - 구미 ASOS 일사량은 89%가 결측이다.
   - 회남 인근 수질측정망(대청댐5) 영양염은 월 1회뿐이다.
   - 해평 지점은 5~10월 1만 이상 관측이 21건뿐이다.
3. **가장 먼저 할 실험 3개**는 아래와 같다(7절).
   - E1 수역별(낙동강 4지점 / 회남) 모델과 통합 모델 비교
   - E2 하천 순서를 반영한 상류 지점 정보 재검증
   - E3 지표 확정에 대비한 다지표 혼합 가중 선택과 평활 기준선 결합

   셋 다 기존 파이프라인 안에서 수 시간 내 구현할 수 있다. 외부 자료를 새로 받을 필요도 없다.

---

## 1. 저장소별 코드 확인 결과

### 1.1 CyanoLakes/forecasting — 1주 예측 기준모델 (master `dc0c5a5`)

| 항목 | 코드에서 확인한 내용 |
|---|---|
| 대상 | 위성 Chl-a 중 남조류 귀속분(`chla_cyano`)과 중앙값(`chla_med`). 세포수가 아니다 |
| 전처리 | [`forecast.py`](https://github.com/CyanoLakes/forecasting/blob/master/forecast.py): `data.resample('W').mean().bfill()`로 주평균을 낸 뒤 **역방향 채움**. 관측이 빈 주에는 미래 값이 들어간다 |
| 모형 | [`models.py`](https://github.com/CyanoLakes/forecasting/blob/master/models.py) `moving_average_seasonal_error_adjusted`: f = a·MA + (1−a)·(다음주 계절평균 + (MA − 이번주 계절평균)), 1주 a=0.8. MA는 최근 2주 평균 |
| 계절평균 | `mcma = cma[cma.index <= dt1]`로 **예측 시점까지의 자료만** 사용(인과적) |
| 검증 | 마지막 1년 보류, 원 단위 RMSE와 위험등급 일치율 |

우리 적용: 지점 × 주차 계절평균을 **각 폴드의 학습 연도로만** 계산하고 로그 세포수로 바꿔 재현했다(2.8절). 그 결과 기준모델로는 유효하지만 현재 모델보다 크게 뒤진다. `bfill`은 우리 규칙상 금지다.

### 1.2 drivendataorg/tick-tick-bloom — 녹조 대회 수상작 (main `c59c9ab`)

과제는 채수 당일 심각도(1~5등급)를 위성 영상으로 맞히는 **탐지**다. 미래 예측이 아니다.

| 순위 | 변수 설계 | 지역별 모델 | 앙상블 | 검증 | 시간 처리 |
|---|---|---|---|---|---|
| 1위 [`Model.py`](https://github.com/drivendataorg/tick-tick-bloom/blob/main/1st%20Place/notebooks/Model.py) | 위성 색상(R,G,B 비율), 월 | 북동·중서부만 LightGBM, 서·남부는 위경도 KNN(거리 역가중 100개) | 100회 반복 평균 | KMeans 100개 공간군집 `GroupKFold` | 무작위·공간 분할 |
| 2위 [`src/get_data.py`](https://github.com/drivendataorg/tick-tick-bloom/blob/main/2nd%20Place/src/get_data.py) `spatial_lag` | 100/300/1000km 안 **과거(`date < pdat`) 라벨 평균**, 고도, 위성 | 지역 순서형 변수 + 수동 군집 | CatBoost·LightGBM·XGBoost 평균 | `train_test_split` 무작위 | 공간 지연은 과거만 사용(올바름) |
| 3위 [`train.py`](https://github.com/drivendataorg/tick-tick-bloom/blob/main/3rd%20Place/train.py) | 위성 통계, HRRR 기온·습도, 영상 촬영 경과일 | **위성 종류 × 지역마다 별도 LightGBM** | 모델 평균 | `StratifiedGroupKFold(shuffle=True)` | [`download_hrrr_forecasts.py`](https://github.com/drivendataorg/tick-tick-bloom/blob/main/3rd%20Place/download_hrrr_forecasts.py): 채수 시각 **앞뒤 12시간** 분석장(fxx=0) |

우리 적용
- **지역별 모델:** 수상작 3개 모두 쓴다. 우리 데이터에서도 호소(회남)와 하천의 강우 반응 부호가 반대였다(V8). 그래서 E1로 시험한다.
- **공간 지연:** 2위의 "과거 라벨만" 원칙을 우리 상류 지점 피처에 그대로 쓴다(E2).
- **그대로 쓰면 안 되는 것:** 무작위·공간 분할과 채수 후 12시간 기상은 탐지 과제에서만 허용된다. 7일 예측에서는 누수다.

### 1.3 donghyunlee-lab/aces-hab-forecasting — 국내 자동측정망 (main `8f2c05e`)

| 항목 | 코드에서 확인한 내용 |
|---|---|
| 대상 | 금강·낙동강 자동측정소 10곳(칠서·대청호 포함)의 **1일 앞 Chl-a**. 세포수가 아니다 |
| 분할 | [`protocols/uq_iso_hab_clean_v2.yml`](https://github.com/donghyunlee-lab/aces-hab-forecasting/blob/main/protocols/uq_iso_hab_clean_v2.yml): 학습 2021–23 / 선택 2024 / 봉인 2025, `embargo_days: 1` |
| 입력 규칙 | 같은 파일: 마스크 후 **인과적 전방 채움 최대 3단계**, 관측 마스크, 관측 후 경과일. 학습 구간 중앙값으로 채움. 금지 목록은 `backfill, bidirectional_interpolation, centered_rolling, full_period_statistics` |
| 스케일링 | [`src/data/preprocessor.py`](https://github.com/donghyunlee-lab/aces-hab-forecasting/blob/main/src/data/preprocessor.py): `fit_period` 없으면 오류. log1p와 MinMax를 학습 구간에서만 적합 |
| 예측구간 | [`scripts/eval_calibration_clean_v2.py`](https://github.com/donghyunlee-lab/aces-hab-forecasting/blob/main/scripts/eval_calibration_clean_v2.py): 정적/확장/이동 conformal과 ACI(γ 갱신). γ·창 길이는 **2024 검증 예측으로만** 선택. 경보 판정은 `upper >= threshold` |

우리 적용
- **누수 방지 규칙:** 우리 규칙과 거의 같다. 금지 목록을 우리 테스트 기준에 명시적으로 넣는다(6절).
- **예측구간:** 대회는 점 예측만 채점하므로 순위에는 직접 쓰이지 않는다. 원 단위 지표로 확정되면, 구간 상단을 피크 보정에 쓰는 실험을 후순위로 둔다. V9 이후 배율 보정이 실패했으므로 기대는 낮다.

### 1.4 IoannisNasios/HarmfulAlgalBloomDetection — 위성·지형·기상 결합 (master `1d12763`)

| 항목 | 코드에서 확인한 내용 |
|---|---|
| 과제 | Tick Tick Bloom과 같은 탐지 과제(5위) |
| 위성 | [`download_satellite_PlanetaryComputer.ipynb`](https://github.com/IoannisNasios/HarmfulAlgalBloomDetection/blob/master/download_satellite_PlanetaryComputer.ipynb) `get_date_range`: 채수일 **이전 15일~당일** 영상(인과적) |
| 기상 | [`download_clima_noaahrrr.ipynb`](https://github.com/IoannisNasios/HarmfulAlgalBloomDetection/blob/master/download_clima_noaahrrr.ipynb): 채수일 HRRR 지표 변수(강우 여부·돌풍 등) |
| 검증 | [`lgbmNN_ee_gkf_S_v42g.ipynb`](https://github.com/IoannisNasios/HarmfulAlgalBloomDetection/blob/master/lgbmNN_ee_gkf_S_v42g.ipynb): KMeans 위치 군집 GroupKFold |

우리 적용
- 우리 지점은 5개의 고정 지점이라 지형(DEM) 변수는 지점 ID와 같은 정보다. 이득이 없다.
- 위성 Chl-a·남조류 지수는 이론상 고빈도 보조 신호가 될 수 있다. 하지만 대회 번들에 없고 외부 자료 허용 여부도 미확인이며, 하천 폭이 좁아 화소 혼합 문제가 크다. 이번 범위에서는 제외한다.

### 1.5 USEPA/LakeTrophicModelling — 영양염·수역 특성 (master `026af1e`)

| 항목 | 코드에서 확인한 내용 |
|---|---|
| 자료 | 미국 NLA 2007 호수 1회 방문 자료([`inst/extdata/data_def.csv`](https://github.com/USEPA/LakeTrophicModelling/blob/master/inst/extdata/data_def.csv)): 토지이용 %, 유역·호수 면적, 최대수심, 같은 방문의 TN·TP·탁도·수온 |
| 세포수 | [`R/getCyanoAbund.R`](https://github.com/USEPA/LakeTrophicModelling/blob/master/R/getCyanoAbund.R): 남조류 세포수를 사분위로 LOW/MED/HIGH 분류 |
| 검증 | [`R/crossval_rf.R`](https://github.com/USEPA/LakeTrophicModelling/blob/master/R/crossval_rf.R): 무작위 80/20을 반복하는 조건부 랜덤포레스트 |

우리 적용
- 호수 **사이**의 차이를 같은 시점의 수질로 설명하는 모델이다. 우리 지점은 5개라 수역 특성 변수가 지점 ID와 겹친다.
- 같은 방문 TN·TP로 같은 방문 Chl-a를 설명하는 것은 7일 예측이 아니다. 영양염 피처 근거로 쓰지 않는다.

### 1.6 USEPA/provisional_habs — 관측 자료 품질관리 (main `bdf71ef`)

- [`README.md`](https://github.com/USEPA/provisional_habs/blob/main/README.md): 측정 시각·장치·수심·반복을 기록하는 긴 형식 자료다.
- 과거 범위 점검과 센서 검출한계 점검을 하되, **값은 바꾸지 않고 `notes`에 플래그만** 남긴다.
- [`.github/workflows/provisional_data_access.yaml`](https://github.com/USEPA/provisional_habs/blob/main/.github/workflows/provisional_data_access.yaml): 하루 3회 자료를 동기화한다(잠정 자료라 값이 바뀔 수 있다).

우리 적용: 극값을 삭제하지 않고 플래그만 단다. 과거 범위는 **학습 연도로만** 정한다.

---

## 2. 데이터 점검 결과 (이번에 실행)

원본 번들과 `data_external/`만 읽었다. 결과는 사실 기록이며 모델에는 아직 반영하지 않았다.

### 2.1 조류 측정 품질
- (지점, 날짜) 중복은 0건이다.
- **4개 속의 합 = 총 유해남조류 세포수**가 100% 일치한다. 대상은 4개 속 합계로 정의된 값이다.
- 세포수 0은 24.2%, 음수는 0건이다.
- 물리 범위 밖 값은 pH 11.9 1건뿐이다. 수온 1.5~33.0℃, Chl-a 0.4~223.6.
- 8일 이내 로그 변화 |Δ|>2(100배)가 147건, |Δ|>3이 12건이다. **실제 대발생·급락일 가능성이 커서 삭제하면 안 된다.** 플래그만 단다.
- pH가 직전 값과 같은 경우가 11%다(측정 해상도 0.1 때문으로 보인다). 센서 고착의 근거는 아니다.
- 결측: 회남은 수온·Chl-a가 28% 결측(세포수는 1%)이다. 해평·강정고령은 세포수가 8~9% 결측이다.

### 2.2 지점별 표본 (모델 분리 가능성)
- 5~10월 평가쌍은 지점·연도당 20~36개다.
- 5~10월 1만 이상 관측: 칠서 118, 강정고령 75, 물금매리 70, 회남 67, **해평 21**.
  - 지점별 단독 모델은 해평 피크를 배울 수 없다.
  - 물금매리는 2020년부터만 있다.
  - 지점 단위보다 **수역 단위(낙동강 4 / 회남 1)** 분리가 현실적이다.

### 2.3 Chl-a와 유해남조류 세포수
같은 날, 5~10월의 순위상관이다.

| 지점 | 상관 | 세포수 1만 이상일 때 Chl-a 중앙값 | 1천 미만일 때 |
|---|---|---|---|
| 해평 | 0.25 | 29.2 | 13.8 |
| 강정고령 | 0.44 | 26.1 | 10.4 |
| 칠서 | 0.56 | 34.1 | 14.1 |
| 물금매리 | 0.58 | 35.4 | 9.6 |
| 회남 | 0.64 | 11.7 | 4.9 |

- Chl-a는 남조류 외 조류까지 포함하는 생물량 지표다. 지점마다 관계가 다르다(해평은 거의 무관).
- **Chl-a를 목표 대신 쓰거나, Chl-a 예측 연구의 성능을 세포수 성능으로 옮기면 안 된다.** 입력 보조 신호로만 쓴다.
- 1만 이상 시료의 우점 속은 Microcystis 305 / Aphanizomenon 18 / Anabaena 14로, 약 90%가 Microcystis다. 제네바호 Planktothrix(약 14℃) 같은 종별 결과는 적용하지 않는다.

### 2.4 영양염(N·P)
- **수질측정망 채수:** 하천 7개 지점(칠곡·선산·다사·남지·현풍·물금·임해진)은 **주 1회**(간격 중앙값 7일), TN·TP를 모두 측정한다. 회남 인근 **대청댐5는 월 1회**(5~10월 간격 중앙값 32.5일, 2011~2026년 187회)라 7일 예측 신호로는 너무 성기다. 옥천은 주 1회지만 상류 하천 지점이다.
- **자동측정망 TN·TP 일자료 보유율**은 고령 0.68, 적포 0.73, 칠곡 0.71, 상동 0.77, 대청호 0.80, 현도 0.79, **양산(물금매리 인근) 0.23**이다. 해평·칠서 인근 자동측정소 파일에는 TP 항목이 없었다.
- **기존 결과:** V7에서 영양염 22개 피처는 LightGBM 전 변형을 악화시켰다. 시기별 순위상관도 모든 시기에 |r| ≤ 0.12였다(V9 사전 점검). 타이후호의 "봄 인, 여름·가을 질소" 패턴은 우리 자료에서 재현되지 않았다.
- **공표 지연:** 수질측정망 실험실 결과는 채수 후 수주 뒤에 공개될 수 있다(V7의 lag14 민감도 실험). 실제 운영에서는 기준일에 최신 영양염을 모른다.

### 2.5 수온·기상
- 현장 수온은 주 1회 오전 순간값이다. ASOS 기온은 일자료로 결측이 거의 없다(≤0.4%).
- **강수 `sumRn` 공란:** 58~67%가 공란이다. 공란인 날의 87%가 일조 5시간 이상이었으므로 **공란은 무강수**로 봐도 된다. v1은 관측 합과 관측 비율을 따로 넣었는데, 0 처리로 단순화해도 정보 손실이 없다.
- **일사량 `sumGsr`:** 구미(해평 대응)가 89%, 창원이 26% 결측이고, 2024년까지 연도별 보유율은 66~83%다. **주 입력으로 쓰지 않는다.** 일조시간 `sumSsHr`을 쓴다.

### 2.6 수문(유입·방류·정체)
- 댐 15개 일자료(2012-01-01~2026-09-11)는 수위·유입·방류의 결측이 0이다.
- **저수량은 `dam_operation_api`의 2026-06-10~09-11에만** 있다. 그래서 "저수량 ÷ 방류량" 체류시간은 과거 학습이 불가능하다. 수위·유입·방류 비율로 대리한다.
- **낙동강 보(구미·칠곡·강정고령·달성·합천창녕·창녕함안보)의 수위·방류 자료가 없다.** 낙동강 지점 정체의 직접 지표가 비어 있는 셈이다. WAMIS 하천 관측소 12곳 유량·수위(`data_external/wamis/`)로 일부 대체할 수 있지만, V5에서 단독 이득은 없었다.
- DataON 댐유역 하천 일평균수위는 승인 절차가 필요해 아직 미확보다(번들 README 안내).

### 2.7 상류 정보
- 낙동강 4지점은 하천 순서가 해평 → 강정고령 → 칠서 → 물금매리다. 주로 같은 요일(월요일 81%)에 채수한다.
- V5의 "다른 지점 조류" 피처는 R² +0.023±0.010, 다른 지점+WAMIS는 **R² +0.019±0.009로 7/7년 개선**이었다. 로그 지표 변화는 없었고 2025 확인에서 악화해 보류됐다. 일관된 원 단위 R² 이득이 있었던 드문 변수군이다.

### 2.8 CyanoLakes MASEA 재현 (로그 세포수, 지점 × 주차 계절평균, 폴드 이전 연도만)

| 모델 | 2018~24 로그 RMSE | 원 단위 NSE | R² | 2025 RMSE / NSE |
|---|---|---|---|---|
| 현재 선택(v1 50/50) | **0.797** | **0.209** | **0.297** | 0.634 / 0.006 |
| MASEA a=0.8 | 0.905 | 0.093 | 0.224 | 0.715 / 0.006 |
| 계절 증분(최신값 + 계절평균 차이) | 0.914 | −0.511 | 0.230 | 0.672 / −0.593 |
| 최신값 유지 | 0.929 | −0.200 | 0.244 | 0.697 / −0.426 |
| 계절평균만 | 0.958 | −0.026 | 0.113 | 0.768 / −0.123 |

- 2개 관측 평균(평활)만으로도 원 단위 NSE가 최신값 유지보다 크게 오른다(−0.20 → 0.09). 피크를 그대로 이어 쓰면 원 단위 오차가 커지기 때문이다.
- 단독으로는 현재 모델에 크게 뒤진다. 다만 현재 모델과 오차 구조가 다를 수 있어 E3에서 혼합 후보로만 본다.

---

## 3. 질문 1 — 유지 / 수정 / 신규

### 유지
| 항목 | 이유 |
|---|---|
| 로그 변화량 목표(log10(y+1) − 최신 로그값) | V1부터 직접 회귀보다 일관되게 좋았다 |
| TabPFN + LightGBM 50/50 로그 평균 | V2~V9의 모든 대안이 잡음 범위였다 |
| 시간순 7폴드(2018~2024), 5~10월 평가, 2025 참고 검증 | 참고 저장소 중 ACES만 같은 수준의 시간 봉인을 쓴다. 무작위·공간 분할은 쓰지 않는다 |
| 일자료 기준일 전날까지(d−1) | V8에서 당일 자료는 이득이 없었다. 규정상 허용이어도 기본값을 유지한다 |
| 극값 비삭제 | 1주 100배 변화 147건은 실제 사건이다. EPA 방식처럼 플래그만 단다 |
| 고정 선택·해시 보호·누수 테스트 | 재현성 실격 규정 대응 |

### 수정
| 항목 | 변경 |
|---|---|
| 선택 지표 | 로그 RMSE 단독 → **4개 순위 지표(로그 RMSE·MAE, 원 단위 R²·NSE)를 모두 기록**하고 수역별로 따로 집계. 지표 계산법이 확정되면 그 결합식으로 선택(E3) |
| 강수 공란 | 공란 87%가 맑은 날이므로 "공란 = 0"을 명시 규칙으로 문서화. 결과가 거의 같으면 단순한 쪽을 택한다 |
| 일사량 | `sumGsr`을 금지 목록에 추가(구미 89% 결측) |
| 외부 영양염 | 쓸 경우 기본 지연을 1일 → 14일로 바꿔 공표 지연을 반영 |
| 품질 플래그 | 과거 범위 플래그를 **학습 연도로만** 계산(EPA 방식 + 인과 규칙) |

### 신규 시험
- **E1** 수역별 모델과 통합 모델 비교·혼합 (Tick Tick Bloom 1·3위의 지역별 모델)
- **E2** 하천 순서·시차를 반영한 상류 지점 정보 재검증 (2위의 과거 전용 공간 지연, Shin 등 2020)
- **E3** 다지표 혼합 가중 선택 + MASEA 평활 기준선 결합 (CyanoLakes)
- 보류 후보는 8절 참고.

---

## 4. 질문 2 — 변수별 실제 활용 가능성

| 변수군 | 자료 현황 (2절) | 지금까지 결과 | 판단 |
|---|---|---|---|
| **계절별 질소·인** | 번들에 없음. 하천 수질측정망 주 1회(TN·TP 100%), 대청댐5 월 1회, 자동측정 일자료 68~80%(양산 23%). 공표 지연 있음 | V7 악화, 시기별 \|r\| ≤ 0.12 | **주 모델에서 제외.** 계절 × 영양염 상호작용은 E1의 하천 모델에서만 14일 지연으로 1회 시험할 수 있는 후순위 후보. 회남은 월 1회라 불가 |
| **수온 추세** | 현장 수온 주 1회(회남 28% 결측), ASOS 기온 완전 | 수준·추세·적산온도(gdd20/25)·고온 지속·냉각 신호 이미 반영(v2·v6) | **유지.** 새 변수는 추가하지 않는다. 회남은 더 긴 기온 창이 맞다(V8 진단). 별도 수역 모델(E1)에서 자연히 반영된다 |
| **강우 즉시 효과** (당일~3일) | ASOS 일강수(공란=0), 댐 강우 | V8: 하천은 큰비 뒤 급락, 회남은 급증(부호 반대). 통합 모델 피처로는 미활용 | **E1에서 재확인.** 수역별 모델이면 부호가 섞이지 않는다 |
| **강우 지연 효과** (7~14일, 호우 후 정체) | 같음 | v6 사건 피처(유량 첨두 경과일, 회복기 고온·일조) 반영, 잡음 범위 | 유지. E1에서 회남 모델의 기여만 확인한다 |
| **유입·방류·정체** | 댐 15개 일자료 완전. 저수량은 2026만. **보 자료 없음** | 체류시간 대리변수, 방류 비율, 방류 이상치(V9) 모두 잡음 범위 | 댐 기반 대리변수는 유지. 진짜 정체 지표(보 수위·방류)는 **자료 공백**. 외부 자료가 허용되고 확보되면 그때 1순위 |
| **상류 정보** | 낙동강 4지점 같은 요일 채수 | V5: R² +0.019, 7/7년. 로그 지표 변화 없음 | **E2로 재검증** |
| Chl-a | 현장 Chl-a 주 1회(세포수와 상관 0.25~0.64), 자동측정 일자료 | v1 입력. 자동측정은 V7에서 회남 −0.005±0.003 | 입력으로만 사용. 회남 모델(E1)의 선택 변형 1개로 자동측정 Chl-a 시험 |
| 일사량 | 결측 과다 | — | 사용 금지. 일조시간 사용 |

---

## 5. 질문 3 — 통합 모델과 지점·수역별 모델 비교 설계

**후보**

| 후보 | 학습 자료 | 근거 |
|---|---|---|
| A 통합(현재) | 5지점 전체 + 지점 ID | 기준 |
| B 수역별 | 낙동강 4지점 모델 + 회남 모델 | Tick Tick Bloom 1·3위, V8 부호 반대 |
| C 수역별 + 통합 혼합 | B와 A의 로그 평균(수역마다 가중 0.5 고정) | 3위의 모델 평균. 작은 표본의 분산 완화 |
| D 지점별(참고만) | 지점 하나씩 | 해평 1만 이상 21건, 물금매리 2020~ → 피크 학습 불가. **채택 대상에서 제외**하고 비교표에만 둔다 |

**비교 방법**
1. 폴드·학습 시작 연도·하이퍼파라미터는 현재와 같다(LightGBM 2017~, TabPFN 2012~). 회남은 2012~2015년 자료가 더 있다.
2. 각 후보의 예측을 5지점 전체로 합쳐 **기존과 같은 전체 지표**를 계산한다. 수역별 지표(낙동강 4 / 회남)도 따로 계산한다.
3. **채택 조건**
   - 해당 수역에서 로그 RMSE가 5/7년 이상 개선된다.
   - 평균 차이가 표준오차보다 크다.
   - 원 단위 NSE·R²의 평균이 −0.005보다 나빠지지 않는다.
   - 다른 수역의 지표는 변하지 않아야 한다. 수역별 교체는 해당 수역 예측만 바꾼다.
4. **표본 수 제약(실측):** 회남 학습쌍은 2012년부터 쓰면 2018 폴드 301쌍, 2024 폴드 609쌍이다. 그런데 LightGBM의 현재 학습 시작 연도(2017~)를 그대로 쓰면 **2018 폴드 55쌍, 2024 폴드 363쌍**뿐이다. 낙동강 모델은 2018 폴드 317쌍, 2024 폴드 1,413쌍이다.
   - 회남 LightGBM은 2012년부터 학습한다(사전 고정).
   - TabPFN은 원래 2012~이므로 변경이 없다.
   - 2018~2019 폴드는 표본이 작아 분산이 크다. 그래서 연도별 개선 횟수와 함께 2020~2024만의 평균도 보고한다. 선택은 7폴드 기준을 유지한다.

---

## 6. 질문 4 — 미래 정보 차단 규칙

참고 저장소에서 실제로 본 위험과 대응 규칙을 함께 적는다.

| 위험 | 저장소 사례 | 우리 규칙 / 점검 |
|---|---|---|
| 역방향 채움·양방향 보간 | CyanoLakes `resample('W').mean().bfill()` | 금지. 결측은 최신 과거값 + 경과일 + 결측 표시(ACES 방식). 기존 `test_causality`의 미래 변조 테스트로 점검 |
| 관측 시각 이후 기상 | 3위 HRRR 채수 ±12시간 | 일자료는 기준일 전날까지(d−1)가 기본. 당일 허용 변형은 별도 표시(V8 테스트 `test_sameday_v8`) |
| 무작위·공간 분할 | 1·2·3위, Nasios, EPA LTM | 시간순 폴드만 사용. 학습쌍은 `target_date < 폴드 연도 1월 1일` |
| 전 기간 통계 | ACES 금지 목록 `full_period_statistics` | 계절평균·과거 범위·스케일러·품질 플래그 기준은 **폴드 이전 연도로만** 계산(2.8 MASEA 재현도 이 규칙으로 계산) |
| 중심 이동평균 | ACES `centered_rolling` | 모든 창은 기준일에서 끝나는 후행 창 |
| 공간 지연의 동시점 누수 | 2위 `date < pdat` | 상류 지점 값은 **기준일 이하에 측정된 값만** 사용. 같은 날 채수값은 "측정 당일 이용 가능" 가정과 같은 수준으로 표시 |
| 미래 채수 일정 | (우리 고유) `target_date`, `lag_days` | 입력 금지(주최 측 확인 전). 간격 ≠ 7일은 미래의 미채수·경보를 반영한다(+0.05 대 −0.21 진단) |
| 실험실 공표 지연 | (우리 고유) 수질측정망 | 외부 영양염은 14일 지연을 기본으로 |
| 선택 누수 | ACES 봉인 연도 | 2025는 참고 검증일 뿐. 선택은 2018~2024만. 가설은 실험 전에 이 문서에 등록 |
| 잠정 자료 갱신 | EPA provisional | 번들 해시 고정(`manifest.json`). 재현 실행 시 해시 비교 |

변수를 새로 만들 때는 다음 세 가지 테스트를 반드시 통과해야 한다.
1. 기준일 이후 자료를 바꿔도 피처가 그대로인지(미래 변조 불변)
2. `target_date`를 바꿔도 피처가 그대로인지(목표일 비입력)
3. 학습 연도 밖 통계를 쓰지 않는지(폴드 경계)

---

## 7. 질문 5 — 가장 먼저 할 실험 3개

공통 채택 기준(이하 "기본 기준")
- 2018~2024 7폴드, 5~10월
- 기존 선택 대비 로그 RMSE 차이 < 0, 5/7년 이상 개선, |차이| > 표준오차
- 원 단위 NSE·R²의 평균 변화 ≥ −0.005
- 누수 테스트 통과, 보호 파일 해시 불변
- 2025는 참고로만 보고, 재선택하지 않는다
- 지표 계산법이 공지되면 그 결합식으로 기준을 바꾸고 이 문서에 기록한다

### E1. 수역별 모델 (우선순위 1)
- **가설:** 강우·체류의 효과가 하천과 호소에서 반대로 작용한다. 통합 모델은 사건 수가 적어 이 상호작용을 배우지 못한다. 수역별로 나누면 각 수역의 급증·급락 시점을 더 잘 맞힌다.
- **필요 변수:** 기존 v1 76개를 그대로 쓰고, 새 자료는 없다.
  - 변형 E1b: 회남 모델에 자동측정 대청호 Chl-a 3/7일 평균·추세(V7 `autowq`, 기준일 전날까지)를 추가
  - 변형 E1c: 하천 모델에 V8 부호 분리 강우 4개를 추가
- **검증:** 5절 A/B/C 후보 × TabPFN·LightGBM × 7폴드. 수역별·전체 지표, 큰비 기준일(>30mm) 오차를 따로 본다.
- **채택:** 기본 기준 + 5절 3항(수역별 교체 조건).
- **비용:** 낮음. `folds()`에 지점 필터만 추가하면 되고, TabPFN 약 20분.
- **참고 코드:** [1위 지역별 LightGBM/KNN](https://github.com/drivendataorg/tick-tick-bloom/blob/main/1st%20Place/notebooks/Model.py), [3위 위성×지역별 모델](https://github.com/drivendataorg/tick-tick-bloom/blob/main/3rd%20Place/train.py)

### E2. 상류 지점 정보 재검증 (우선순위 2)
- **가설:** 낙동강에서는 상류 지점의 최근 세포수와 그 변화가 하류의 7일 뒤 변화를 앞서 보여 준다. V5의 R² 7/7년 개선은 우연이 아니며, 하천 순서와 측정 경과일을 명시하면 로그 지표도 개선된다.
- **필요 변수(낙동강 지점만, 기준일 이하 측정값만, 회남은 결측):**
  - 바로 위 지점의 최신 로그 세포수와 경과일
  - 위 지점 − 자기 지점 차이
  - 위 지점의 직전 대비 변화량
  - 상류 지점 중 최대값
  - 상류 1만 이상 여부
- **검증:** 통합 모델(A)과 E1 하천 모델(B) 양쪽에 넣어 비교한다. 지점 순서를 섞은 대조군(무작위 지점 순서)도 함께 돌려 이득이 하천 연결 때문인지 확인한다.
- **채택:** 기본 기준. 대조군보다 좋아야 한다.
- **비용:** 낮음. V5 피처 코드를 재사용한다.
- **참고 코드:** [2위 `spatial_lag`(과거 전용)](https://github.com/drivendataorg/tick-tick-bloom/blob/main/2nd%20Place/src/get_data.py), 문헌 Shin 등(2020).

### E3. 다지표 혼합 가중 + 평활 기준선 (우선순위 3)
- **가설:** 원 단위 NSE는 피크를 그대로 이어 쓸 때 크게 손해 본다. 평활된 기준선(MASEA)을 작은 가중으로 섞으면 로그 지표 손실이 작은 상태로 원 단위 NSE·R²를 안정시킬 수 있다. 지표 결합식이 확정되면 같은 틀로 바로 최적 가중을 고를 수 있다.
- **필요 변수:** 새 피처 없음. 기존 CV 예측(TabPFN, LightGBM, E1·E2 후보)과 MASEA 예측(2.8절)만 쓴다.
- **검증:** 가중 격자(TabPFN/LightGBM/MASEA 0~1, 0.1 단위)를 **내포 방식**으로 고른다. 2018~Y−1에서 선택하고 Y에 적용하며, Y = 2019~2024다. 4개 지표를 각각 기록하고, 결합식 후보(4지표 순위 평균, 로그·원 단위 동일 가중)별 최적 가중을 비교한다.
- **채택:** 내포 검증에서 기본 기준을 만족하는 가중만. 결합식이 미확정인 동안에는 **로그 RMSE를 해치지 않는 가중**만 채택한다.
- **비용:** 매우 낮음. 학습이 없어 수 분이면 된다.
- **참고 코드:** [CyanoLakes `moving_average_seasonal_error_adjusted`](https://github.com/CyanoLakes/forecasting/blob/master/models.py), [계절평균의 인과 계산](https://github.com/CyanoLakes/forecasting/blob/master/forecast.py)

---

## 8. 보류·제외 목록

| 항목 | 상태 | 이유 / 재개 조건 |
|---|---|---|
| 낙동강 보 수위·방류 | 보류(최우선 외부 자료) | 정체의 직접 지표. 외부 자료 허용이 확인되고 K-water 보 운영 자료를 확보하면 |
| 예측구간·conformal(ACES) | 후순위 | 점 예측 채점이다. 원 단위 지표로 확정되면 구간 상단 활용을 1회 시험 |
| 계절 × 영양염 상호작용 | 후순위 | V7 악화, 상관 미약. E1 하천 모델에서 14일 지연으로 1회만 |
| 위성 자료 | 제외 | 번들 외, 하천 폭 대비 해상도, 허용 미확인 |
| DEM·토지이용·수역 특성 | 제외 | 지점 5개라 지점 ID와 같은 정보 |
| 지점별 단독 모델 | 제외(비교표만) | 해평 피크 21건, 물금매리 2020~ |
| 무작위·공간 교차검증 | 사용 금지 | 7일 예측의 시간 누수 |
| 기준일 당일 일자료 | 기각(V8) | 잡음 범위 |
| 경보 상태 피처 | 보류(V9) | 원 단위 지표로 확정되면 LightGBM NSE 6/7년 개선을 재검토 |

## 9. 참고 코드 링크

- CyanoLakes: [`forecast.py`](https://github.com/CyanoLakes/forecasting/blob/master/forecast.py), [`models.py`](https://github.com/CyanoLakes/forecasting/blob/master/models.py), [`baseline.py`](https://github.com/CyanoLakes/forecasting/blob/master/baseline.py)
- Tick Tick Bloom: [1위 `Model.py`](https://github.com/drivendataorg/tick-tick-bloom/blob/main/1st%20Place/notebooks/Model.py), [2위 `src/get_data.py`](https://github.com/drivendataorg/tick-tick-bloom/blob/main/2nd%20Place/src/get_data.py), [2위 `main_preds.py`](https://github.com/drivendataorg/tick-tick-bloom/blob/main/2nd%20Place/main_preds.py), [2위 `src/mod.py`](https://github.com/drivendataorg/tick-tick-bloom/blob/main/2nd%20Place/src/mod.py), [3위 `train.py`](https://github.com/drivendataorg/tick-tick-bloom/blob/main/3rd%20Place/train.py), [3위 `download_hrrr_forecasts.py`](https://github.com/drivendataorg/tick-tick-bloom/blob/main/3rd%20Place/download_hrrr_forecasts.py)
- ACES: [`protocols/uq_iso_hab_clean_v2.yml`](https://github.com/donghyunlee-lab/aces-hab-forecasting/blob/main/protocols/uq_iso_hab_clean_v2.yml), [`src/data/preprocessor.py`](https://github.com/donghyunlee-lab/aces-hab-forecasting/blob/main/src/data/preprocessor.py), [`scripts/eval_calibration_clean_v2.py`](https://github.com/donghyunlee-lab/aces-hab-forecasting/blob/main/scripts/eval_calibration_clean_v2.py)
- Nasios: [`download_satellite_PlanetaryComputer.ipynb`](https://github.com/IoannisNasios/HarmfulAlgalBloomDetection/blob/master/download_satellite_PlanetaryComputer.ipynb), [`download_clima_noaahrrr.ipynb`](https://github.com/IoannisNasios/HarmfulAlgalBloomDetection/blob/master/download_clima_noaahrrr.ipynb), [`lgbmNN_ee_gkf_S_v42g.ipynb`](https://github.com/IoannisNasios/HarmfulAlgalBloomDetection/blob/master/lgbmNN_ee_gkf_S_v42g.ipynb)
- EPA LakeTrophicModelling: [`R/crossval_rf.R`](https://github.com/USEPA/LakeTrophicModelling/blob/master/R/crossval_rf.R), [`R/getCyanoAbund.R`](https://github.com/USEPA/LakeTrophicModelling/blob/master/R/getCyanoAbund.R), [`inst/extdata/data_def.csv`](https://github.com/USEPA/LakeTrophicModelling/blob/master/inst/extdata/data_def.csv)
- EPA provisional_habs: [`README.md`](https://github.com/USEPA/provisional_habs/blob/main/README.md), [`.github/workflows/provisional_data_access.yaml`](https://github.com/USEPA/provisional_habs/blob/main/.github/workflows/provisional_data_access.yaml)
