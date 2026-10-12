# Claude 인계 프롬프트

## 최신 상태: V8 (2026-10-12)

`research/STRATEGY_RESEARCH_2026-10-11.md`(환경부 녹조 자료·남조류 글, 측정 시각 진단)의 T1을 검증했다.
상세 `results/EXPERIMENT_REPORT_V8.md`, 산출물 `artifacts/cv8/`, 코드 `algae/sameday.py`, `algae/exp8.py`.

- 진단: ASOS·댐 일자료는 0~24시 달력일. 수질측정망 채수 시각은 10~12시가 가장 많다.
  직전 세포수가 1만 이상이면 채수 간격 중앙값 4일(경보 단계가 채수 빈도에 드러남).
- 기준일 당일 일자료 4변형(full/extra/minta/signed) × LightGBM·TabPFN × 2018~2024: 최저 −0.0012 ± 0.0054,
  모두 잡음 수준. **미채택.** 선택·제출 파일 불변. 테스트 24개 통과.
- 당일 30mm 이상 강우 뒤 하천 지점은 급락, 회남은 급증한다(부호 반대). 피처로 넣어도 모델이 쓰지 않았다(사례 60쌍).
- 남은 방향: R1 경보 상태기계 + 관리 방류, T2 지점별 열 시차. `target_date`(간격)는 주최 측 확인 전 사용 금지.

### V9 (2026-10-12): 경보 상태·사건 위치 — 미채택

`results/EXPERIMENT_REPORT_V9.md`, `algae/alert.py`, `algae/exp9.py`, `artifacts/cv9/`. 테스트 27개 통과.
혼합 기준 전부 잡음 범위(RMSE −0.002~+0.005). LightGBM 단독은 NSE +0.027(6/7)이지만 로그 RMSE +0.006 악화.
지표 확정 후(원 단위면) 재검토 후보. V2~V9 피처 추가 이득은 소진된 것으로 판단.

### V8 이후 추가 확인 (2026-10-12)

- 주최 측 Q&A 답변: 분할은 README 기준(학습 ~2024 / 검증 2025 / 평가 2026 5~10월). 분류 부과제 삭제, 회귀 100%,
  10,000 초과 F1은 참고지표(순위 미반영). **회귀 지표 계산 방법(원 단위/로그, R² 정의)은 출제 기관과 협의 중, 미확정.**
- 대회 페이지 목표(R² 0.6·NSE 0.5)의 근거 0.880은 당일 변수 + 무작위 8:2 분할 결과다. 우리 자료로 재현하면
  무작위 분할 NSE 0.48~0.55, 시간순 분할 0.16으로, 무작위 분할이 크게 부풀린다.
- 현재 v1 모델(5~10월): 원 단위 NSE 0.21·R² 0.30(2018~24), 로그 공간 NSE 0.50·R² 0.53(2025는 0.60·0.65).
- 고농도 예측 배율 후처리(k×, 임계 1천~1만): 사후 그리드에서도 NSE가 오르지 않고, 내포 검증 2019~24 NSE −0.063. **기각.**
  피크 오차는 크기 과소보다 시점 불일치 때문이다.

## 이전 상태: V7 (2026-10-09)

외부 신규 관측을 수집해 검증했다. 상세는 `results/EXPERIMENT_REPORT_V7.md`, 산출물은 `artifacts/cv7/`.

- 수집: 수질측정망 채수 자료(TN·TP·PO4 등, 2011~2026-07, `data_external/nier_wq/`),
  수질자동측정망 일자료 15개소(Chl-a·pH·DO 등, 2012-07~2026-10-08, `data_external/nier_auto/`).
  보 수위는 방류량이 없어 미사용, 과거 기상예보는 인증키가 없어 미수집.
- 코드: `algae/nutrients.py`, `algae/autowq.py`, `algae/exp7.py`, `scripts/convert_nier_wq.py`,
  `scripts/fetch_nier_auto.py`. 누수 테스트 4개 추가, 전체 21개 통과.
- 결과: LightGBM은 신규 피처로 전부 악화. TabPFN 교체 최저 후보 `blend_tab_nutr`는
  2018~2024 RMSE −0.0051 ± 0.0050(5/7)이지만 NSE −0.014. 2025 참고 검증 0.63363(기존 0.63350).
  **미채택.** 선택·제출 파일 불변(exp6 verify 통과).
- 모든 TabPFN 교체 후보가 같은 연도 패턴(2019 개선, 2022·2024 악화)을 보인다. 자료 고유 효과가 아니라
  피처 추가의 공통 효과로 해석했다.
- 남은 방향: 과거 발표 기상예보(사용자 API허브 키 필요). 외부 자료 사용 허용 여부는 여전히 미확인.
- `.venv`에 openpyxl 추가 설치.

## 이전 상태: V6 (2026-10-05)

`research/STRATEGY_RESEARCH_2026-10-05.md`의 전략을 실제로 구현·검증했다.
먼저 `results/EXPERIMENT_REPORT_V6.md`, `artifacts/cv6/run_summary.json`을 읽고 완료 실험을 반복하지 마.

- 30개 신규 피처(사건 11, 종/수질 변화 6, 일별 수온 이력 8, 세포수 Kalman 상태 5).
- LightGBM 16개 설정 × 7연도, TabPFN 2설정 × 7연도 = 126개 과거 비교. 17개 검사 통과.
- 사건/LDS 가중치, 이력 마스킹 376쌍, 원자료 블록 혼합 155쌍, 일별 로그 변화율,
  내부 OOF 급증/유지/급락 잔차 보정을 포함한다. 원자료 블록 혼합을 공개 SMOGN이라고 부르지 마.
- 최저 로그 RMSE 후보: TabPFN `environment`(사건+종변화+수온이력) + 기존 LightGBM 50/50.
  2018~2024 평균 RMSE 0.79579(기존 0.79733), 차이 −0.00154 ± 표준오차 0.00491, 3/7년 개선.
  NSE 0.21012(기존 0.20913). 후보는 2025년 새 예측 전에 `artifacts/cv6/frozen_candidate.json`에 기록했다.
- 2025년 참고 검증: 혼합 RMSE 0.61969(기존 0.63350), NSE 0.01643(기존 0.00599).
  새 TabPFN 단독 RMSE 0.59115도 관찰됐지만 이 값으로 재선택하지 않았다.
- **미채택:** 과거 연도 개선이 변동 범위 안이며 일관되지 않는다. 기존 선택·제출 파일 불변.
- 저장 모델: `artifacts/cv6/2025_tabpfn_environment/model.joblib`, 재로딩 예측 일치 확인.
- v6 코드는 `algae/features_v6.py`, `algae/augment_v6.py`, `algae/exp6.py`, 보고 생성은
  `python -m scripts.write_report_v6`. 전체 결과·계획·해시·검증은 `artifacts/cv6/`.
- R²를 Pearson²로 정한 이전 V5 해석은 확정 규칙이 아니다. v6는 r2_score/NSE와 Pearson²를 분리한다.
- 2025는 이미 여러 번 관찰된 참고 검증이다. 신규 독립 holdout으로 주장하지 마.
- 남은 방향: 실제 신규 정보(TN/TP·고빈도 생물량·과거 발표 예보) 확보 가능성.
  이번에 외부 자료를 확보하지 않았으며, 베이지안 전체 과정모형·GRU-D 등의 대규모 탐색도 하지 않았다.

아래는 이전 버전 기록이다. 최신 결론과 충돌하면 V6 보고서의 명시적인 정정을 따른다.

이 폴더의 유해남조류 7일 예측 대회 프로젝트를 이어서 맡아줘. 먼저 README.md, FULL_DATA_STRATEGY.md, results/EXPERIMENT_REPORT.md(있으면), results/run_summary.json, artifacts/selection.json을 읽고 실제 파일 상태를 확인해줘. 완료된 실험을 처음부터 다시 돌리지 말고, 남은 검증을 마무리한 뒤 다음 개선 실험을 제안해줘. 설명과 보고는 한국어로 해줘.

## 환경과 자료

- 작업 폴더: C:\2026데이터분석경진대회
- Windows PowerShell, Python: .venv\Scripts\python.exe
- i7-1260P, RAM 약 16GB. 현재 PyTorch는 CPU만 사용 가능.
- 원본: 02_dataset_bundle_algae2026.zip, 03_collect_dataset.py, 06_data_dictionary.xlsx. 원본은 변경하지 마.
- 대회: https://aida.kisti.re.kr/competition/main/problem/PROB_000000000004602/detail.do
- 실제 번들 README 기준 회귀 100%, train 2310행(~2024), valid 256행(2025), test 124행(2026), excluded 64행. 웹의 오래된 예시 설명과 다르므로 실제 번들 기준으로 구현했다.
- 목표는 기준일 +7일에 가장 가까운 ±3일 이내 관측값. 지점 5개: HP, GJGR, CS, MGMR, DCHN.
- 평가 지표는 log10(y+1) RMSE/MAE와 원 단위 R²/NSE. 공식 결합 점수 방식은 미확인.

## 구현과 검증 완료 내용

- algae/features.py: 76개 특징. 조류·달력·지점 25, 수질·종 구성 18, 기상 17, 수문 16.
- 조류·수질은 기준일까지, 일별 기상·수문은 전날까지만 사용한다. 조류 측정 결과를 측정 당일 이용 가능하다는 가정이 있다.
- 결측 조류는 최신 유효 관측과 경과일을 사용한다. 강수 결측은 0으로 단정하지 않고 관측 합/관측 비율을 입력한다.
- 수문은 낙동강 지점에 안동·임하, 회남에 대청·용담 신호를 별도 입력한다. 전체 하천망/이동 시간 모델은 아니다.
- target_date, lag_days, split은 특징에서 제외. test/excluded 정답은 지도학습 테이블에 NaN으로 둔다.
- 각 검증 연도의 1월 1일 전에 target_date가 있는 공식 train만 학습한다. 2025 모델은 경계를 넘는 train 5행을 제외한 최대 2305행으로 학습한다.
- 2026 관측도 각 기준일까지 알려진 과거 입력으로만 사용한다. 미래 관측을 사용하지 않는다.
- tests/test_causality.py의 5개 테스트 통과: 미래 변조, 목표 날짜 변조, 당일 기상·수문 차단, 조류 결측 대체, 학습 정답 날짜 경계.
- 11개 설정 × 2021~2024 4개 연도 = 44회 CV 완료. artifacts/cv_scores.csv에 전부 기록.
- CatBoost/LightGBM/TabPFN 각각 미래 로그값 직접 예측과 최신 로그값 대비 변화량 예측을 비교했다.
- TabPFN은 공개 v2 가중치, CPU, n_estimators=1. 최신 TabPFN에 대한 결론으로 일반화하지 마. 가중치/라이선스/해시는 .cache/tabpfn에 있다. 설치 버전은 requirements-lock.txt와 artifacts/environment.json 참고.

## 고정된 선택과 실제 결과

2025년 확인 전에 artifacts/selection.json에 고정했다. 이 파일을 덮어쓰거나 2025 결과로 기존 선택을 소급 변경하지 마.

- 선택: tabpfn_delta_full_all_2012 50% + lightgbm_delta_full_all_2017 50%, 로그 예측 평균.
- 과거 4개 연도 5~10월 평균 로그 RMSE: 선택 앙상블 0.738295, TabPFN delta 0.745223, LightGBM delta 2017~ 0.756067, LightGBM delta 전체 0.780823, CatBoost delta 0.780852.
- 2025 전체 256행: 앙상블 RMSE 0.765398, MAE 0.555037, R² 0.107597. 최신값 유지 기준선 RMSE 0.891034.
- 2025 5~10월 151행: 앙상블 RMSE 0.633503, MAE 0.476911, R² 0.005990. 기준선 RMSE 0.696975, MAE 0.489556, R² -0.425812.
- 2025 5~10월 TabPFN 단독 RMSE 0.607861로 앙상블보다 좋았다. 이 사실은 보고하되 같은 holdout을 보고 선택을 바꾸고 독립 검증이라고 주장하지 마.
- 원 단위 R²가 낮아 급증·극단값 예측에는 개선 여지가 크다.

## 산출물과 마무리 절차

- artifacts/holdout/: 고정 후보 3개 저장 모델 model.joblib, 설정, 예측, 지표. 저장 후 다시 읽은 예측 일치 확인 완료.
- results/submission_draft.csv: 124행 제출 초안 생성 완료. 키·순서·비음수 정수 검사 완료. 외부 제출하지 않았다.
- results/validation_scores.csv, validation_predictions.csv, validation_2025.png, run_summary.json 생성 완료.
- 2025 valid의 최종 재학습 허용 여부가 불명확해서 제출 초안은 2024까지의 공식 train으로만 학습했다.
- 원본 ZIP에서 특징을 재생성하는 독립 추론 검증도 완료했다. submission_replayed.csv와 초안의 124개 예측 및 파일 바이트가 정확히 일치하며, run_summary.json의 independent_replay_identical은 true다.
- results/EXPERIMENT_REPORT.md 생성과 validation_2025.png 육안 확인까지 완료했다. 현재 필수 마무리 작업은 없다. 다음 단계는 결과 검토와 별도 버전의 개선 실험이다.

필요한 경우 순서대로 실행:

```powershell
& .venv\Scripts\python.exe -m algae.predict --output results\submission_replayed.csv
& .venv\Scripts\python.exe scripts\verify_submission.py
& .venv\Scripts\python.exe scripts\write_report.py
```

EXPERIMENT_REPORT.md가 있으면 수치·한계와 그림을 검토하고 사용자에게 결과를 설명해줘. 이후 개선 실험은 기존 결과와 별도 버전으로 기록하고 과거 시계열 CV로 판단해줘. 우선 검토할 항목은 급증 사례의 과소예측, 지점별 오차, TabPFN 구성 확대의 CPU 비용, 수문 연결·지연 특징이다. 공식 평가식과 valid 재학습 규칙을 확인하면 그에 맞춰 최종 제출 모델을 정리할 수 있다. 사용자 승인 없이 외부 제출하지 마.

## 버전 2 이후 상태 (추가 기록)

버전 1 검증을 재실행해 전부 통과했다. 테스트 5개, 독립 추론 재현, 제출 초안 바이트 일치를
확인했고 v1 코드·선택·초안의 SHA-256은 지금도 불변이다.

이후 별도 버전으로 개선 실험을 수행했고 결과는 `results/EXPERIMENT_REPORT_V2.md`에 있다.

- 특징 54개 확장: **기각.** LightGBM 6폴드·TabPFN 4폴드 모두 표준오차를 넘지 못했고,
  두 모델이 특징 묶음 순위를 정반대로 매긴다.
- 역변환 편향 보정: **기각.** 오프셋을 전년도에서 고르면 전이되지 않는다.
  누적 OOF에서 보였던 +0.05 이득은 같은 데이터에서 고른 값이라 착시였다.
- `results/submission_refit2025.csv`: 고정 선택 그대로 2025년까지 학습한 두 번째 제출 후보.
  최근 연도 추가 효과는 대리 검증에서 3/4 폴드 개선(-0.0118)이다. 외부 제출은 하지 않았다.
- 평가 방식 정정: 번들 README 기준 회귀 100% 단일 과제이고, 순위에 원 단위 R²·NSE가 함께
  반영된다. 웹 개요의 "회귀 70% + 분류 30%"는 적용되지 않는다.

다음 후보는 보고서 5절에 있다. 우선순위는 TabPFN `n_estimators` 확대, 폴드 확대,
DataON 하천 수위 보강이다. 선택 기준 잡음이 0.01 수준이므로 그보다 작은 차이로
고정 선택을 바꾸지 마라.

## 버전 3 이후 상태 (추가 기록)

TabPFN `n_estimators` 1 → 4 실험을 완료했다. 결과는 `results/EXPERIMENT_REPORT_V3.md`에 있다.

- `algae/tabpfn_ensemble.py`, 산출물은 `artifacts/cv3`. v1/v2 코드·선택·초안은 건드리지 않았다.
- 2021~2024 4폴드 평균 RMSE: n=1 0.745223 → n=4 0.751113 (악화, t=0.87로 표준오차 1배 미만).
  연도별 방향도 섞여 있고(2021·2023 개선, 2022·2024 악화), CPU 비용은 3.26배로 늘었다.
- **기각.** `artifacts/selection.json`의 `tabpfn_delta_full_all_2012`(n_estimators=1) 유지.
- 남은 후보: 폴드 확대(2017~2024, 8폴드), DataON 하천 수위 보강. n_estimators를 더 키우는
  방향은 방향성 자체가 일관되지 않아 우선순위를 낮춘다.

## 버전 4 이후 상태 (추가 기록)

폴드 확대(2017~2020년 추가)를 완료했다. 결과는 `results/EXPERIMENT_REPORT_V4.md`에 있다.
`artifacts/cv`에 11개 설정의 2017~2020년 fold를 추가했고 `artifacts/cv_scores.csv`를
갱신했다(v1/v2/v3 코드·선택·초안은 건드리지 않음).

- 11개 설정 중 `lightgbm_delta_full_all_2017`만 2017년 fold가 구조상 불가능해 7폴드,
  나머지는 8폴드다. 폴드 수가 다른 설정을 전체 평균으로 그대로 비교하면 착시가 생긴다
  (2017년이 전 모델 공통으로 쉬운 해라서, 2017년을 가진 설정이 유리해 보인다).
- 같은 2018~2024 7폴드로 맞춰 비교하면 원래 선택 순위(`tabpfn_delta_full_all_2012` 1위,
  `lightgbm_delta_full_all_2017`이 트리 계열 중 1위)가 그대로 유지된다. **기각할 이유 없음,
  선택 유지.**
- 새 사실: 2018~2020년은 네 가지 모델·여섯 가지 특징 묶음 전부에서 2021~2024년보다
  체계적으로 어렵다(RMSE 약 0.84~0.96 대 0.64~0.84). 선택에 쓴 구간이 비교적 쉬웠을 수
  있다는 뜻이다.
- 참고(선택 미변경): 2018~2020 CV 예측으로 "TabPFN 단독 대 50/50 블렌드"를 다시 보면
  2018·2020년은 단독이 더 좋고 2019년은 거의 동일하다. 2025 홀드아웃까지 합치면 4번 중
  3번 단독이 더 좋고 블렌드가 이긴 건 선택에 쓴 2021~2024 CV 한 번뿐이다. 핸드오프 규칙상
  같은 홀드아웃 반복 관찰로 선택을 바꾸지 않으므로 `selection.json`은 그대로 두었고,
  TabPFN 단독 채택 여부는 사용자 판단으로 남겼다.
- 남은 후보: DataON 하천 수위 보강이 유일하게 아직 시도하지 않은 항목이다.

## 버전 5 이후 상태 (추가 기록)

공식 지표 재정의와 급증 대응 실험. 결과는 `results/EXPERIMENT_REPORT_V5.md`, 코드 `algae/exp5.py`,
산출물 `artifacts/cv5*`. 선택·v1~v4 산출물 불변.

- 평가 기준 정정: 평가 구간은 2026 **5~10월**. R²(피어슨²)와 NSE(1−SSE/SST)를 분리해 계산한다.
  분류는 순위 미반영, 참고지표는 >10,000 cells/mL F1. 비교는 2018~2024 7폴드 5~10월 평균으로 한다.
- v1 블렌드 CV: RMSE 0.797, R² 0.297, NSE 0.209. 목표(R² 0.6, NSE 0.5)와 거리가 크다.
  순위 지표 전체로는 블렌드가 TabPFN 단독보다 약간 낫다(잡음 수준).
- 기각: Tweedie/가중/분위수/√ 손실, 최신값 원 단위 혼합. NSE는 연 5건 남짓의 급증·급락이 좌우한다.
- 다른 지점 조류 특징: R² +0.023±0.010(블렌드), NSE·로그 지표 변화 없음. 보류 후보.
- 다음: DataON 하천 수위(사용자 직접 다운로드 필요), 급락 신호 특징.
- v5b WAMIS: `scripts/fetch_wamis.py` → `data_external/wamis/`(12개 관측소, 2012~2026-10-04 유량·수위 스냅샷),
  특징 `algae/wamis.py`. WAMIS 단독은 개선 없음. 다른 지점+WAMIS 블렌드는 CV R² +0.019(7/7)였지만
  2025 확인에서 NSE·R²가 악화 → 미채택. 외부 데이터 사용 가능 여부는 대회 Q&A 미확인.
- v5c 고온×정체 상호작용 특징 8개(기온/유량 등): 블렌드 기준 전 지표 악화 → 기각. 보고서 6절.
- v5d 결측 대체 검토: 예측 시점 결측이 거의 없음(기준일 세포수 1.4%, SSE 기여 0%; 수질 결측은 DCHN 2012~15뿐).
  수온 대체 시험 결과 동일. 병목 아님. 보고서 7절.
- v5e 측정 사이 보간 가상 학습쌍(`algae/augment.py`, 로그선형/수온·유량 가중 2종, 7,726쌍): LightGBM 2012~는
  RMSE −0.012(7/7)로 개선되나 앙상블·TabPFN에서는 이득 없음(NSE 소폭 악화) → 기각. 보고서 8절.
