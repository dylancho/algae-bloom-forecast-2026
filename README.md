# 유해남조류 7일 예측 실험

전체 번들의 공식 train/valid/test 구분을 보존하며, 지점별 기준일에 이용 가능한 자료만으로 세포수를 예측한다. 설계 배경은 [FULL_DATA_STRATEGY.md](FULL_DATA_STRATEGY.md), 실제 실행 결과는 생성 후 `results/EXPERIMENT_REPORT.md`에서 확인한다.

## 환경

프로젝트 전용 `.venv`를 사용한다. `requirements-lock.txt`는 실제 설치 버전을 고정한다. 현재 실행 환경은 Windows, Intel i7-1260P, RAM 약 16GB, CPU 추론이다. 코어 모델 라이브러리는 CatBoost, LightGBM, TabPFN이다.

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-lock.txt
& .venv\Scripts\python.exe scripts\download_tabpfn.py
```

TabPFN 실험은 공개 `Prior-Labs/TabPFN-v2-reg`의 `tabpfn-v2-regressor-v2_default.ckpt`, CPU, `n_estimators=1`을 사용한다. 최신 TabPFN에 대한 실험 결과로 해석하지 않는다. 모델 가중치와 라이선스는 `.cache/tabpfn`에 저장되고 SHA-256이 기록된다. 데이터는 로컬에서 처리하며 클라우드 추론은 사용하지 않는다.

## 실행

저장소 최상위에서 실행한다.

```powershell
& .venv\Scripts\python.exe -m algae.experiment build
& .venv\Scripts\python.exe -m unittest discover -s tests -v
& .venv\Scripts\python.exe scripts\run_cv.py --models catboost lightgbm
& .venv\Scripts\python.exe scripts\run_ablations.py
& .venv\Scripts\python.exe scripts\run_cv.py --models tabpfn --timeout 900
& .venv\Scripts\python.exe -m algae.experiment summarize
& .venv\Scripts\python.exe -m algae.selection
& .venv\Scripts\python.exe -m algae.finalize
& .venv\Scripts\python.exe -m algae.predict --output results\submission_replayed.csv
& .venv\Scripts\python.exe scripts\verify_submission.py
& .venv\Scripts\python.exe scripts\write_report.py
```

`algae.selection`은 2021~2024년 결과만으로 후보·혼합 비중을 고정한다. 이미 `artifacts/selection.json`이 있으면 덮어쓰지 않는다. `finalize` 이후 2025년 결과를 보고 같은 holdout에서 재선택하는 것을 방지하기 위한 동작이다. CV 각 작업은 성공 결과가 있으면 재사용한다. 의도적으로 실험 설정을 바꿀 경우 작업 태그와 기록을 구분해야 한다.

`finalize`는 사전에 고정한 후보를 2025년에 확인하고 모델을 저장한다. 최종 제출 초안은 이 모델로 2026년 test를 예측한다. valid를 학습에 포함하는 규칙이 확인되지 않았으므로 재학습에 넣지 않는다. 외부 제출은 수행하지 않는다.

## 특징과 누수 제한

- 총 76개 입력: 조류·달력·지점 25개, 수질·종 구성 18개, 기상 17개, 수문 16개.
- 조류·수질: 기준일 이하의 최신 유효 관측과 경과일. 기준일 분석 결과를 이용할 수 있다는 날짜 기준 가정이 있다.
- 기상·수문: 기준일 전날까지의 관측. 7/14일 통계에는 미래 관측을 포함하지 않는다.
- 정답일, 실제 예측 간격, split, 미래 관측은 입력 변수에서 제외한다.
- 학습 행은 fold의 1월 1일 이전에 target_date가 있는 공식 train만 사용한다.
- test/excluded의 정답은 지도학습 테이블에 넣지 않는다. 2026년의 과거 관측은 각 예측 시점까지 공개된 입력이라는 가정하에 사용된다.
- 강수 공란은 0으로 단정하지 않는다. 관측된 값의 합과 관측 비율을 별도로 입력한다. 원자료 정의가 확인되면 별도 버전으로 개선할 수 있다.
- 2026년에만 있는 보조 댐운영, 측정 체계가 달라진 일사량, 대부분 결측인 독소·냄새물질은 핵심 입력에서 제외한다.
- 낙동강 4지점에는 공통 상류 신호인 안동·임하댐을 별도 입력으로 둔다. 회남에는 대청·용담댐을 별도 입력으로 둔다. 두 댐의 유량을 합산하지 않는다. 이것은 보수적인 첫 수문 모델이며 완전한 하천망·이동시간 모델은 아니다.

수문 관계 참고: [K-water 낙동강 시설 설명](https://www.kwater.or.kr/busi/sub02/facilitiespresent03Page.do?s_mid=1509), [금강 시설 설명](https://www.kwater.or.kr/busi/sub02/facilitiespresent02Page.do?s_mid=1511). 현지 보·하천 유량은 후속 확장 대상이다.

## 실험 정의

1. 기본 비교: 3개 모델 × 직접 로그 회귀/로그 변화량 회귀 × 2021~2024년.
2. 변화량의 기준은 최신 유효 세포수다. 그 관측의 경과일을 입력에 포함한다.
3. CatBoost: 600 trees, depth 5, learning_rate 0.04, L2 8. LightGBM: 500 trees, 15 leaves, depth 5, min_child_samples 30, learning_rate 0.035, L2 10. 고정된 시작 설정이며 holdout으로 early stopping하지 않는다.
4. 4개 과거 연도의 5~10월 로그 RMSE 평균으로 가장 좋은 트리 후보를 고르고, 특징 묶음/4~11월 학습/2017년 이후 학습을 추가 비교한다.
5. 최상위 완료 후보 3개에서 단독 또는 두 모델의 25/50/75% 로그 혼합을 비교한다. 0.002보다 작은 개선으로 혼합 복잡도를 늘리지 않는다.
6. 선택 후 2025년을 한 번 확인한다. 공식 지표 결합식은 미확정이므로 선택에 쓴 로그 RMSE를 공식 종합 점수라고 부르지 않는다. 원 단위 R²/NSE, MAE와 지점별 결과도 기록한다.

## 산출물

- `artifacts/features/manifest.json`: 변수 목록, 원본 해시, 처리 가정.
- `artifacts/features/provenance.csv`: 행별 원천 날짜.
- `artifacts/cv/`: 각 실험의 설정·예측·지표.
- `artifacts/selection.json`: 2025년 확인 전 고정한 선택.
- `artifacts/holdout/`: 2025년 검증과 저장 모델.
- `results/validation_scores.csv`, `validation_predictions.csv`: 검증 지표와 예측.
- `results/submission_draft.csv`: 제출 양식과 같은 124행 초안.
- `results/run_summary.json`: 실행 요약과 해시.

저장 모델 재로딩 예측, 제출 행·키·순서·정수·비음수 검사와 미래 관측 변조 검사를 수행한다. `submission_replayed.csv`는 원본 ZIP에서 특징을 다시 만든 독립 추론 결과이므로 제출 초안과 비교할 수 있다.

## 버전 2 개선 실험 (별도 기록)

버전 1의 코드·`artifacts/selection.json`·`results/submission_draft.csv`는 변경하지 않는다.
실행 결과와 해석은 [results/EXPERIMENT_REPORT_V2.md](results/EXPERIMENT_REPORT_V2.md)에 있다.
결론은 특징 확장·역변환 보정 모두 **기각**이며, 고정 선택을 바꿀 근거는 없다.

```powershell
& .venv\Scripts\python.exe -m algae.features_v2
& .venv\Scripts\python.exe -m unittest discover -s tests -v
& .venv\Scripts\python.exe scripts\run_cv_v2.py --models lightgbm tabpfn
& .venv\Scripts\python.exe -m algae.calibrate --output artifacts\calibration_lgbm_v1.csv
& .venv\Scripts\python.exe scripts\test_recency_value.py
& .venv\Scripts\python.exe scripts\build_refit_submission.py
& .venv\Scripts\python.exe scripts\write_report_v2.py
```

- `algae/features_v2.py`: v1 특징 76개를 그대로 둔 채 54개를 더한다. 시점 규칙은 v1과 같다.
- `algae/exp2.py`: 학습·평가 코드는 `algae.experiment`에서 그대로 가져오고 특징표만 바꾼다.
  산출물은 `artifacts/cv2`로 나가며 v1 기록을 덮어쓰지 않는다.
- `algae/calibrate.py`: 로그 오프셋을 예측 연도 Y가 아닌 Y-1년에서만 고르는 내포 검증이다.
- `scripts/build_refit_submission.py`: 고정 선택 그대로, 학습 창만 2025년까지 넓힌
  `results/submission_refit2025.csv`를 만든다. 초안은 건드리지 않는다.
- `tests/test_causality_v2.py`: v2 특징에 대한 미래 변조·당일 차단·목표일 비입력 검증과
  v1 특징 동일성 검증. 기존 5개와 합쳐 10개 테스트를 모두 통과한다.

평가 방식은 번들 README 기준 **회귀 단일 과제 100%**이며, 순위에는 로그 RMSE·MAE와
원 단위 R²·NSE가 함께 반영된다. 10,000 cells/mL 초과 F1은 순위 미반영 참고지표다.

## TabPFN 출처

[공식 구현](https://github.com/PriorLabs/TabPFN), [사용한 v2 회귀 가중치](https://huggingface.co/Prior-Labs/TabPFN-v2-reg), Hollmann et al., *Accurate predictions on small data with a tabular foundation model*, Nature 637, 319–326 (2025), DOI [10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6). 가중치 라이선스 원문은 다운로드된 `LICENSE.txt`를 참조한다.

## 버전 6: 연구 전략 실제 검증

[연구 전략](research/STRATEGY_RESEARCH_2026-10-05.md)의 로컬 데이터 적용 결과는
[V6 실험 보고서](results/EXPERIMENT_REPORT_V6.md)에 있다. 새 피처 30개, 사건/LDS 가중치,
관측 이력 마스킹 및 원자료 이력 혼합 증강, 일별 변화율, OOF 급증·급락 보정을 비교했다.
2018~2024년 126개 모델×연도 비교를 완료했으며 전체 17개 검사가 통과했다.

최저 로그 RMSE 후보는 사건·종 변화·수온 이력을 추가한 TabPFN + 기존 LightGBM의 50/50 혼합이다.
과거 연도 평균 RMSE 0.79733 → 0.79579(차이 −0.00154, 표준오차 0.00491, 3/7년 개선),
2025년 참고 검증 RMSE 0.63350 → 0.61969, NSE 0.00599 → 0.01643이다.
과거 CV 개선이 일관되지 않아 **최종 모델 교체는 미채택**이다.

코드: `algae/features_v6.py`, `algae/augment_v6.py`, `algae/exp6.py`.
산출물: `artifacts/features_v6/`, `artifacts/cv6/`. 후보 저장 모델은
`artifacts/cv6/2025_tabpfn_environment/model.joblib`에 있으며 재로딩 예측을 확인했다.
2025년은 이전 버전에서 이미 관찰한 참고 검증이고, 2026 test 정답은 이번 평가에 사용하지 않았다.
원본·기존 선택·제출 초안은 보존했다. 원 R² 정의와 공식 지표 결합식은 여전히 확인 대상이다.
원자료 이력 혼합은 공개 SMOGN과 동일한 알고리즘이 아니다. 외부 영양염·센서·과거 예보 확보는 후속 범위다.
