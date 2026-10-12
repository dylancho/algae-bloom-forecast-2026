"""Generate the v6 research report from completed, reproducible artifacts."""
from __future__ import annotations

import contextlib
import io
import json

import numpy as np
import pandas as pd

from algae.exp6 import (OUT, YEARS, baseline, digest, initialize, path_for,
                        score, summary, verify_baseline)
from algae.experiment import ROOT
from algae.features import load_bundle
from algae.features_v6 import GROUPS, OUT as FEATURES, prepare


def table(frame, columns):
    lines = ['| 설정 | '+' | '.join(columns)+' |', '|---|'+'---:|'*len(columns)]
    for name, row in frame.iterrows():
        values = [f'{row[c]:.5f}' if pd.notna(row[c]) else '—' for c in columns]
        lines.append('| '+str(name)+' | '+' | '.join(values)+' |')
    return '\n'.join(lines)


def candidate(year, frozen):
    keys = ['station_id', 'base_date', 'target_date']
    p = pd.read_csv(path_for(frozen['model'], frozen['variant'], year)/'predictions.csv',
                    parse_dates=['base_date', 'target_date']).set_index(keys).sort_index()
    if frozen['model'] == 'tabpfn':
        tag = 'lightgbm_delta_full_all_2017'
    else:
        tag = 'tabpfn_delta_full_all_2012'
    folder = 'holdout' if year == 2025 else 'cv'
    q = pd.read_csv(ROOT/f'artifacts/{folder}/{year}_{tag}/predictions.csv',
                    parse_dates=['base_date', 'target_date']).set_index(keys).reindex(p.index)
    assert q.pred_cells.notna().all()
    p['pred_log10'] = (np.log10(p.pred_cells+1)+np.log10(q.pred_cells+1))/2
    p['pred_cells'] = np.power(10., p.pred_log10)-1
    return p.reset_index()


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        scores, means = summary()
        verify_baseline()
    frozen = json.loads((OUT/'frozen_candidate.json').read_text(encoding='utf-8'))
    conf_path = path_for(frozen['model'], frozen['variant'], 2025)/'metrics.json'
    confirmation = json.loads(conf_path.read_text(encoding='utf-8'))
    assert confirmation.get('serialization_replay_verified')
    reference = scores[(scores.year == 2025) & scores.name.isin(['v1_selected_5050', frozen['candidate']])].set_index('name')
    assert len(reference) == 2
    delta = reference.loc[frozen['candidate'], 'rmse_log10']-reference.loc['v1_selected_5050', 'rmse_log10']
    delta_nse = reference.loc[frozen['candidate'], 'nse']-reference.loc['v1_selected_5050', 'nse']

    a = pd.read_csv(FEATURES/'thermal_forward_checks.csv', parse_dates=['date'])
    a = a[a.date.dt.year.isin(YEARS)].copy()
    algae, _ = prepare(load_bundle(ROOT/'02_dataset_bundle_algae2026.zip'))
    a['persistence'] = np.nan
    for site, idx in a.groupby('station_id').groups.items():
        old = algae[site].iem_wtrtp.dropna().shift(1)
        a.loc[idx, 'persistence'] = old.reindex(pd.DatetimeIndex(a.loc[idx, 'date'])).to_numpy()
    a['sq'] = (a.actual-a.pred)**2
    a['sq_persistence'] = (a.actual-a.persistence)**2
    thermal = a.groupby('station_id').agg(n=('actual', 'size'), mse=('sq', 'mean'), mse_persistence=('sq_persistence', 'mean'))
    thermal['수온_RMSE_C'] = np.sqrt(thermal.mse)
    thermal['직전수온유지_RMSE_C'] = np.sqrt(thermal.mse_persistence)
    thermal.to_csv(OUT/'thermal_diagnostics.csv')

    diagnostics, annual = [], []
    for year in YEARS:
        old, _ = baseline(year)
        new = candidate(year, frozen)
        for name, p in [('v1', old), ('v6_candidate', new)]:
            p = p[p.base_date.dt.month.between(5, 10)].copy()
            z = np.log10(p.target+1)-p.cells_log_lag0
            p['state'] = np.where(z < -.7, 'decline', np.where(z > .7, 'rise', 'stable'))
            p['log_sq_error'] = (np.log10(p.pred_cells+1)-np.log10(p.target+1))**2
            p['raw_sq_error'] = (p.pred_cells-p.target)**2
            p['name'], p['year'] = name, year
            diagnostics.append(p)
        old_s = score(old[old.base_date.dt.month.between(5, 10)])
        new_s = score(new[new.base_date.dt.month.between(5, 10)])
        annual.append({'year': year, 'v1_rmse': old_s['rmse_log10'], 'candidate_rmse': new_s['rmse_log10'],
                       'delta_rmse': new_s['rmse_log10']-old_s['rmse_log10'],
                       'v1_nse': old_s['nse'], 'candidate_nse': new_s['nse']})
    pd.DataFrame(annual).to_csv(OUT/'candidate_by_year.csv', index=False)
    joined = pd.concat(diagnostics, ignore_index=True)
    grouped = joined.groupby(['name', 'state']).agg(n=('target', 'size'), mse_log=('log_sq_error', 'mean'), raw_sse=('raw_sq_error', 'sum'))
    grouped['rmse_log10'] = np.sqrt(grouped.mse_log)
    grouped.to_csv(OUT/'candidate_by_state.csv')
    raw_mask = pd.read_pickle(FEATURES/'history_masked.pkl')
    raw_mix = pd.read_pickle(FEATURES/'history_mixup.pkl')
    configuration_files = sorted(OUT.glob('*_*/metrics.json'))
    historical_jobs = [p for p in configuration_files if int(p.parent.name.split('_')[0]) in YEARS]
    keep = ['v1_selected_5050', 'blend_tabpfn_environment', 'blend_tabpfn_thermal', 'blend_event_weight',
            'blend_lds_weight', 'blend_thermal', 'blend_state', 'blend_events', 'blend_biology',
            'blend_regime', 'blend_rate', 'blend_masked', 'blend_history_mixup']
    chosen = means.loc[keep]
    accepted = False
    # Historical gains are below their standard error; reference validation
    # does not override that insufficiency, regardless of its direction.
    rationale = '과거 CV 개선폭이 표준오차보다 작고 연도별 방향이 일관되지 않아 채택 근거가 부족하다.'
    summary_json = {'historical_jobs': len(historical_jobs), 'feature_counts': {k: len(v) for k, v in GROUPS.items()},
                    'candidate': frozen, 'confirmation': reference.to_dict(orient='index'),
                    'reference_rmse_difference': delta, 'reference_nse_difference': delta_nse,
                    'adopted': accepted, 'decision_reason': rationale,
                    'masked_real_pairs': len(raw_mask), 'history_mixup_pairs': len(raw_mix),
                    'test_target_scoring': False, 'protected_unchanged': True,
                    'serialization_replay_verified': True,
                    'code_sha256': {f: digest(ROOT/f) for f in ['algae/features_v6.py', 'algae/exp6.py',
                                                            'algae/augment_v6.py', 'tests/test_causality_v6.py']}}
    (OUT/'run_summary.json').write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding='utf-8')

    text = f'''# 개선 실험 보고서 V6 — 도메인 이력·증강·불균형·상태 구분

작성일: 2026-10-05. 연구 제안의 로컬 데이터 적용을 완료했다. **새 후보를 기존 최종 모델로 채택할 근거는 부족하다.** 가장 좋은 과거 CV 후보도 개선폭이 작고 연도별 방향이 섞인다. 이번 결과는 이 구현·설정의 검증이며, 해당 방법군 전체가 효과 없거나 데이터가 성능 상한에 도달했다는 뜻은 아니다.

## 1. 실행 범위와 비교 조건

- LightGBM 16개 설정 × 7개 연도, TabPFN 2개 설정 × 7개 연도: **{len(historical_jobs)}개 외부 fold 실행**. 급증/유지/급락 보정에는 별도의 내부 전진 검증도 사용했다.
- 비교 연도 2018~2024, 평가 월 5~10월, 실제 관측 쌍 총 934개. 아래 표는 각 연도 점수의 동일 가중 평균이다.
- LightGBM은 기존 2017년 이후 학습 설정, TabPFN은 기존 2012년 이후 설정과 같은 시작 모델·하이퍼파라미터를 썼다.
- 기본 앙상블은 로그 공간 50/50이다. 개별 전략 비교에서는 한 구성 요소만 교체한다. 새로운 두 모델을 동시에 바꾸는 비율 탐색은 수행하지 않았다.
- 2025년 새 예측 전에 후보를 `frozen_candidate.json`에 기록했다. 2025년은 이전 버전에서 이미 본 자료이므로 독립 최종 holdout이라고 부르지 않는다.
- 지표는 log10(y+1) RMSE·MAE, 원 단위 NSE/r2_score, Pearson 상관 제곱을 구분했다. 공식 R² 정의·결합식은 계속 미확인이다.

## 2. 구현한 전략

1. **사건 순서 11개:** 유량 급증 최고점 이후 경과일, 감소 폭, 이후 고온·일조 누적, 강풍 이후 경과일, 저풍속 지속, 관측된 고농도 지속. 유량은 기존 WAMIS 사본을 사용하며 강수 공란을 0으로 추정하지 않았다.
2. **생물학적 변화 6개:** Microcystis 비중 변화, 속 구성 변화량·우점 전환, 세포수/Chl-a 로그 비율 변화, pH·DO 변화.
3. **수온 이력 8개:** 이전 연도 수온으로 학습한 동적 Ridge 회귀와 Kalman 잔차 갱신으로 일별 수온을 추정하고, 누적 고온·급랭·성장×일조·상태 불확실성을 산출했다. 원 Air2Water 구현은 아니다.
4. **세포수 잠재 상태 5개:** 고정 오차 모수를 가진 불규칙 간격 local-linear Kalman filter의 상태·추세·불확실성·관측 혁신. Bayesian 전체 모형이나 미래 실측 날씨를 사용하지 않았다.
5. **불균형 완화:** 급증/유지/급락 에피소드 길이·희소성을 반영한 가중치, 로그 변화량 분포를 평활한 LDS 참고 가중치. 모두 훈련 fold에서만 계산하고 완만하게 제한했다.
6. **실제 정답을 유지하는 증강:** 이전 실제 관측 하나를 가린 {len(raw_mask)}개 실제 쌍, 가중치 0.25. 기준일 실측과 실제 미래 정답을 유지하며 lag·통계를 원자료에서 다시 계산했다.
7. **조건 제한 원자료 이력 혼합:** 같은 지점·계절·변화 상태·관측 간격의 두 실제 사례를 섞은 {len(raw_mix)}쌍, 가중치 0.25. 과거 조류·수질·기상·댐·유량 블록을 상대 날짜로 맞춰 섞고 파생 피처를 다시 계산했다. **공개 SMOGN 알고리즘과 동일하지 않다.** 두 부모의 정답이 모두 훈련 경계 이전인 경우만 사용했다.
8. **타깃/상태 변환:** 실제 관측 간격으로 나눈 일별 로그 변화율을 7일로 예측하는 후보, 내부 시간 OOF 급증/급락 확률을 이용한 제한적 잔차 보정.

## 3. 과거 CV 결과

`blend_*`는 기존 앙상블의 한 구성 요소만 해당 후보로 교체한 결과다. `blend_tabpfn_*`는 TabPFN 교체, 나머지는 LightGBM 교체다. 음수 `rmse_vs_v1`는 개선이다.

{table(chosen, ['rmse_log10', 'mae_log10', 'nse', 'pearson_r2', 'rmse_vs_v1', 'rmse_diff_se'])}

최저 로그 RMSE 후보는 **{frozen['candidate']}**다. 기존 모델 대비 평균 차이 {means.loc[frozen['candidate'], 'rmse_vs_v1']:+.5f}, 표준오차 {means.loc[frozen['candidate'], 'rmse_diff_se']:.5f}, 개선 연도 {int(means.loc[frozen['candidate'], 'rmse_improved_years'])}/7이다. 원 단위 NSE 차이도 {means.loc[frozen['candidate'], 'nse_vs_v1']:+.5f} ± {means.loc[frozen['candidate'], 'nse_diff_se']:.5f}로 작다.

사건 가중치는 로그 RMSE를 조금 줄였지만 MAE·NSE가 나빠졌다. 이력 마스킹과 원자료 혼합 증강은 선택 앙상블에서 개선하지 못했다. 모든 피처를 한꺼번에 넣는 것보다 작은 묶음 비교가 필요하다는 기존 판단은 유지되지만, 이번 결과로 최적 묶음을 확정하지 않는다. 여러 후보 중 최고를 고른 CV에는 선택 편향이 있다.

## 4. 수온 복원 자체의 품질

관측이 들어오기 **전**의 수온 예측을 실제 수온과 비교했다. 해당 연도 수온을 회귀 계수 학습에 넣지 않으며, 과거 실측의 필터 갱신은 사용한다. 전체 계절 2018~2024의 지점별 진단이다.

{table(thermal, ['n', '수온_RMSE_C', '직전수온유지_RMSE_C'])}

수온 복원이 어느 정도 정확하더라도 녹조 7일 예측 개선과는 별개다. 일별 추정값의 참값이 없는 날짜를 검증했다고 주장하지 않는다. 현재 실험은 주간 실측 직전 예측을 검증했고, 녹조 모델에는 과거 일별 추정 노출량을 사용했다.

## 5. 고정 후보의 2025년 참고 검증

5~10월 실제 관측 151쌍. 후보를 고른 뒤 아래 결과를 산출했다.

{table(reference, ['rmse_log10', 'mae_log10', 'nse', 'pearson_r2'])}

기존 모델 대비 로그 RMSE 차이 {delta:+.5f}, NSE 차이 {delta_nse:+.5f}. 이 결과를 근거로 후보를 바꾸거나 다른 2025년 후보를 추가 탐색하지 않았다. **판정: 미채택.** {rationale}

## 6. 검증 및 재현

- 전체 17개 causality/정합성 검사 통과. 미래 조류·당일 이후 기상/유량 변조, 미래 정답일 변경, 상태 필터 prefix 불변, 증강 부모 양쪽의 시간 경계, 파생 성장률 일치를 확인했다.
- 7개 연도의 v6 기준 모델 예측은 기존 LightGBM 예측과 일치한다. 피처 구현만 다른 실험의 통제군이 재현된다.
- 저장한 2025년 후보 모델을 다시 읽어 예측 일치를 확인했다.
- 원본 ZIP, 기존 코드·선택·제출 초안/재학습본을 보존했다. 보호 파일 해시 확인은 `verification.json`에 있다.
- 2026 test 정답은 지도학습 표에 없으며, 이번 비교·선택·보고서에 사용하지 않았다.
- 코드/결과/계획 해시 및 상세 수치는 `artifacts/cv6/`에 기록했다. 합성 표본은 독립 사건 수를 늘리지 않는다.

```powershell
& .venv\\Scripts\\python.exe -m algae.features_v6 build
& .venv\\Scripts\\python.exe -m algae.features_v6 mask
& .venv\\Scripts\\python.exe -m algae.augment_v6
& .venv\\Scripts\\python.exe -m unittest discover -s tests -v
& .venv\\Scripts\\python.exe -m algae.exp6 run
& .venv\\Scripts\\python.exe -m algae.exp6 run --model tabpfn --variants thermal environment
& .venv\\Scripts\\python.exe -m algae.exp6 freeze
& .venv\\Scripts\\python.exe -m algae.exp6 run --model {frozen['model']} --variants {frozen['variant']} --years 2025 --save
& .venv\\Scripts\\python.exe -m algae.exp6 verify
& .venv\\Scripts\\python.exe -m scripts.write_report_v6
```

## 7. 남은 범위

이번에 실행한 것은 현재 확보된 자료로 가능한 전략의 제한된 비교다. TN/TP·고빈도 생물 센서·과거 발표 기상예보는 새 자료 확보와 시점/규칙 확인이 필요해 실행에 포함하지 않았다. 저장량 V가 역사 자료에 없으므로 실제 Q/V 체류시간이나 물 교환율도 만들지 않았다. 장기 식별 가능한 Bayesian 과정 모형·GRU-D·GAN 및 확산모델의 대규모 탐색은 수행하지 않았다.

다음 작업의 우선순위는 추가 변환을 더 탐색하는 것보다, 고농도 유지와 급락을 가를 실제 신규 관측(영양염·고빈도 생물량·당시 예보)을 확보할 수 있는지 확인하는 것이다. 이번 결과는 그 정보가 반드시 성능을 올린다는 증거도 아니다.
'''
    (ROOT/'results/EXPERIMENT_REPORT_V6.md').write_text(text, encoding='utf-8')
    print('REPORT', ROOT/'results/EXPERIMENT_REPORT_V6.md', flush=True)
    print(json.dumps({'historical_jobs': len(historical_jobs), 'adopted': accepted,
                      'candidate': frozen['candidate'], 'reference_delta_rmse': delta,
                      'reference_delta_nse': delta_nse}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
