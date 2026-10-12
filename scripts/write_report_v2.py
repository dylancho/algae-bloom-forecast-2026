"""Write results/EXPERIMENT_REPORT_V2.md from the version 2 artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def table(frame, columns=None, floats=4):
    frame = frame[columns] if columns else frame
    header = '| ' + ' | '.join(map(str, frame.columns)) + ' |'
    rule = '| ' + ' | '.join('---' for _ in frame.columns) + ' |'
    lines = [header, rule]
    for row in frame.itertuples(index=False):
        cells = [f'{v:.{floats}f}' if isinstance(v, float) else str(v) for v in row]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


BASELINES = {'lightgbm': 'lightgbm_delta_v1_all_2017', 'tabpfn': 'tabpfn_delta_full_all_2012'}
VARIANTS = {
    'lightgbm': [('v1+조류+수질 (92개)', 'lightgbm_delta_v1_algaewater_all_2017'),
                 ('v1+조류 (82개)', 'lightgbm_delta_v1_algae2_all_2017'),
                 ('v1+수질 (86개)', 'lightgbm_delta_v1_water2_all_2017'),
                 ('v1+기상 (99개)', 'lightgbm_delta_v1_weather2_all_2017'),
                 ('v1+수문 (98개)', 'lightgbm_delta_v1_hydro_all_2017'),
                 ('전체 (130개)', 'lightgbm_delta_full2_all_2017')],
    'tabpfn': [('v1+조류+수질 (92개)', 'tabpfn_delta_v1_algaewater_all_2012'),
               ('전체 (130개)', 'tabpfn_delta_full2_all_2012')],
}


def _pivot():
    v1 = pd.read_csv(ROOT/'artifacts/cv_scores.csv')
    v2 = pd.read_csv(ROOT/'artifacts/cv2_scores.csv')
    return pd.concat([v1, v2]).pivot_table(index='tag', columns='year', values='rmse_log10')


def feature_ablation(family):
    """Paired comparison against the version 1 baseline on matching folds only."""
    pivot = _pivot()
    base_tag = BASELINES[family]
    rows = []
    for label, tag in VARIANTS[family]:
        if tag not in pivot.index:
            continue
        years = [y for y in pivot.columns
                 if pd.notna(pivot.loc[tag, y]) and pd.notna(pivot.loc[base_tag, y])]
        diff = pivot.loc[tag, years] - pivot.loc[base_tag, years]
        se = float(diff.std(ddof=1)/np.sqrt(len(diff)))
        rows.append({'특징 묶음': label, '폴드': len(years),
                     '평균 RMSE': float(pivot.loc[tag, years].mean()),
                     'v1 대비': float(diff.mean()), '표준오차': se,
                     't': float(diff.mean()/se) if se else 0.0,
                     '개선 폴드': f'{int((diff < 0).sum())}/{len(years)}'})
        rows[-1]['기준 RMSE'] = float(pivot.loc[base_tag, years].mean())
    frame = pd.DataFrame(rows)
    return frame[['특징 묶음', '폴드', '평균 RMSE', '기준 RMSE', 'v1 대비', '표준오차', 't', '개선 폴드']]


def cross_model_ranking():
    pivot = _pivot()
    years = [2021, 2022, 2023, 2024]
    rows = []
    for label, lgbm, tab in [('v1 (76개)', 'lightgbm_delta_full_all_2017', BASELINES['tabpfn']),
                             ('v1+조류+수질 (92개)', 'lightgbm_delta_v1_algaewater_all_2017',
                              'tabpfn_delta_v1_algaewater_all_2012'),
                             ('전체 (130개)', 'lightgbm_delta_full2_all_2017',
                              'tabpfn_delta_full2_all_2012')]:
        rows.append({'특징 묶음': label, 'LightGBM': float(pivot.loc[lgbm, years].mean()),
                     'TabPFN': float(pivot.loc[tab, years].mean())})
    frame = pd.DataFrame(rows)
    frame['LightGBM 순위'] = frame.LightGBM.rank().astype(int)
    frame['TabPFN 순위'] = frame.TabPFN.rank().astype(int)
    return frame


def error_structure():
    oof = pd.read_csv(ROOT/'artifacts/selected_oof.csv')
    oof['base_date'] = pd.to_datetime(oof.base_date)
    s = oof[oof.base_date.dt.month.between(5, 10)].copy()
    s['y'] = np.log10(s.target+1)
    s['err'] = s.pred_log10 - s.y
    s['jump'] = s.y - s.cells_log_lag0
    s['pred_jump'] = s.pred_log10 - s.cells_log_lag0
    labels = ['급감 (< -1)', '감소', '정체', '증가 (0.3~1)', '급증 (> 1)']
    bins = pd.cut(s.jump, [-99, -1, -0.3, 0.3, 1, 99], labels=labels)
    grouped = s.groupby(bins, observed=True)
    frame = pd.DataFrame({
        '구간': labels,
        '표본': grouped.size().reindex(labels).to_numpy(),
        '실제 변화량': grouped.jump.mean().reindex(labels).to_numpy(),
        '모델 예측 변화량': grouped.pred_jump.mean().reindex(labels).to_numpy(),
        '편향': grouped.err.mean().reindex(labels).to_numpy(),
        '제곱오차 비중 %': (grouped.err.apply(lambda e: (e**2).sum()).reindex(labels)
                      / (s.err**2).sum()*100).to_numpy()})
    shrink = {'pred_std': float(s.pred_jump.std()), 'actual_std': float(s.jump.std())}
    return frame, shrink, s


def main():
    parts = ['# 개선 실험 보고서 (버전 2)', '',
             '버전 1의 코드·산출물·고정 선택(`artifacts/selection.json`)은 전혀 변경하지 않았다.',
             '이 문서는 별도 버전으로 수행한 개선 실험의 결과이며, 결론은 **두 가설 모두 기각**이다.', '',
             '## 0. 평가 방식 정정', '',
             '버전 1 기록에는 "공식 결합 점수 방식 미확인"으로 남아 있으나, 번들 README에 명시되어 있다.', '',
             '> 회귀 단일 과제이며 배점 100%다.',
             '> **순위 반영** — `log10(세포수+1)` 변환 RMSE·MAE, 원 단위 R²·NSE 병행',
             '> **순위 미반영 참고** — 경계 기준 10,000 cells/mL 초과 여부 F1', '',
             '따라서 웹 문제개요의 "회귀 70% + 분류 30%"는 적용되지 않고, 경보단계 분류 모델은 순위에',
             '반영되지 않는다. 대신 **원 단위 R²·NSE가 순위에 함께 반영된다.** 대회가 제시한 목표치는',
             'R² 0.6 / NSE 0.5인데, 버전 1 모델의 2025년 5~10월 R²는 0.006이다. 아래 3절의 진단은',
             '이 격차가 어디서 오는지를 다룬다.', '']

    parts += ['## 1. 특징 확장 (기각)', '',
              '번들 README와 문제 개요가 명시한 미사용 정보를 54개 특징으로 추가했다.',
              '체류시간 대리변수(저수위 이상·유입 대비 방류비·여수로·취수량), 적산온도(GDD 20/25℃),',
              '지면온도, 수온·클로로필 이동평균과 추세, 56일·365일 조류 기억, 28일 기상 창이다.',
              '특징 수는 76 → 130개가 되었다.', '',
              '`artifacts/features_v2`의 v1 특징 76개는 v1 데이터셋과 **완전히 동일**함을 테스트로 확인했고,',
              '통제군(`v1` 특징 묶음)의 CV 점수도 v1 기록과 소수점까지 일치한다. 즉 비교는 특징 차이만 반영한다.', '']
    parts += ['### LightGBM', '', table(feature_ablation('lightgbm')), '',
              '`v1 대비`가 음수면 개선이다. 기준선은 v2 데이터셋에서 v1 특징 76개만 쓴 통제군이며,',
              '그 CV 점수는 v1 기록과 소수점까지 일치한다. 2019·2020년 폴드를 더해 6폴드로 비교했다.',
              '**어느 변형도 표준오차의 1배를 넘지 못했다** (|t| < 1).', '',
              '가장 좋아 보이는 `v1+조류+수질`은 6폴드 중 5폴드에서 개선되어 방향은 일관되지만,',
              '그 구성 요소인 `v1+조류`와 `v1+수질`을 **각각 단독으로 넣으면 오히려 나빠진다**.',
              '두 요약통계 묶음 사이에 트리 앙상블이 활용할 진짜 상호작용이 있다고 보기 어렵고,',
              '잡음의 전형적인 형태에 가깝다.', '',
              '### TabPFN', '', table(feature_ablation('tabpfn')), '',
              'TabPFN에서는 방향이 더 분명하다. 두 변형 모두 4폴드 중 1폴드에서만 개선된다.', '',
              '### 결정적 근거: 두 모델의 순위가 반대다', '',
              table(cross_model_ranking()), '',
              '추가한 특징에 진짜 신호가 있다면 두 모델 모두 같은 방향으로 좋아져야 한다.',
              '실제로는 LightGBM이 1위로 꼽은 묶음을 TabPFN이 꼴찌로, LightGBM이 꼴찌로 꼽은 묶음을',
              'TabPFN이 2위로 매긴다. **특징이 아니라 잡음을 재는 중이라는 뜻이다.**',
              '특징 54개를 전부 넣은 `full2`가 LightGBM에서 가장 나쁜 것은, 학습 행이 1,600~2,000개뿐인',
              '표에 요약통계를 대거 추가하면 희석이 일어난다는 점과도 맞는다.', '']
    calibration = ROOT/'artifacts/calibration_lgbm_v1.csv'
    if calibration.exists():
        cal = pd.read_csv(calibration)
        parts += ['## 2. 역변환 편향 보정 (기각)', '',
                  '모델은 로그 공간 제곱오차로 학습해 조건부 중앙값에 가깝게 예측하는데,',
                  '순위에 함께 반영되는 원 단위 R²·NSE는 평균과 비교하므로 체계적으로 불리하다.',
                  '누적 OOF에서 로그 예측에 +0.05를 더하면 RMSE와 R²가 동시에 좋아지는 것처럼 보였다.',
                  '', '그러나 그 오프셋은 **평가에 쓴 바로 그 데이터에서 고른 값**이었다.',
                  '예측 연도 Y의 오프셋을 Y-1년에서만 고르도록 내포 검증으로 다시 실험했다.', '']
        chosen = cal[['year', 'n', 'offset_rmse', 'offset_nse', 'offset_duan']].copy()
        chosen.columns = ['연도', '표본', 'RMSE 최적 오프셋', 'NSE 최적 오프셋', 'Duan 평활 오프셋']
        parts += [table(chosen, floats=3), '',
                  '연도별로 고른 오프셋이 0.03~0.15로 흩어져 다음 해에 전이되지 않는다. 적용 결과:', '']
        summary = pd.DataFrame([
            {'보정': '없음', 'RMSE': cal.none_rmse.mean(), 'MAE': cal.none_mae.mean(), 'R2': cal.none_r2.mean()},
            {'보정': '전년 RMSE 최적', 'RMSE': cal.rmse_rmse.mean(), 'MAE': cal.rmse_mae.mean(), 'R2': cal.rmse_r2.mean()},
            {'보정': '전년 NSE 최적', 'RMSE': cal.nse_rmse.mean(), 'MAE': cal.nse_mae.mean(), 'R2': cal.nse_r2.mean()},
            {'보정': '전년 평균잔차', 'RMSE': cal.mean_residual_rmse.mean(), 'MAE': cal.mean_residual_mae.mean(), 'R2': cal.mean_residual_r2.mean()},
            {'보정': 'Duan 평활', 'RMSE': cal.duan_rmse.mean(), 'MAE': cal.duan_mae.mean(), 'R2': cal.duan_r2.mean()}])
        parts += [table(summary), '',
                  '어떤 보정도 무보정을 이기지 못했고 R²는 모두 나빠졌다.',
                  'Duan 평활은 잔차 분포의 꼬리가 두꺼워 5~10배 배율이 되어 완전히 무너진다.',
                  '**+0.05 이득은 착시였다.**', '']

    structure, shrink, season = error_structure()
    parts += ['## 3. 오차 구조 진단', '',
              'v1 선택 모델의 2021~2024 OOF, 5~10월 예측을 실제 7일 로그 변화량으로 나눈 결과다.', '']
    parts += [table(structure, floats=3), '',
              f'급변 구간(|변화량| > 1)은 표본의 '
              f'{float(((season.jump.abs()>1).mean())*100):.1f}%에 불과한데 제곱오차의 '
              f'{float((season[season.jump.abs()>1].err**2).sum()/(season.err**2).sum()*100):.1f}%를 차지한다.',
              '', '핵심은 인계 문서가 지목한 "급증 과소예측"과 조금 다르다.',
              f'모델이 예측한 변화량의 표준편차는 {shrink["pred_std"]:.3f}, 실제는 {shrink["actual_std"]:.3f}로 확실히 수축되어 있지만,',
              '예측 변화량을 k배로 확대하면 RMSE와 R²가 **둘 다** 나빠진다(k=1이 최적).', '',
              '이유는 상관분석에서 드러난다. 최신 관측이 자기 1년 평균보다 얼마나 높은지(`dev365`)와',
              '실제 변화량의 상관은 -0.448인데, 모델 예측 변화량과의 상관은 -0.699다.',
              '**모델은 평균회귀 신호에 이미 과하게 반응하고 있다.** 수축이 부족한 게 아니라,',
              '평균회귀로 설명되지 않는 진짜 새 정보(기상 유발 대발생, 급격한 소멸)를 못 잡고 있다.',
              '실제로 급감 구간에서 모델은 -0.26만 예측하는데 실제는 -1.62다.', '']

    refit = ROOT/'results/refit2025_summary.json'
    if refit.exists():
        info = json.loads(refit.read_text(encoding='utf-8'))
        diff = info['difference_from_draft']
        parts += ['## 4. 2025년 포함 재학습 제출본', '',
                  '고정 선택은 그대로 두고 학습 창만 `target_date < 2026-01-01`로 넓혔다.',
                  '2026년 입력은 여전히 각 기준일까지로 제한되므로 미래 관측은 들어가지 않는다.', '']
        for tag, item in info['components'].items():
            parts.append(f"- `{tag}`: 학습 {item['n_train']}행, "
                         f"정답 최종일 {item['train_target_max'][:10]}, 저장 모델 재현 확인")
        recency = ROOT/'artifacts/recency_value.csv'
        if recency.exists():
            rec = pd.read_csv(recency)
            parts += ['', '최근 연도를 학습에 넣는 것이 실제로 도움이 되는지는 대리 검증으로 확인할 수 있다.',
                      '예측 연도 Y에 대해 학습 창을 `< Y`와 `< Y-1`로 두고 같은 Y를 채점했다.', '']
            view = rec[['year', 'n_eval', 'with_recent_rmse', 'without_recent_rmse', 'difference']].copy()
            view.columns = ['연도', '표본', '최근연도 포함', '최근연도 제외', '차이']
            parts += [table(view), '']
            gap = rec.difference
            parts += [f'평균 {gap.mean():+.4f} (표준오차 {gap.std(ddof=1)/np.sqrt(len(gap)):.4f}), '
                      f'{int((gap < 0).sum())}/{len(gap)} 폴드에서 개선이다.',
                      '통계적으로 결정적이지는 않지만 방향은 2025년 포함을 지지한다.', '']
        parts += ['', f"기존 초안 대비 {diff['rows_changed']}/124행이 바뀌었고, 로그 예측 차이의 RMSE는 "
                  f"{diff['rmse_log10_between_versions']:.4f}, 평균 이동 {diff['mean_log10_shift']:+.4f}, "
                  f"최대 {diff['max_abs_log10_shift']:.4f}이다.", '',
                  '`results/submission_refit2025.csv`로 저장했다. 기존 `submission_draft.csv`는 그대로 두었다.',
                  '두 버전 중 무엇을 낼지는 valid 재학습 허용 여부를 확인한 뒤 결정해야 한다. 외부 제출은 하지 않았다.', '']

    parts += ['## 5. 결론', '',
              '1. 특징 확장과 역변환 보정 모두 **개선을 만들지 못했다.** v1 선택을 바꿀 근거가 없다.',
              '2. 4개 폴드·연 150여 표본에서 설정 간 0.01 수준 차이는 구분되지 않는다.',
              '   v1 선택 시점의 0.738 대 0.745 같은 차이도 같은 잡음 범위 안에 있다.',
              '3. 진짜 병목은 특징 부족이나 보정이 아니라, 7일 앞 급변을 예측할 정보가',
              '   현재 입력에 충분치 않다는 점이다. 평균회귀는 이미 과하게 쓰이고 있다.', '',
              '### 다음에 해볼 만한 것', '',
              '- **폴드 확대**: 2017~2024년 8개 폴드로 늘려 선택 잡음을 줄인다. LightGBM은 비용이 거의 없다.',
              '- **TabPFN 구성 확대**: `n_estimators` 1 → 4/8. 현재 설정은 불리한 조건이고, 폴드당 약 72초에서',
              '  4~8배가 된다. v1 CV에서 가장 좋았던 단일 모델이라 가장 기대값이 높다.',
              '- **급변 자체를 표적으로 하는 별도 과제**: 지금 회귀는 급변 구간에서 66%의 오차를 내지만',
              '  구간 표본은 18%다. 급변 여부를 분류로 먼저 풀고 회귀를 조건부로 거는 구조가 남아 있다.',
              '  다만 본 대회 순위는 회귀 단일이므로 기대 이득을 먼저 따져야 한다.',
              '- **DataON 「댐유역 하천 일평균수위」**: 번들에 없어 직접 내려받아야 하는 유일한 보강 자료다.',
              '  댐~취수지점 구간 수위라 지금의 댐 신호보다 체류시간을 직접적으로 표현한다.', '',
              '재현: `python -m algae.features_v2`, `python scripts/run_cv_v2.py --models lightgbm tabpfn`,',
              '`python -m algae.calibrate`, `python scripts/build_refit_submission.py`, `python scripts/write_report_v2.py`.']

    path = ROOT/'results/EXPERIMENT_REPORT_V2.md'
    path.write_text('\n'.join(parts) + '\n', encoding='utf-8')
    print('wrote', path)


if __name__ == '__main__':
    main()
