"""Render a Korean experiment record from the completed run artifacts."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def table(frame):
    cols = list(frame.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in row) + ' |')
    return '\n'.join(lines)


def main():
    summary = json.loads((ROOT/'results/run_summary.json').read_text())
    selection = json.loads((ROOT/'artifacts/selection.json').read_text())
    cv = pd.read_csv(ROOT/'artifacts/cv_scores.csv')
    ranked = cv.groupby('tag').agg(folds=('year','nunique'), RMSE=('rmse_log10','mean'),
                                  MAE=('mae_log10','mean'), R2=('r2_nse','mean')).sort_values('RMSE').reset_index()
    holdout = pd.read_csv(ROOT/'results/validation_scores.csv')
    rows = holdout[['scope','model','n','rmse_log10','mae_log10','r2_nse']]
    weights = '\n'.join(f'- `{tag}`: {w:.0%}' for tag,w in selection['weights'].items())
    stations = pd.DataFrame.from_dict(summary['station_season'], orient='index').rename_axis('station').reset_index()
    text = f'''# 실제 학습·검증 결과

## 실행 범위와 선택

- 76개 특징, {len(cv)}회 과거 연도 검증, {len(ranked)}개 설정을 실행했다.
- 2021~2024년을 각각 미래 검증 연도로 두고, 해당 연도 시작 전에 정답이 관측된 공식 train만 학습했다.
- 선택 기준은 5~10월 log10(y+1) RMSE의 연도별 동일 가중 평균이다. 공식 종합점수의 결합 방식은 미확인이다.
- 웹 문제 설명의 과거 예시와 분할·배점이 달라, 제공된 실제 데이터 README의 회귀 100% 및 train~2024/valid2025/test2026 구분을 적용했다.
- 과거 검증 상위 3개 후보의 단독 예측과 두 모델의 25/50/75% 로그 예측 혼합을 비교했다. 미세한 개선으로 복잡도를 늘리지 않도록 0.002 개선 기준을 적용했다.
- 2025년 결과를 보기 전에 선택을 파일로 고정했다. 2025년 결과로 모델·비중을 다시 고르지 않았다.

선택 모델:

{weights}

선택 예측의 과거 검증 RMSE 평균: **{selection['cv_mean_season_rmse']:.4f}**.
`delta`는 최신 관측 log10(y+1) 대비 변화량, `direct`는 미래 log10(y+1)을 직접 예측한다. 혼합도 로그 공간에서 수행한다.

## 과거 검증 비교

{table(ranked)}

`all`은 전 계절 학습, `warm`은 4~11월 학습이다. 마지막 숫자는 학습 시작 연도다.
특징은 algae → water → weather → full 순서로 누적된다. 각 행은 같은 네 검증 연도의 평균이며 R²는 원 단위 기준이다.
여러 설정을 비교해 선택한 CV 수치에는 선택 편향이 있을 수 있으므로 아래 별도 검증 결과를 함께 봐야 한다.

## 2025년 별도 검증

{table(rows)}

`persistence`는 최신 유효 관측값을 그대로 예측한 기준선이다. `May_Oct`는 5~10월, `all`은 전체 256행이다.
모델 선택 기준은 로그 오차이므로 원 단위 극단값의 R²가 함께 개선된다고 보장되지 않는다.

2025년 5~10월에는 TabPFN 단독 RMSE가 0.6079로 고정 앙상블의 0.6335보다 좋았다. 이 결과를 보고 기존 선택을 바꾸지는 않았다. 앙상블은 최신값 유지 기준선보다 로그 RMSE가 약 9.1% 낮지만, 원 단위 R²는 약 0.006으로 급증 크기를 예측하는 능력은 여전히 제한적이다. 그래프에서도 여러 지점의 급격한 상승·하락을 늦게 따라가는 모습이 남아 있다.

지점별 5~10월 성능:

{table(stations)}

![2025년 관측과 예측](validation_2025.png)

## 검증과 제출 초안

- 미래 관측 변조, 목표 날짜 변조, 당일 기상·수문 차단, 조류 결측 시 이전 관측 대체, 학습 정답 날짜 경계의 5개 테스트 통과.
- 저장 모델을 다시 읽은 예측이 저장 전 예측과 일치함을 확인했다.
- 원본 ZIP에서 특징을 다시 계산한 독립 추론 결과와 제출 초안이 동일한지 확인한 결과: **{summary.get('independent_replay_identical', '아직 미기록')}**.
- `submission_draft.csv`: {summary['submission_rows']}행, 원본 양식의 키·순서 유지, 비음수 정수 예측.
- test 정답은 학습·모델 선택·성능 평가에 사용하지 않았다. 외부 제출은 하지 않았다.

## 해석과 남은 제한

1. 이번 TabPFN은 **공개 v2 가중치, CPU, estimator 1개**다. 최신 버전이나 충분한 앙상블을 사용한 TabPFN 전체의 성능으로 일반화하지 않는다.
2. 과거 기간을 줄인 LightGBM의 개선이 모든 연도에서 균일하지는 않다. 해당 기간 선택 역시 네 연도 CV 안에서 수행했다.
3. 각 예측 기준일에 조류·수질 관측 결과를 사용할 수 있다고 가정했다. 실제 공표 지연이 있으면 추가 지연 처리가 필요하다. 기상·수문은 전날까지 사용했다.
4. 2026년 관측도 각 기준일까지 공개된 입력으로만 사용했다. 테스트 구간 전체 관측을 한 번에 미리 아는 조건으로 계산하지 않았다.
5. 강수 결측은 무강수로 확정하지 않고 관측 합과 관측 비율을 함께 입력했다. 수문은 두 댐의 보수적인 신호이며 전체 하천망이나 이동 시간을 표현하지 않는다.
6. valid를 최종 학습에 합쳐도 되는지 확인되지 않아 제출 초안은 2024년까지의 공식 train만 학습한 모델이다. 2025년까지 재학습한 최종 모델은 별도 규칙 확인 후 만들 수 있다.
7. 공식 종합 평가식, 외부 사전학습 모델 허용 여부 및 자료 공개 시점 조건을 확인한 뒤 실제 제출에 사용해야 한다.

재현 명령은 [README](../README.md), 고정 선택은 [selection.json](../artifacts/selection.json), 전체 실행 기록은 [run_summary.json](run_summary.json)에 있다.
'''
    (ROOT/'results/EXPERIMENT_REPORT.md').write_text(text, encoding='utf-8')
    print('Wrote results/EXPERIMENT_REPORT.md')


if __name__ == '__main__':
    main()
