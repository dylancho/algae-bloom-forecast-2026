"""Does the most recent year of training data help the next forecast?

This is the proxy check behind the 2025-included refit variant. For forecast
year Y the model is trained twice, on target dates before Y and before Y-1, and
both are scored on Y. Nothing from year Y or later is ever used for training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from algae.experiment import ROOT, fit_predict, get_data, metrics

YEARS = [2021, 2022, 2023, 2024]
SINCE = 2017


def training_window(data, upper, since=SINCE):
    start = pd.Timestamp(year=upper, month=1, day=1)
    window = data[data.split.eq('train') & data.base_date.lt(start)
                  & data.target_date.lt(start) & data.base_date.dt.year.ge(since)].copy()
    assert window.target.notna().all() and window.target_date.max() < start
    return window


def main():
    data, names = get_data('full')
    rows = []
    for year in YEARS:
        evaluation = data[data.base_date.dt.year.eq(year) & data.split.isin(['train', 'valid'])].copy()
        mask = evaluation.base_date.dt.month.between(5, 10).to_numpy()
        y = evaluation.target.to_numpy(float)[mask]
        row = {'year': year, 'n_eval': int(mask.sum())}
        for upper, label in [(year, 'with_recent'), (year-1, 'without_recent')]:
            train = training_window(data, upper)
            _, z = fit_predict('lightgbm', 'delta', train, evaluation, names)
            m = metrics(y, z[mask])
            row[f'{label}_n_train'] = len(train)
            row[f'{label}_rmse'] = m['rmse_log10']
            row[f'{label}_r2'] = m['r2_nse']
        row['difference'] = row['with_recent_rmse'] - row['without_recent_rmse']
        rows.append(row)
    frame = pd.DataFrame(rows)
    path = ROOT/'artifacts/recency_value.csv'
    frame.to_csv(path, index=False)
    diff = frame.difference
    print(frame.round(4).to_string(index=False))
    print(f'\nmean {diff.mean():+.4f}  se {diff.std(ddof=1)/np.sqrt(len(diff)):.4f}  '
          f'improved {(diff < 0).sum()}/{len(diff)} folds')
    print('saved', path)


if __name__ == '__main__':
    main()
