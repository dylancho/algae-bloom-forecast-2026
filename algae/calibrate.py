"""Nested log-space calibration.

The models are fitted with squared error on log10(y+1) and predict close to a
conditional median. The competition ranks on log10 RMSE/MAE and on raw-unit
R2/NSE at the same time, and the raw-unit score compares against a mean, so a
median-like prediction is biased low there.

The offset for forecast year Y is estimated only from data before Y: an inner
fold trains on years before Y-1 and is scored on Y-1. No evaluation year, and no
holdout year, is ever used to pick an offset.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from algae.experiment import ROOT, fit_predict, folds, metrics

GRID = np.round(np.arange(-0.10, 0.51, 0.01), 2)


def _season(frame, values):
    mask = frame.base_date.dt.month.between(5, 10).to_numpy()
    return frame.target.to_numpy(float)[mask], np.asarray(values, float)[mask]


def offsets_from_inner_fold(data, names, model, objective, year, season, since):
    """Fit on years before Y-1, score Y-1, and read the offsets off that year."""
    inner = year - 1
    train, evaluation = folds(data, inner, season, since)
    _, z = fit_predict(model, objective, train, evaluation, names)
    y, z = _season(evaluation, z)
    log_y = np.log10(y + 1)
    scores = {}
    for c in GRID:
        zc = np.maximum(z + c, 0)
        raw = np.power(10., zc) - 1
        scores[float(c)] = {
            'rmse': float(np.sqrt(np.mean((zc - log_y) ** 2))),
            'nse': float(1 - np.sum((y - raw) ** 2) / np.sum((y - y.mean()) ** 2)),
        }
    best_rmse = min(scores, key=lambda c: scores[c]['rmse'])
    best_nse = max(scores, key=lambda c: scores[c]['nse'])
    smearing = float(np.mean(np.power(10., log_y - z)))
    return {'inner_year': inner, 'n_inner': int(len(y)),
            'offset_rmse': best_rmse, 'offset_nse': best_nse,
            'offset_duan': float(np.log10(smearing)), 'duan_smearing': smearing,
            'offset_mean_residual': float(np.mean(log_y - z))}


def evaluate(data, names, model, objective, year, season, since):
    train, evaluation = folds(data, year, season, since)
    _, z = fit_predict(model, objective, train, evaluation, names)
    chosen = offsets_from_inner_fold(data, names, model, objective, year, season, since)
    y, z = _season(evaluation, z)
    row = {'year': year, 'n': int(len(y)), **{k: v for k, v in chosen.items() if k != 'duan_smearing'}}
    for label, c in [('none', 0.0), ('rmse', chosen['offset_rmse']),
                     ('nse', chosen['offset_nse']), ('duan', chosen['offset_duan']),
                     ('mean_residual', chosen['offset_mean_residual'])]:
        m = metrics(y, np.maximum(z + c, 0))
        row[f'{label}_rmse'] = m['rmse_log10']
        row[f'{label}_mae'] = m['mae_log10']
        row[f'{label}_r2'] = m['r2_nse']
        row[f'{label}_offset'] = float(c)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='lightgbm')
    parser.add_argument('--objective', default='delta')
    parser.add_argument('--features', default='v1')
    parser.add_argument('--season', default='all')
    parser.add_argument('--since', type=int, default=2017)
    parser.add_argument('--years', nargs='+', type=int, default=[2021, 2022, 2023, 2024])
    parser.add_argument('--output', default='artifacts/calibration_cv.csv')
    args = parser.parse_args()
    from algae.exp2 import get_data
    data, names = get_data(args.features)
    rows = [evaluate(data, names, args.model, args.objective, y, args.season, args.since)
            for y in args.years]
    frame = pd.DataFrame(rows)
    path = ROOT/args.output
    frame.to_csv(path, index=False)
    columns = [c for c in frame.columns if c.endswith(('_rmse', '_mae', '_r2')) and not c.startswith('offset')]
    print(frame[['year', 'n', 'offset_rmse', 'offset_nse', 'offset_duan']].to_string(index=False))
    print()
    print(frame[columns].mean().round(4).to_string())
    print()
    print('saved', path)


if __name__ == '__main__':
    main()
