"""TabPFN with more than one estimator.

Version 1 ran the public TabPFN-v2 regressor with `n_estimators=1`, which is a
handicapped configuration: the model is built around averaging several
preprocessing draws. This module changes that one setting and nothing else.
Every other argument, the objective, the folds and the metrics are the frozen
version 1 code, so a difference can only come from the estimator count.

Results go to artifacts/cv3 and never touch version 1 or version 2 records.
"""
from __future__ import annotations

import argparse
import json
import time

# algae.experiment must be imported before numpy and pandas here. It sets the
# thread and cache environment variables and imports joblib first, exactly as
# the version 1 entry point does. A different import order changes BLAS
# threading and the fitted TabPFN predictions drift in the last decimals.
from algae.experiment import ROOT, design, folds, metrics  # noqa: I001

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def make_tabpfn(n_estimators):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(4)
    return TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, device='cpu', n_estimators=n_estimators, random_state=42,
        model_path=str(ROOT/'.cache/tabpfn/tabpfn-v2-regressor-v2_default.ckpt'),
        ignore_pretraining_limits=True, categorical_features_indices=[0],
        fit_mode='fit_preprocessors', n_preprocessing_jobs=1)


def fit_predict(objective, train, test, names, n_estimators):
    """Mirror of algae.experiment.fit_predict with the estimator count exposed."""
    train = train.sort_values(['base_date', 'station_id'])
    y = np.log10(train.target.to_numpy(float)+1)
    if objective == 'delta':
        y -= train.cells_log_lag0.to_numpy(float)
    estimator = make_tabpfn(n_estimators)
    estimator.fit(design(train, names, 'tabpfn'), y)
    z = np.asarray(estimator.predict(design(test, names, 'tabpfn')), dtype=float)
    if objective == 'delta':
        z += test.cells_log_lag0.to_numpy(float)
    z = np.maximum(z, 0)
    assert np.isfinite(z).all()
    return estimator, z


def run_job(args):
    tag = f'tabpfn{args.estimators}_{args.objective}_{args.features}_{args.season}_{args.since}'
    output = ROOT/'artifacts'/args.stage/f'{args.year}_{tag}'
    output.mkdir(parents=True, exist_ok=True)
    if (output/'metrics.json').exists() and not args.force:
        print('CACHED', output.name, flush=True)
        return
    config = vars(args).copy()
    config['tag'] = tag
    config['tabpfn_n_estimators'] = args.estimators
    config['selection_note'] = ('Version 3 historical CV only; 2025 remains reserved. '
                                'Only n_estimators differs from the version 1 TabPFN run.')
    (output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    from algae.exp2 import get_data
    data, names = get_data(args.features)
    train, test = folds(data, args.year, args.season, args.since)
    print('START', output.name, 'train', len(train), 'eval', len(test),
          'features', len(names), 'estimators', args.estimators, flush=True)
    started = time.time()
    try:
        _, z = fit_predict(args.objective, train, test, names, args.estimators)
        predictions = test[['station_id', 'base_date', 'target_date', 'target', 'cells_log_lag0']].copy()
        predictions['pred_log10'] = z
        predictions['pred_cells'] = np.power(10., z)-1
        predictions.to_csv(output/'predictions.csv', index=False)
        warm = test.base_date.dt.month.between(5, 10).to_numpy()
        report = {'tag': tag, 'year': args.year, 'n_train': len(train), 'n_features': len(names),
                  'tabpfn_n_estimators': args.estimators, 'elapsed_seconds': time.time()-started,
                  'all': metrics(test.target, z),
                  'season': metrics(test.target.to_numpy()[warm], z[warm]),
                  'station_season': {site: metrics(g.target, g.pred_log10) for site, g in
                                     predictions[predictions.base_date.dt.month.between(5, 10)].groupby('station_id')}}
        (output/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('DONE', output.name, json.dumps(report['season']),
              round(report['elapsed_seconds'], 1), 'sec', flush=True)
    except Exception as exc:
        (output/'error.txt').write_text(repr(exc), encoding='utf-8')
        raise


def summarize(stage='cv3'):
    rows = []
    for file in (ROOT/'artifacts'/stage).glob('*/metrics.json'):
        m = json.loads(file.read_text())
        rows.append({'tag': m['tag'], 'year': m['year'], **m['season'],
                     'all_rmse_log10': m['all']['rmse_log10'], 'seconds': m['elapsed_seconds']})
    frame = pd.DataFrame(rows)
    if len(frame):
        frame.sort_values(['tag', 'year']).to_csv(ROOT/'artifacts'/f'{stage}_scores.csv', index=False)
        print(frame.groupby('tag').agg(folds=('year', 'nunique'), mean_rmse=('rmse_log10', 'mean'),
                                       mean_mae=('mae_log10', 'mean'), mean_r2=('r2_nse', 'mean'),
                                       seconds=('seconds', 'sum')).sort_values('mean_rmse').to_string())
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run', 'summarize'])
    parser.add_argument('--estimators', type=int, default=4)
    parser.add_argument('--objective', default='delta')
    parser.add_argument('--year', type=int, default=2024)
    parser.add_argument('--features', default='v1')
    parser.add_argument('--season', default='all')
    parser.add_argument('--since', type=int, default=2012)
    parser.add_argument('--stage', default='cv3')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.command == 'run':
        run_job(args)
    else:
        summarize(args.stage)


if __name__ == '__main__':
    main()
