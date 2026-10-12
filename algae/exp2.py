"""Version 2 experiment runner. Reuses the frozen version 1 training code.

Only the feature table changes; folds, models, objectives, and metrics are
imported from algae.experiment so the comparison is like for like. Results go to
artifacts/cv2 and artifacts/holdout2 and never overwrite version 1 records.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from algae.experiment import ROOT, design, fit_predict, folds, metrics

FEATURE_SETS = ['v1', 'full2', 'v1_thermal', 'v1_hydro', 'v1_algaewater',
                'v1_algae2', 'v1_water2', 'v1_weather2']


def get_data(feature_set='full2'):
    folder = ROOT/'artifacts/features_v2'
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    groups = manifest['feature_groups']
    v1 = manifest['v1_features']
    if feature_set == 'v1':
        names = v1
    elif feature_set == 'full2':
        names = manifest['features']
    elif feature_set == 'v1_thermal':
        names = v1 + groups['v2_weather'] + groups['v2_water']
    elif feature_set == 'v1_hydro':
        names = v1 + groups['v2_hydro']
    elif feature_set == 'v1_algaewater':
        names = v1 + groups['v2_algae'] + groups['v2_water']
    elif feature_set == 'v1_algae2':
        names = v1 + groups['v2_algae']
    elif feature_set == 'v1_water2':
        names = v1 + groups['v2_water']
    elif feature_set == 'v1_weather2':
        names = v1 + groups['v2_weather']
    else:
        raise ValueError(feature_set)
    data = pd.read_pickle(folder/'dataset.pkl')
    for col in ['base_date', 'target_date']:
        data[col] = pd.to_datetime(data[col])
    return data, names


def run_job(args):
    tag = f'{args.model}_{args.objective}_{args.features}_{args.season}_{args.since}'
    output = ROOT/'artifacts'/args.stage/f'{args.year}_{tag}'
    output.mkdir(parents=True, exist_ok=True)
    if (output/'metrics.json').exists() and not args.force:
        print('CACHED', output.name, flush=True)
        return
    config = vars(args).copy()
    config['tag'] = tag
    config['dataset_version'] = 2
    config['selection_note'] = ('Version 2 historical CV only; 2025 reserved for a separately '
                                'frozen version 2 candidate.')
    (output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    data, names = get_data(args.features)
    train, test = folds(data, args.year, args.season, args.since)
    print('START', output.name, 'train', len(train), 'eval', len(test), 'features', len(names), flush=True)
    started = time.time()
    try:
        estimator, z = fit_predict(args.model, args.objective, train, test, names)
        predictions = test[['station_id', 'base_date', 'target_date', 'target', 'cells_log_lag0']].copy()
        predictions['pred_log10'] = z
        predictions['pred_cells'] = np.power(10., z)-1
        predictions.to_csv(output/'predictions.csv', index=False)
        warm = test.base_date.dt.month.between(5, 10).to_numpy()
        report = {'tag': tag, 'year': args.year, 'n_train': len(train), 'n_features': len(names),
                  'elapsed_seconds': time.time()-started,
                  'all': metrics(test.target, z),
                  'season': metrics(test.target.to_numpy()[warm], z[warm]),
                  'station_season': {site: metrics(g.target, g.pred_log10) for site, g in
                                     predictions[predictions.base_date.dt.month.between(5, 10)].groupby('station_id')}}
        if args.save:
            payload = {'model': estimator, 'model_name': args.model, 'objective': args.objective,
                       'features': names, 'config': config, 'dataset_version': 2,
                       'train_target_max': str(train.target_date.max())}
            joblib.dump(payload, output/'model.joblib')
            restored = joblib.load(output/'model.joblib')
            replay = restored['model'].predict(design(test, names, args.model))
            if args.objective == 'delta':
                replay += test.cells_log_lag0.to_numpy(float)
            np.testing.assert_allclose(np.maximum(replay, 0), z, rtol=1e-7, atol=1e-7)
            report['serialization_replay_verified'] = True
        (output/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('DONE', output.name, json.dumps(report['season']), round(report['elapsed_seconds'], 1), 'sec', flush=True)
    except Exception as exc:
        (output/'error.txt').write_text(repr(exc), encoding='utf-8')
        raise


def summarize(stage='cv2'):
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
    parser.add_argument('--model', choices=['catboost', 'lightgbm', 'tabpfn'], default='lightgbm')
    parser.add_argument('--objective', choices=['direct', 'delta'], default='delta')
    parser.add_argument('--year', type=int, default=2024)
    parser.add_argument('--features', choices=FEATURE_SETS, default='full2')
    parser.add_argument('--season', choices=['all', 'warm'], default='all')
    parser.add_argument('--since', type=int, default=2012)
    parser.add_argument('--stage', default='cv2')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.command == 'run':
        run_job(args)
    else:
        summarize(args.stage)


if __name__ == '__main__':
    main()
