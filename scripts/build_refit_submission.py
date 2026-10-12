"""Second submission variant: refit the frozen selection with 2025 included.

The frozen selection in artifacts/selection.json is not changed. Only the
training window moves from "official train, target_date < 2025-01-01" to
"official train and valid, target_date < 2026-01-01". Every 2026 input is still
restricted to each base date, so no future observation enters the forecast.

Version 1 artifacts, the frozen selection and results/submission_draft.csv are
left untouched. Outputs go to artifacts/final2026 and results/.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from algae.experiment import ROOT, design, fit_predict, get_data
from algae.features import load_bundle

OUTPUT = ROOT/'artifacts/final2026'
BOUNDARY = pd.Timestamp('2026-01-01')


def refit_window(data, season, since):
    train = data[data.split.isin(['train', 'valid']) & data.base_date.lt(BOUNDARY)
                 & data.target_date.lt(BOUNDARY) & data.base_date.dt.year.ge(since)].copy()
    if season == 'warm':
        train = train[train.base_date.dt.month.between(4, 11)]
    assert train.target.notna().all()
    assert train.target_date.max() < BOUNDARY
    return train


def main():
    selection = json.loads((ROOT/'artifacts/selection.json').read_text())
    selection_hash = hashlib.sha256((ROOT/'artifacts/selection.json').read_bytes()).hexdigest()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data, _ = get_data()
    test = data[data.split.eq('test')].copy()
    assert test.target.isna().all() and len(test) == 124

    z = np.zeros(len(test))
    summary = {'selection_sha256': selection_hash, 'weights': selection['weights'],
               'training_policy': 'Official train plus 2025 valid; target_date < 2026-01-01.',
               'selection_changed': False, 'components': {}}
    for tag, weight in selection['weights'].items():
        cfg = selection['all_configs'][tag]
        names = get_data(cfg['features'])[1]
        train = refit_window(data, cfg['season'], cfg['since'])
        estimator, pred = fit_predict(cfg['model'], cfg['objective'], train, test, names)
        folder = OUTPUT/tag
        folder.mkdir(exist_ok=True)
        joblib.dump({'model': estimator, 'model_name': cfg['model'], 'objective': cfg['objective'],
                     'features': names, 'config': cfg, 'train_target_max': str(train.target_date.max()),
                     'training_policy': summary['training_policy']}, folder/'model.joblib')
        restored = joblib.load(folder/'model.joblib')
        replay = restored['model'].predict(design(test, names, cfg['model']))
        if cfg['objective'] == 'delta':
            replay = replay + test.cells_log_lag0.to_numpy(float)
        np.testing.assert_allclose(np.maximum(replay, 0), pred, rtol=1e-7, atol=1e-7)
        summary['components'][tag] = {'weight': weight, 'n_train': len(train),
                                      'train_target_max': str(train.target_date.max()),
                                      'serialization_replay_verified': True}
        z += weight*pred
    assert np.isfinite(z).all()

    submission = load_bundle(ROOT/'02_dataset_bundle_algae2026.zip')['sample_submission'].copy()
    for col in ['station_id', 'base_date', 'target_date']:
        actual = test[col].tolist() if col == 'station_id' else test[col].dt.strftime('%Y-%m-%d').tolist()
        assert submission[col].tolist() == actual
    submission['pred_cells'] = np.rint(np.maximum(10**z-1, 0)).astype(np.int64)
    assert submission.pred_cells.ge(0).all() and len(submission) == 124
    assert not submission.duplicated(['station_id', 'base_date', 'target_date']).any()
    path = ROOT/'results/submission_refit2025.csv'
    submission.to_csv(path, index=False, encoding='utf-8')
    reread = pd.read_csv(path)
    assert list(reread) == ['station_id', 'base_date', 'target_date', 'pred_cells']
    assert pd.api.types.is_integer_dtype(reread.pred_cells)
    np.testing.assert_array_equal(reread.pred_cells, submission.pred_cells)

    draft = pd.read_csv(ROOT/'results/submission_draft.csv')
    assert draft[['station_id', 'base_date', 'target_date']].equals(
        submission[['station_id', 'base_date', 'target_date']])
    ratio = np.log10(submission.pred_cells+1) - np.log10(draft.pred_cells+1)
    summary['difference_from_draft'] = {
        'rows_changed': int((submission.pred_cells != draft.pred_cells).sum()),
        'mean_log10_shift': float(ratio.mean()), 'max_abs_log10_shift': float(ratio.abs().max()),
        'rmse_log10_between_versions': float(np.sqrt((ratio**2).mean()))}
    summary['submission_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary['submission_rows'] = len(submission)
    summary['test_labels_used'] = False
    (ROOT/'results/refit2025_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('\nwrote', path)


if __name__ == '__main__':
    main()
