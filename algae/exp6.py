"""Version 6 strategy experiments, historical time splits only.

All v1 files are preserved. Stage-one candidates are fixed before CV. Selection
uses the 2018-2024 folds; 2025 is explicitly a reused reference validation year.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, log_loss

from algae.experiment import ROOT, design, folds, get_data, make_model
from algae.features_v6 import GROUPS, OUT as FEATURE_DIR

OUT = ROOT/'artifacts/cv6'
YEARS = list(range(2018, 2025))
VARIANTS = {
    'baseline': {'groups': [], 'method': 'delta'},
    'events': {'groups': ['events'], 'method': 'delta'},
    'biology': {'groups': ['bio'], 'method': 'delta'},
    'thermal': {'groups': ['thermal'], 'method': 'delta'},
    'state': {'groups': ['state'], 'method': 'delta'},
    'static': {'groups': ['events', 'bio'], 'method': 'delta'},
    'environment': {'groups': ['events', 'bio', 'thermal'], 'method': 'delta'},
    'all': {'groups': list(GROUPS), 'method': 'delta'},
    'event_weight': {'groups': [], 'method': 'event_weight'},
    'lds_weight': {'groups': [], 'method': 'lds_weight'},
    'environment_event_weight': {'groups': ['events', 'bio', 'thermal'], 'method': 'event_weight'},
    'environment_lds_weight': {'groups': ['events', 'bio', 'thermal'], 'method': 'lds_weight'},
    'rate': {'groups': [], 'method': 'rate'},
    'masked': {'groups': ['events', 'bio'], 'method': 'masked'},
    'history_mixup': {'groups': ['events', 'bio'], 'method': 'history_mixup'},
    'regime': {'groups': ['events', 'bio', 'thermal'], 'method': 'regime'},
}
PROTECTED = ['artifacts/selection.json', 'results/submission_draft.csv',
             'results/submission_refit2025.csv', 'algae/features.py', 'algae/experiment.py']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT/'plan.json'
    if not p.exists():
        plan = {'years': YEARS, 'since': 2017, 'variants': VARIANTS,
                'weights': 'normalized, bounded [.5,2] before normalization; no validation resampling',
                'selection': 'minimum mean historical log RMSE; multimetric result reported separately',
                'confirmation': '2025 reused reference, not untouched holdout',
                'protected': {f: digest(ROOT/f) for f in PROTECTED if (ROOT/f).exists()}}
        p.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    return json.loads(p.read_text(encoding='utf-8'))


def dataset(variant):
    frame = pd.read_pickle(FEATURE_DIR/'dataset.pkl')
    manifest = json.loads((FEATURE_DIR/'manifest.json').read_text(encoding='utf-8'))
    names = manifest['v1_features'][:]
    for group in VARIANTS[variant]['groups']:
        names.extend(GROUPS[group])
    return frame, names


def score(frame):
    y = frame.target.to_numpy(float)
    p = np.maximum(frame.pred_cells.to_numpy(float), 0)
    errors = np.log10(p+1)-np.log10(y+1)
    den = np.sum((y-y.mean())**2)
    nse = 1-np.sum((y-p)**2)/den if den else np.nan
    r = np.corrcoef(y, p)[0, 1] if y.std() and p.std() else 0.
    return {'n': len(y), 'rmse_log10': np.sqrt(np.mean(errors**2)),
            'mae_log10': np.abs(errors).mean(), 'nse': nse, 'r2_score': nse,
            'pearson_r2': r*r, 'f1_high': f1_score(y > 10000, p > 10000, zero_division=0)}


def states(delta):
    return np.where(delta < -.7, 0, np.where(delta > .7, 2, 1))


def training_weights(train, delta, method):
    if method == 'lds_weight':
        edges = np.linspace(-6, 6, 49)
        hist, _ = np.histogram(delta, edges)
        density = gaussian_filter1d(hist.astype(float), 1.5)+1
        bins = np.clip(np.searchsorted(edges, delta, side='right')-1, 0, len(hist)-1)
        w = np.sqrt(density.mean()/density[bins])
    elif method == 'event_weight':
        # Label information is training-only; one prolonged same-state event
        # must not dominate just because it was sampled twice weekly.
        f = train[['station_id', 'base_date']].copy()
        f['state'] = states(delta)
        f['position'] = np.arange(len(f))
        f = f.sort_values(['station_id', 'base_date'])
        change = f.station_id.ne(f.station_id.shift()) | f.state.ne(f.state.shift())
        change |= f.base_date.diff().dt.days.gt(14)
        f['episode'] = change.cumsum()
        size = f.groupby('episode').position.transform('size')
        episodes = f.drop_duplicates('episode').state.value_counts()
        rarity = np.sqrt(episodes.mean()/f.state.map(episodes))
        f['w'] = rarity/np.sqrt(size)
        w = f.sort_values('position').w.to_numpy(copy=True)
        w /= w.mean()
    else:
        w = np.ones(len(train))
    w = np.clip(w, .5, 2)
    return w/w.mean()


def classifier():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=180, num_leaves=7, max_depth=3,
                          min_child_samples=30, learning_rate=.035, reg_lambda=15,
                          random_state=42, n_jobs=4, verbosity=-1, deterministic=True,
                          force_col_wise=True)


def predict_prob(model, x):
    out = np.zeros((len(x), 3))
    p = model.predict_proba(x)
    for i, label in enumerate(model.classes_):
        out[:, int(label)] = p[:, i]
    return out


def gating_matrix(probs, lag):
    return np.column_stack([probs[:, 0], probs[:, 2],
                            probs[:, 0]*lag, probs[:, 2]*lag])


def regime_predict(train, test, names):
    """Inner forward OOF gate fitted ONLY on years before the outer year."""
    model = make_model('lightgbm')
    delta = np.log10(train.target.to_numpy(float)+1)-train.cells_log_lag0.to_numpy(float)
    model.fit(design(train, names, 'lightgbm'), delta)
    base_z = model.predict(design(test, names, 'lightgbm'))+test.cells_log_lag0.to_numpy(float)
    c = classifier()
    c.fit(design(train, names, 'lightgbm'), states(delta))
    test_probs = predict_prob(c, design(test, names, 'lightgbm'))
    inner_cache = OUT/'inner_regime'
    inner_cache.mkdir(exist_ok=True)
    xs, residuals, audit_probs, audit_truth = [], [], [], []
    for year in sorted(train.base_date.dt.year.unique()):
        t = train[train.target_date < pd.Timestamp(f'{year}-01-01')]
        v = train[train.base_date.dt.year.eq(year) & train.target_date.lt(pd.Timestamp(f'{year+1}-01-01'))]
        if len(t) < 150 or len(v) < 20:
            continue
        path = inner_cache/f'{year}.npz'
        if path.exists():
            stored = np.load(path)
            assert np.array_equal(stored['indices'], v.index.to_numpy())
            probs, pred = stored['probs'], stored['pred']
        else:
            d = np.log10(t.target.to_numpy(float)+1)-t.cells_log_lag0.to_numpy(float)
            inner = make_model('lightgbm')
            inner.fit(design(t, names, 'lightgbm'), d)
            pred = inner.predict(design(v, names, 'lightgbm'))+v.cells_log_lag0.to_numpy(float)
            ic = classifier()
            ic.fit(design(t, names, 'lightgbm'), states(d))
            probs = predict_prob(ic, design(v, names, 'lightgbm'))
            np.savez(path, indices=v.index.to_numpy(), probs=probs, pred=pred)
        xs.append(gating_matrix(probs, v.cells_log_lag0.to_numpy(float)))
        residuals.append(np.log10(v.target.to_numpy(float)+1)-pred)
        audit_probs.append(probs)
        audit_truth.append(states(np.log10(v.target.to_numpy(float)+1)-v.cells_log_lag0.to_numpy(float)))
    correction = np.zeros(len(test))
    audit = {'inner_n': sum(len(x) for x in xs), 'correction_strength': .5, 'correction_cap_log10': .3}
    gate = None
    if audit['inner_n'] >= 100:
        gate = Ridge(alpha=50)
        gate.fit(np.concatenate(xs), np.concatenate(residuals))
        correction = .5*np.clip(gate.predict(gating_matrix(test_probs, test.cells_log_lag0.to_numpy(float))), -.3, .3)
        audit['inner_classifier_logloss'] = float(log_loss(np.concatenate(audit_truth), np.concatenate(audit_probs), labels=[0, 1, 2]))
    audit['mean_abs_correction'] = float(np.abs(correction).mean())
    return {'model': model, 'classifier': c, 'gate': gate}, base_z+correction, audit


def fit(train, test, names, method, model_name='lightgbm'):
    train = train.sort_values(['base_date', 'station_id'])
    if method == 'regime':
        return regime_predict(train, test, names)
    lag = train.cells_log_lag0.to_numpy(float)
    delta = np.log10(train.target.to_numpy(float)+1)-lag
    w = training_weights(train, delta, method)
    y = delta.copy()
    audit = {'n_train_real': len(train), 'weight_min': float(w.min()), 'weight_max': float(w.max())}
    if method == 'rate':
        days = (train.target_date-train.base_date).dt.days.to_numpy(float)
        assert ((days >= 4) & (days <= 10)).all()
        y /= days
        audit['prediction_horizon'] = 7
    if method in ['masked', 'history_mixup']:
        aug = pd.read_pickle(FEATURE_DIR/('history_masked.pkl' if method == 'masked' else 'history_mixup.pkl'))
        # Admission by actual parent's membership in THIS training fold.
        aug = aug[aug.parent_index.isin(train.index)]
        if method == 'history_mixup':
            aug = aug[aug.donor_index.isin(train.index)]
            assert (aug.info_end < test.base_date.min().replace(month=1, day=1)).all()
        assert (aug.target_date < test.base_date.min().replace(month=1, day=1)).all()
        extra_y = np.log10(aug.target.to_numpy(float)+1)-aug.cells_log_lag0.to_numpy(float)
        y = np.concatenate([y, extra_y])
        w = np.concatenate([w, np.full(len(aug), .25)])
        audit['n_augmented'] = len(aug)
        train = pd.concat([train, aug], ignore_index=True)
    m = make_model(model_name)
    if model_name == 'lightgbm':
        m.fit(design(train, names, model_name), y, sample_weight=w)
    else:
        if method != 'delta':
            raise ValueError('TabPFN comparison uses the unweighted delta objective only')
        m.fit(design(train, names, model_name), y)
    pred = m.predict(design(test, names, model_name))
    if method == 'rate':
        pred *= 7
    return m, pred+test.cells_log_lag0.to_numpy(float), audit


def path_for(model, variant, year):
    return OUT/f'{year}_{model}_{variant}'


def run(variants, years, model='lightgbm', save=False):
    initialize()
    for v in variants:
        frame, names = dataset(v)
        for year in years:
            out = path_for(model, v, year)
            if (out/'metrics.json').exists():
                print('CACHED', out.name, flush=True)
                continue
            train, test = folds(frame, year, since=2017 if model == 'lightgbm' else 2012)
            started = time.time()
            print('START', out.name, len(train), len(names), flush=True)
            m, z, audit = fit(train, test, names, VARIANTS[v]['method'], model)
            z = np.maximum(np.asarray(z, float), 0)
            assert np.isfinite(z).all()
            p = test[['station_id', 'base_date', 'target_date', 'target', 'cells_log_lag0']].copy()
            p['pred_log10'] = z
            p['pred_cells'] = np.power(10., z)-1
            out.mkdir(exist_ok=True)
            p.to_csv(out/'predictions.csv', index=False)
            warm = p[p.base_date.dt.month.between(5, 10)]
            result = {'model': model, 'variant': v, 'year': year, 'n_features': len(names),
                      'seconds': time.time()-started, **audit, 'season': score(warm),
                      'all': score(p), 'by_station': {s: score(g) for s, g in warm.groupby('station_id')}}
            (out/'config.json').write_text(json.dumps({'features': names, 'spec': VARIANTS[v],
                                                      'train_target_max': str(train.target_date.max()),
                                                      'evaluation_role': 'historical' if year < 2025 else 'reused reference'}, indent=2), encoding='utf-8')
            if save:
                joblib.dump({'model': m, 'features': names, 'spec': VARIANTS[v]}, out/'model.joblib')
                restored = joblib.load(out/'model.joblib')
                saved = restored['model']
                if isinstance(saved, dict):
                    x = design(test, names, model)
                    replay = saved['model'].predict(x)
                    if saved['gate'] is not None:
                        probs = predict_prob(saved['classifier'], x)
                        replay += .5*np.clip(saved['gate'].predict(gating_matrix(probs, test.cells_log_lag0.to_numpy(float))), -.3, .3)
                else:
                    replay = saved.predict(design(test, names, model))
                if VARIANTS[v]['method'] == 'rate':
                    replay *= 7
                replay = np.maximum(replay+test.cells_log_lag0.to_numpy(float), 0)
                np.testing.assert_allclose(replay, z, rtol=1e-7, atol=1e-7)
                result['serialization_replay_verified'] = True
            (out/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            print('DONE', out.name, json.dumps(result['season']), flush=True)


def baseline(year):
    pieces = []
    for tag in ['tabpfn_delta_full_all_2012', 'lightgbm_delta_full_all_2017']:
        p = ROOT/'artifacts/cv'/f'{year}_{tag}'/'predictions.csv'
        if year == 2025:
            p = ROOT/'artifacts/holdout'/f'{year}_{tag}'/'predictions.csv'
        pieces.append(pd.read_csv(p, parse_dates=['base_date', 'target_date']))
    a, b = pieces
    keys = ['station_id', 'base_date', 'target_date']
    a = a.set_index(keys).sort_index()
    b = b.set_index(keys).reindex(a.index)
    assert b.pred_cells.notna().all()
    a['pred_log10'] = (np.log10(a.pred_cells+1)+np.log10(b.pred_cells+1))/2
    a['pred_cells'] = np.power(10., a.pred_log10)-1
    return a.reset_index(), pieces[0]


def summary():
    rows = []
    for path in OUT.glob('*_*/metrics.json'):
        m = json.loads(path.read_text(encoding='utf-8'))
        rows.append({'name': m['model']+'_'+m['variant'], 'year': m['year'], **m['season']})
    for year in YEARS+[2025]:
        b, tab = baseline(year)
        for name, p in [('v1_selected_5050', b), ('v1_tabpfn', tab)]:
            rows.append({'name': name, 'year': year, **score(p[p.base_date.dt.month.between(5, 10)])})
        # Fixed 50/50 pairing isolates changing only the LightGBM component.
        keys = ['station_id', 'base_date', 'target_date']
        for v in VARIANTS:
            path = path_for('lightgbm', v, year)/'predictions.csv'
            if not path.exists():
                continue
            p = pd.read_csv(path, parse_dates=['base_date', 'target_date']).set_index(keys).sort_index()
            t = tab.set_index(keys).reindex(p.index)
            assert t.pred_cells.notna().all()
            p['pred_cells'] = np.power(10., (np.log10(t.pred_cells+1)+np.log10(p.pred_cells+1))/2)-1
            p = p.reset_index()
            rows.append({'name': 'blend_'+v, 'year': year, **score(p[p.base_date.dt.month.between(5, 10)])})
        for v in ['thermal', 'environment']:
            path = path_for('tabpfn', v, year)/'predictions.csv'
            lpath = path_for('lightgbm', 'baseline', year)/'predictions.csv'
            if not path.exists():
                continue
            p = pd.read_csv(path, parse_dates=['base_date', 'target_date']).set_index(keys).sort_index()
            if lpath.exists():
                l = pd.read_csv(lpath, parse_dates=['base_date', 'target_date']).set_index(keys).reindex(p.index)
            else:
                loc = ROOT/'artifacts/holdout'/f'{year}_lightgbm_delta_full_all_2017'/'predictions.csv'
                l = pd.read_csv(loc, parse_dates=['base_date', 'target_date']).set_index(keys).reindex(p.index)
            p['pred_cells'] = np.power(10., (np.log10(p.pred_cells+1)+np.log10(l.pred_cells+1))/2)-1
            p = p.reset_index()
            rows.append({'name': 'blend_tabpfn_'+v, 'year': year, **score(p[p.base_date.dt.month.between(5, 10)])})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'scores.csv', index=False)
    historical = frame[frame.year.isin(YEARS)]
    complete = historical.groupby('name').year.nunique()
    keep = complete[complete.eq(len(YEARS))].index
    means = historical[historical.name.isin(keep)].groupby('name')[['rmse_log10', 'mae_log10', 'nse', 'pearson_r2']].mean()
    base = historical[historical.name.eq('v1_selected_5050')].set_index('year')
    for name in means.index:
        own = historical[historical.name.eq(name)].set_index('year').reindex(YEARS)
        d = own.rmse_log10-base.rmse_log10
        dn = own.nse-base.nse
        means.loc[name, 'rmse_vs_v1'] = d.mean()
        means.loc[name, 'rmse_diff_se'] = d.std(ddof=1)/np.sqrt(len(YEARS))
        means.loc[name, 'rmse_improved_years'] = (d < -1e-8).sum()
        means.loc[name, 'nse_vs_v1'] = dn.mean()
        means.loc[name, 'nse_diff_se'] = dn.std(ddof=1)/np.sqrt(len(YEARS))
    means.sort_values('rmse_log10').to_csv(OUT/'summary.csv')
    print(means.sort_values('rmse_log10').round(5).to_string(), flush=True)
    return frame, means


def freeze():
    _, means = summary()
    pool = means.loc[means.index.str.startswith('blend_')].sort_values('rmse_log10')
    best = pool.index[0]
    model = 'tabpfn' if best.startswith('blend_tabpfn_') else 'lightgbm'
    variant = best.removeprefix('blend_tabpfn_' if model == 'tabpfn' else 'blend_')
    selected = {'candidate': best, 'model': model, 'variant': variant,
                'rule': 'minimum mean 2018-2024 log RMSE among fixed 50/50 blends',
                'historical': means.loc[best].to_dict(), 'reference_year': 2025,
                'note': 'Research candidate, not an update of v1 selection or a submission.'}
    p = OUT/'frozen_candidate.json'
    if p.exists():
        assert json.loads(p.read_text())['candidate'] == best, 'Freeze would change; use a new version.'
    else:
        p.write_text(json.dumps(selected, indent=2), encoding='utf-8')
    print('FROZEN', best, flush=True)


def verify_baseline():
    checks = []
    for year in YEARS:
        fresh = pd.read_csv(path_for('lightgbm', 'baseline', year)/'predictions.csv')
        old = pd.read_csv(ROOT/'artifacts/cv'/f'{year}_lightgbm_delta_full_all_2017'/'predictions.csv')
        keys = ['station_id', 'base_date', 'target_date']
        old = old.set_index(keys).reindex(pd.MultiIndex.from_frame(fresh[keys]))
        error = float(np.max(np.abs(fresh.pred_cells.to_numpy()-old.pred_cells.to_numpy())))
        np.testing.assert_allclose(fresh.pred_cells, old.pred_cells, rtol=1e-9, atol=1e-7)
        checks.append({'year': year, 'max_abs_difference': error})
    plan = initialize()
    for path, h in plan['protected'].items():
        assert digest(ROOT/path) == h, f'Protected file changed: {path}'
    (OUT/'verification.json').write_text(json.dumps({'baseline_checks': checks, 'protected_unchanged': True}, indent=2), encoding='utf-8')
    print('VERIFIED baseline replay and protected files', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run', 'summary', 'freeze', 'verify'])
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument('--years', type=int, nargs='+', default=YEARS)
    parser.add_argument('--model', choices=['lightgbm', 'tabpfn'], default='lightgbm')
    parser.add_argument('--save', action='store_true')
    a = parser.parse_args()
    if a.command == 'run':
        run(a.variants, a.years, a.model, a.save)
    elif a.command == 'summary':
        summary()
    elif a.command == 'freeze':
        freeze()
    else:
        verify_baseline()
