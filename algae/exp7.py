"""Version 7: external nutrient data (NIER water-quality network) on top of the v6 frame.

Same folds, models, and 50/50 blend comparison as v6 (2018-2024 historical,
5-10 months). 2025 stays a reused reference year. v1-v6 artifacts are untouched.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from algae import autowq, nutrients
from algae.exp6 import OUT as CV6, YEARS, fit, score
from algae.experiment import ROOT, folds
from algae.features_v6 import GROUPS, OUT as FEATURE_DIR

OUT = ROOT/'artifacts/cv7'
ENV = GROUPS['events']+GROUPS['bio']+GROUPS['thermal']
VARIANTS = {
    'nutr': {'lag': 1, 'slots': (0, 1), 'env': False},
    'nutr_near': {'lag': 1, 'slots': (0,), 'env': False},
    'nutr_lag14': {'lag': 14, 'slots': (0, 1), 'env': False},
    'env_nutr': {'lag': 1, 'slots': (0, 1), 'env': True},
    # automatic network (daily) features; 'slots' empty = no grab-sample nutrients
    'auto': {'lag': 1, 'slots': (), 'env': False, 'auto': True},
    'auto_bio': {'lag': 1, 'slots': (), 'env': False, 'auto': 'bio'},
    'auto_nutr': {'lag': 1, 'slots': (0, 1), 'env': False, 'auto': True},
}
KEYS = ['station_id', 'base_date', 'target_date']


def nutrient_frame(lag):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT/f'nutrients_lag{lag}.pkl'
    if not p.exists():
        base = pd.read_pickle(FEATURE_DIR/'dataset.pkl')
        nutrients.build(base, lag).to_pickle(p)
    return pd.read_pickle(p)


def dataset(variant):
    spec = VARIANTS[variant]
    frame = pd.read_pickle(FEATURE_DIR/'dataset.pkl')
    manifest = json.loads((FEATURE_DIR/'manifest.json').read_text(encoding='utf-8'))
    nu = nutrient_frame(spec['lag'])
    cols = nutrients.names(nu, spec['slots'])
    frame = frame.join(nu[cols])
    if spec.get('auto'):
        au = auto_frame()
        acols = [c for c in au if spec['auto'] is True or c.startswith('au_bio_')]
        frame = frame.join(au[acols])
        cols = cols+acols
    names = manifest['v1_features']+(ENV if spec['env'] else [])+cols
    return frame, names


def auto_frame():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT/'autowq.pkl'
    if not p.exists():
        autowq.build(pd.read_pickle(FEATURE_DIR/'dataset.pkl')).to_pickle(p)
    return pd.read_pickle(p)


def run(variants, years, model='lightgbm'):
    for v in variants:
        frame, names = dataset(v)
        for year in years:
            out = OUT/f'{year}_{model}_{v}'
            if (out/'metrics.json').exists():
                print('CACHED', out.name, flush=True)
                continue
            train, test = folds(frame, year, since=2017 if model == 'lightgbm' else 2012)
            started = time.time()
            print('START', out.name, len(train), len(names), flush=True)
            _, z, audit = fit(train, test, names, 'delta', model)
            z = np.maximum(np.asarray(z, float), 0)
            assert np.isfinite(z).all()
            p = test[KEYS+['target', 'cells_log_lag0']].copy()
            p['pred_log10'] = z
            p['pred_cells'] = np.power(10., z)-1
            out.mkdir(parents=True, exist_ok=True)
            p.to_csv(out/'predictions.csv', index=False)
            warm = p[p.base_date.dt.month.between(5, 10)]
            result = {'model': model, 'variant': v, 'year': year, 'n_features': len(names),
                      'seconds': time.time()-started, **audit, 'season': score(warm),
                      'by_station': {s: score(g) for s, g in warm.groupby('station_id')}}
            (out/'config.json').write_text(json.dumps({'features': names, 'spec': VARIANTS[v]}, indent=2,
                                                      ensure_ascii=False), encoding='utf-8')
            (out/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            print('DONE', out.name, json.dumps(result['season']), flush=True)


def load(path):
    return pd.read_csv(path, parse_dates=['base_date', 'target_date']).set_index(KEYS).sort_index()


def blend(a, b):
    b = b.reindex(a.index)
    assert b.pred_cells.notna().all()
    p = a.copy()
    p['pred_cells'] = np.power(10., (np.log10(a.pred_cells+1)+np.log10(b.pred_cells+1))/2)-1
    return p.reset_index()


def components(year):
    """v1 components: TabPFN delta 2012 and LightGBM delta 2017."""
    root = ROOT/('artifacts/holdout' if year == 2025 else 'artifacts/cv')
    tab = load(root/f'{year}_tabpfn_delta_full_all_2012'/'predictions.csv')
    lgb = load(root/f'{year}_lightgbm_delta_full_all_2017'/'predictions.csv')
    return tab, lgb


def summary(years=YEARS):
    rows = []
    for year in years:
        tab, lgb = components(year)
        cands = {'v1_selected_5050': blend(tab, lgb)}
        env_tab = CV6/f'{year}_tabpfn_environment'/'predictions.csv'
        if env_tab.exists():
            cands['v6_blend_tabpfn_environment'] = blend(load(env_tab), lgb)
        for v in VARIANTS:
            lp = OUT/f'{year}_lightgbm_{v}'/'predictions.csv'
            tp = OUT/f'{year}_tabpfn_{v}'/'predictions.csv'
            if lp.exists():
                cands[f'blend_lgb_{v}'] = blend(tab, load(lp))
            if tp.exists():
                cands[f'blend_tab_{v}'] = blend(load(tp), lgb)
            if lp.exists() and tp.exists():
                cands[f'blend_both_{v}'] = blend(load(tp), load(lp))
        for name, p in cands.items():
            warm = p[p.base_date.dt.month.between(5, 10)]
            rows.append({'name': name, 'year': year, **score(warm)})
            for s, g in warm.groupby('station_id'):
                rows.append({'name': name, 'year': year, 'station': s, **score(g)})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/('scores.csv' if years == YEARS else 'scores_2025.csv'), index=False)
    total = frame[frame.station.isna()] if 'station' in frame else frame
    complete = total.groupby('name').year.nunique()
    keep = complete[complete.eq(len(years))].index
    means = total[total.name.isin(keep)].groupby('name')[['rmse_log10', 'mae_log10', 'nse', 'pearson_r2']].mean()
    base = total[total.name.eq('v1_selected_5050')].set_index('year')
    for name in means.index:
        own = total[total.name.eq(name)].set_index('year').reindex(years)
        d, dn = own.rmse_log10-base.rmse_log10, own.nse-base.nse
        means.loc[name, 'rmse_vs_v1'] = d.mean()
        means.loc[name, 'rmse_diff_se'] = d.std(ddof=1)/np.sqrt(len(years)) if len(years) > 1 else np.nan
        means.loc[name, 'improved_years'] = (d < -1e-8).sum()
        means.loc[name, 'nse_vs_v1'] = dn.mean()
    means = means.sort_values('rmse_log10')
    means.to_csv(OUT/('summary.csv' if years == YEARS else 'summary_2025.csv'))
    print(means.round(5).to_string(), flush=True)
    return frame, means


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run', 'summary'])
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument('--years', type=int, nargs='+', default=YEARS)
    parser.add_argument('--model', choices=['lightgbm', 'tabpfn'], default='lightgbm')
    a = parser.parse_args()
    if a.command == 'run':
        run(a.variants, a.years, a.model)
    else:
        summary(a.years)
