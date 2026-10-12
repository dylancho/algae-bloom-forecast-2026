"""Version 8: same-day daily drivers (strategy T1, research/STRATEGY_RESEARCH_2026-10-11.md).

Variants on top of the v1 delta models, same folds/blends as v6/v7
(2018-2024 historical, months 5-10). 2025 stays a reused reference year.
  sameday_full : all v1 weather/dam windows end on base_date instead of base_date-1
  sameday_extra: v1 features + 11 day-d values (rain, minTa, wind, sun, dam rf/inflow/release jump)
  sameday_minta: v1 features + day-d minimum air temperature only (occurs before sampling)
  sameday_signed: v1 features + day-d / 3-day rain split into river vs lake (DCHN) columns
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from algae import sameday
from algae.exp6 import YEARS, fit, score
from algae.exp7 import KEYS, blend, components, load
from algae.experiment import ROOT, folds
from algae.features import load_bundle
from algae.features_v6 import OUT as FEATURE_DIR

OUT = ROOT/'artifacts/cv8'
VARIANTS = ['sameday_full', 'sameday_extra', 'sameday_minta', 'sameday_signed']
DAILY_PREFIX = ('wx_', 'dam')


def frame_sd():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT/'sameday.pkl'
    if not p.exists():
        tables = load_bundle(ROOT/'02_dataset_bundle_algae2026.zip')
        sameday.build(tables, tables['eval_pairs']).to_pickle(p)
    return pd.read_pickle(p)


def dataset(variant):
    frame = pd.read_pickle(FEATURE_DIR/'dataset.pkl')
    v1 = json.loads((FEATURE_DIR/'manifest.json').read_text(encoding='utf-8'))['v1_features']
    sd = frame_sd()
    merged = frame.merge(sd, on=KEYS, how='left', validate='one_to_one')
    merged.index = frame.index
    assert merged.sd_minTa.notna().mean() > .95
    if variant == 'sameday_full':
        names = []
        for c in v1:
            if c.startswith(DAILY_PREFIX):
                merged[c] = merged[f'sdf_{c}']
            names.append(c)
    elif variant == 'sameday_extra':
        names = v1+sameday.EXTRA
    elif variant == 'sameday_signed':
        # added after seeing pooled day-d rain cancel by station type (pre-stated hypothesis R2)
        names = v1+sameday.SIGNED
    else:
        names = v1+sameday.MINTA
    return merged, names


def run(variants, years, model):
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
            (out/'config.json').write_text(json.dumps({'features': names}, indent=2), encoding='utf-8')
            (out/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            print('DONE', out.name, json.dumps(result['season']), flush=True)


def drop_sse(p, threshold=-0.5):
    """Raw-unit SSE share on pairs whose observed log change is a big drop."""
    y = p.target.to_numpy(float)
    d = np.log10(y+1)-p.cells_log_lag0.to_numpy(float)
    e = (np.maximum(p.pred_cells.to_numpy(float), 0)-y)**2
    lg = (np.log10(np.maximum(p.pred_cells.to_numpy(float), 0)+1)-np.log10(y+1))**2
    m = d < threshold
    return {'drop_n': int(m.sum()), 'drop_rmse_log10': float(np.sqrt(lg[m].mean())) if m.any() else np.nan,
            'drop_sse_raw': float(e[m].sum())}


def summary(years=YEARS):
    rows = []
    for year in years:
        tab, lgb = components(year)
        cands = {'v1_selected_5050': blend(tab, lgb)}
        for v in VARIANTS:
            lp = OUT/f'{year}_lightgbm_{v}'/'predictions.csv'
            tp = OUT/f'{year}_tabpfn_{v}'/'predictions.csv'
            if lp.exists():
                cands[f'blend_lgb_{v}'] = blend(tab, load(lp))
            if tp.exists():
                cands[f'blend_tab_{v}'] = blend(load(tp), lgb)
            if lp.exists() and tp.exists():
                cands[f'blend_both_{v}'] = blend(load(tp), load(lp))
        lgb_only = lgb.reset_index()
        cands['v1_lgb_only'] = lgb_only
        for v in VARIANTS:
            lp = OUT/f'{year}_lightgbm_{v}'/'predictions.csv'
            if lp.exists():
                cands[f'lgb_only_{v}'] = load(lp).reset_index()
        for name, p in cands.items():
            warm = p[p.base_date.dt.month.between(5, 10)]
            rows.append({'name': name, 'year': year, **score(warm), **drop_sse(warm)})
            for s, g in warm.groupby('station_id'):
                rows.append({'name': name, 'year': year, 'station': s, **score(g)})
    frame = pd.DataFrame(rows)
    tag = '' if years == YEARS else '_2025'
    frame.to_csv(OUT/f'scores{tag}.csv', index=False)
    total = frame[frame.station.isna()]
    complete = total.groupby('name').year.nunique()
    keep = complete[complete.eq(len(years))].index
    cols = ['rmse_log10', 'mae_log10', 'nse', 'pearson_r2', 'drop_rmse_log10', 'drop_sse_raw']
    means = total[total.name.isin(keep)].groupby('name')[cols].mean()
    for name in means.index:
        ref = 'v1_lgb_only' if name.startswith('lgb_only') else 'v1_selected_5050'
        base = total[total.name.eq(ref)].set_index('year').reindex(years)
        own = total[total.name.eq(name)].set_index('year').reindex(years)
        d = own.rmse_log10-base.rmse_log10
        means.loc[name, 'ref'] = ref
        means.loc[name, 'rmse_vs_ref'] = d.mean()
        means.loc[name, 'rmse_diff_se'] = d.std(ddof=1)/np.sqrt(len(years)) if len(years) > 1 else np.nan
        means.loc[name, 'improved_years'] = (d < -1e-8).sum()
        means.loc[name, 'nse_vs_ref'] = (own.nse-base.nse).mean()
        means.loc[name, 'drop_rmse_vs_ref'] = (own.drop_rmse_log10-base.drop_rmse_log10).mean()
    means = means.sort_values(['ref', 'rmse_log10'])
    means.to_csv(OUT/f'summary{tag}.csv')
    with pd.option_context('display.width', 250):
        print(means.round(5).to_string(), flush=True)
    return frame, means


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['build', 'run', 'summary'])
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=VARIANTS)
    parser.add_argument('--years', type=int, nargs='+', default=YEARS)
    parser.add_argument('--model', choices=['lightgbm', 'tabpfn'], default='lightgbm')
    a = parser.parse_args()
    if a.command == 'build':
        print(frame_sd().describe().T.round(3).to_string())
    elif a.command == 'run':
        run(a.variants, a.years, a.model)
    else:
        summary(a.years)
