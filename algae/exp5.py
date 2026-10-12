"""Version 5: score on the official metric set and attack spike under-prediction.

The bundle README ranks on log10 RMSE/MAE and raw-unit R2 and NSE, scored on
May-Oct only. R2 and NSE are listed separately, so R2 is the squared Pearson
correlation and NSE is 1 - SSE/SST. The >10,000 cells/mL F1 is reference only.

All selection uses historical CV folds 2018-2024 (the years every v1-v4 setting
has). 2025 stays a confirmation year for whatever is fixed here.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from algae.experiment import ROOT, design, fit_predict, folds, get_data

YEARS = list(range(2018, 2025))
CAUTION = 10_000
OUT = ROOT/'artifacts/cv5'


def score(y, cells):
    y = np.asarray(y, float)
    p = np.maximum(np.asarray(cells, float), 0)
    err = np.log10(p+1)-np.log10(y+1)
    sst = np.sum((y-y.mean())**2)
    r = np.corrcoef(y, p)[0, 1] if p.std() > 0 and y.std() > 0 else 0.0
    hit, pred_pos = y > CAUTION, p > CAUTION
    tp = np.sum(hit & pred_pos)
    f1 = 2*tp/(hit.sum()+pred_pos.sum()) if hit.sum()+pred_pos.sum() else np.nan
    return {'n': len(y), 'rmse_log10': float(np.sqrt(np.mean(err**2))),
            'mae_log10': float(np.abs(err).mean()),
            'r2': float(r**2), 'nse': float(1-np.sum((y-p)**2)/sst),
            'f1_caution': float(f1)}


def warm(frame):
    return frame[pd.to_datetime(frame.base_date).dt.month.between(5, 10)]


def load_cv(tag, year):
    for stage in ['cv', 'cv5']:
        f = ROOT/'artifacts'/stage/f'{year}_{tag}'/'predictions.csv'
        if f.exists():
            return pd.read_csv(f, parse_dates=['base_date', 'target_date'])
    return None


def blend(parts, weights, space='log'):
    """Blend prediction frames (pred_cells) aligned on key order."""
    if space == 'log':
        z = sum(w*np.log10(np.maximum(p.pred_cells.to_numpy(), 0)+1) for p, w in zip(parts, weights))
        cells = np.power(10., z)-1
    else:
        cells = sum(w*np.maximum(p.pred_cells.to_numpy(), 0) for p, w in zip(parts, weights))
    out = parts[0][['station_id', 'base_date', 'target_date', 'target', 'cells_log_lag0']].copy()
    out['pred_cells'] = cells
    return out


# ---- new models -----------------------------------------------------------

def lgb(**kw):
    from lightgbm import LGBMRegressor
    params = dict(n_estimators=500, num_leaves=15, max_depth=5, min_child_samples=30,
                  learning_rate=.035, reg_lambda=10, reg_alpha=.1, colsample_bytree=.9,
                  random_state=42, n_jobs=4, verbosity=-1, deterministic=True,
                  force_col_wise=True)
    params.update(kw)
    return LGBMRegressor(**params)


VARIANTS = {
    # raw-scale conditional mean; log link keeps predictions positive
    'tweedie13': dict(kind='tweedie', power=1.3),
    'tweedie15': dict(kind='tweedie', power=1.5),
    'tweedie17': dict(kind='tweedie', power=1.7),
    # log-delta model, rows with large targets weigh more
    'logw05': dict(kind='logw', alpha=.5),
    'logw10': dict(kind='logw', alpha=1.0),
    # upper quantile of the log-delta
    'q60': dict(kind='quantile', q=.6),
    'q70': dict(kind='quantile', q=.7),
    # sqrt target: between log and raw squared error
    'sqrt': dict(kind='sqrt'),
}


def fit_variant(spec, train, test, names):
    train = train.sort_values(['base_date', 'station_id'])
    xtr, xte = design(train, names, 'lightgbm'), design(test, names, 'lightgbm')
    y = train.target.to_numpy(float)
    lag_tr, lag_te = train.cells_log_lag0.to_numpy(float), test.cells_log_lag0.to_numpy(float)
    kind = spec['kind']
    if kind == 'tweedie':
        m = lgb(objective='tweedie', tweedie_variance_power=spec['power'])
        m.fit(xtr, y)
        return np.maximum(m.predict(xte), 0)
    if kind == 'sqrt':
        m = lgb()
        m.fit(xtr, np.sqrt(y))
        return np.maximum(m.predict(xte), 0)**2
    d = np.log10(y+1)-lag_tr
    if kind == 'logw':
        # weight 1 at <=1,000 cells, growing with log level above that
        w = 1+spec['alpha']*np.maximum(np.log10(y+1)-3, 0)**2
        m = lgb()
        m.fit(xtr, d, sample_weight=w)
    elif kind == 'quantile':
        m = lgb(objective='quantile', alpha=spec['q'])
        m.fit(xtr, d)
    z = np.maximum(m.predict(xte)+lag_te, 0)
    return np.power(10., z)-1


def run(variants, years):
    data, names = get_data('full')
    for v in variants:
        for year in years:
            out = OUT/f'{year}_lgb5_{v}'
            if (out/'predictions.csv').exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            train, test = folds(data, year, 'all', 2012)
            t = time.time()
            cells = fit_variant(VARIANTS[v], train, test, names)
            pred = test[['station_id', 'base_date', 'target_date', 'target', 'cells_log_lag0']].copy()
            pred['pred_cells'] = cells
            pred['pred_log10'] = np.log10(np.maximum(cells, 0)+1)
            pred.to_csv(out/'predictions.csv', index=False)
            (out/'config.json').write_text(json.dumps({'variant': v, **VARIANTS[v], 'year': year,
                                                       'since': 2012, 'n_train': len(train),
                                                       'seconds': time.time()-t}, indent=2))
            print('DONE', out.name, json.dumps(score(warm(pred).target, warm(pred).pred_cells)), flush=True)


# ---- scoring --------------------------------------------------------------

BASE = ['tabpfn_delta_full_all_2012', 'lightgbm_delta_full_all_2017', 'lightgbm_delta_full_all_2012',
        'catboost_delta_full_all_2012', 'tabpfn_direct_full_all_2012']


def candidates(year):
    """Every prediction set to score for one year, keyed by name."""
    c = {}
    for tag in BASE:
        p = load_cv(tag, year)
        if p is not None:
            c[tag] = p
    for v in VARIANTS:
        p = load_cv(f'lgb5_{v}', year)
        if p is not None:
            c[f'lgb5_{v}'] = p
    for _, tag in CROSS+WAMIS+WAMIS_X+INTER:
        p = load_cv(tag, year)
        if p is not None:
            c[tag] = p
    for name, t_tag, l_tag in [('x_5050', 'tabpfnx_delta_2012', 'lgbx_delta_2017'),
                               ('w_5050', 'tabpfnw_delta_2012', 'lgbw_delta_2017'),
                               ('xw_5050', 'tabpfnxw_delta_2012', 'lgbxw_delta_2017'),
                               ('i_5050', 'tabpfni_delta_2012', 'lgbi_delta_2017')]:
        if t_tag in c and l_tag in c:
            c[name] = blend([c[t_tag], c[l_tag]], [.5, .5])
    t = c['tabpfn_delta_full_all_2012']
    pers = t.copy()
    pers['pred_cells'] = np.power(10., t.cells_log_lag0)-1
    c['persistence'] = pers
    c['v1_selected_5050'] = blend([t, c['lightgbm_delta_full_all_2017']], [.5, .5])
    for v in VARIANTS:
        k = f'lgb5_{v}'
        if k in c:
            for w in [.3, .5]:
                c[f'tabpfn+{v}_raw{w}'] = blend([t, c[k]], [1-w, w], 'raw')
                c[f'tabpfn+{v}_log{w}'] = blend([t, c[k]], [1-w, w], 'log')
    return c


def rescore(years):
    rows = []
    for year in years:
        for name, p in candidates(year).items():
            w = warm(p)
            rows.append({'name': name, 'year': year, **score(w.target, w.pred_cells)})
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT/'artifacts/cv5_scores.csv', index=False)
    full = frame.groupby('name').year.nunique()
    keep = full[full == len(years)].index
    summary = (frame[frame.name.isin(keep)].groupby('name')
               [['rmse_log10', 'mae_log10', 'r2', 'nse', 'f1_caution']].mean()
               .sort_values('nse', ascending=False))
    summary.to_csv(ROOT/'artifacts/cv5_summary.csv')
    pd.set_option('display.width', 200)
    print(summary.round(4).to_string())
    return frame


# ---- cross-station features -------------------------------------------------

NAKDONG = ['HP', 'GJGR', 'CS', 'MGMR']  # upstream -> downstream


def cross_station(data):
    """Other stations' algae state known on base_date (measurements <= base_date).

    Same availability assumption as the station's own lag0 feature. For a Nakdong
    site, `up_*` summarises stations upstream of it; DCHN (Geum) gets NaN there.
    """
    from algae.features import TARGET, load_bundle
    a = load_bundle(ROOT/'02_dataset_bundle_algae2026.zip')['algae_measure']
    a = a[['station_id', 'chck_de', TARGET]].dropna()
    a['date'] = pd.to_datetime(a.chck_de)
    a['logc'] = np.log10(a[TARGET].clip(lower=0)+1)
    series = {s: g.sort_values('date').set_index('date').logc for s, g in a.groupby('station_id')}
    rows = []
    for site, base in zip(data.station_id, data.base_date):
        row = {}
        for other in NAKDONG:
            s = series[other].loc[:base]
            recent = s.loc[base-pd.Timedelta(days=14):]
            row[f'x_{other}_log'] = s.iloc[-1] if len(s) else np.nan
            row[f'x_{other}_age'] = (base-s.index[-1]).days if len(s) else np.nan
            row[f'x_{other}_max14'] = recent.max() if len(recent) else np.nan
        if site in NAKDONG:
            ups = NAKDONG[:NAKDONG.index(site)]
            vals = [row[f'x_{u}_log'] for u in ups if row[f'x_{u}_age'] <= 10]
            row['up_max_log'] = max(vals) if vals else np.nan
            row['up_minus_own'] = row['up_max_log']-row[f'x_{site}_log'] if vals else np.nan
        else:
            row['up_max_log'] = row['up_minus_own'] = np.nan
        row['nak_mean_log'] = np.nanmean([row[f'x_{o}_log'] for o in NAKDONG
                                          if row[f'x_{o}_age'] <= 10] or [np.nan])
        rows.append(row)
    return pd.DataFrame(rows, index=data.index)


def interactions(data):
    """Heat x stagnation x seed interactions from inputs already known on base_date.

    Local river flow comes from the WAMIS gauge next to the site (g0, <= base_date-1);
    dam release from the bundle. Water temperature is the latest measurement.
    """
    from algae.wamis import build
    g = build(data)
    flow7 = np.power(10., g.g0_flow_log_7d)-1
    flow3 = np.power(10., g.g0_flow_log_3d)-1
    release7 = np.power(10., data.dam0_tdqty_logmean_7d)-1
    wt, ta = data.wq_iem_wtrtp, data.wx_avgTa_7d
    heat = (wt-20).clip(lower=0)
    out = pd.DataFrame(index=data.index)
    out['i_ta_over_flow'] = ta/(flow7+1)
    out['i_wt_over_flow'] = wt/(flow3+1)
    out['i_ta_over_release'] = ta/(release7+1)
    out['i_heat_x_stagnation'] = heat/np.log10(flow7+10)
    out['i_heat_x_seed'] = heat*data.cells_log_lag0
    out['i_heat_x_light'] = heat*data.wx_sumSsHr_7d
    out['i_seed_x_stagnation'] = data.cells_log_lag0/np.log10(flow7+10)
    out['i_growth_x_heat'] = data.growth_per_day*heat
    return out


def run_cross(models, years, extras=('cross',)):
    data, names = get_data('full')
    for kind in extras:
        if kind == 'cross':
            extra = cross_station(data)
        elif kind == 'inter':
            extra = interactions(data)
        else:
            from algae.wamis import build
            extra = build(data)
        data = data.join(extra)
        names = names+list(extra.columns)
    for model, tag in models:
        since = 2017 if tag.endswith('2017') else 2012
        for year in years:
            out = OUT/f'{year}_{tag}'
            if (out/'predictions.csv').exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            train, test = folds(data, year, 'all', since)
            t = time.time()
            _, z = fit_predict(model, 'delta', train, test, names)
            pred = test[['station_id', 'base_date', 'target_date', 'target', 'cells_log_lag0']].copy()
            pred['pred_log10'] = z
            pred['pred_cells'] = np.power(10., z)-1
            pred.to_csv(out/'predictions.csv', index=False)
            (out/'config.json').write_text(json.dumps({'tag': tag, 'model': model, 'since': since,
                                                       'year': year, 'n_train': len(train),
                                                       'n_features': len(names),
                                                       'seconds': time.time()-t}, indent=2))
            w = warm(pred)
            print('DONE', out.name, json.dumps(score(w.target, w.pred_cells)), flush=True)


CROSS = [('lightgbm', 'lgbx_delta_2012'), ('lightgbm', 'lgbx_delta_2017'),
         ('tabpfn', 'tabpfnx_delta_2012')]
# WAMIS river gauges alone (w) and with cross-station features (xw)
WAMIS = [('lightgbm', 'lgbw_delta_2017'), ('tabpfn', 'tabpfnw_delta_2012')]
WAMIS_X = [('lightgbm', 'lgbxw_delta_2017'), ('tabpfn', 'tabpfnxw_delta_2012')]
# heat x stagnation interactions on top of v1 features
INTER = [('lightgbm', 'lgbi_delta_2017'), ('lightgbm', 'lgbi_delta_2012'), ('tabpfn', 'tabpfni_delta_2012')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['run', 'rescore', 'cross', 'wamis', 'inter'])
    ap.add_argument('--variants', nargs='*', default=list(VARIANTS))
    ap.add_argument('--years', nargs='*', type=int, default=YEARS)
    a = ap.parse_args()
    if a.command == 'run':
        run(a.variants, a.years)
    elif a.command == 'cross':
        run_cross(CROSS, a.years)
    elif a.command == 'wamis':
        run_cross(WAMIS, a.years, ('wamis',))
        run_cross(WAMIS_X, a.years, ('cross', 'wamis'))
    elif a.command == 'inter':
        run_cross(INTER, a.years, ('inter',))
    else:
        rescore(a.years)


if __name__ == '__main__':
    main()
