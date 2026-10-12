"""Pseudo training pairs from domain-informed daily interpolation of cell counts.

Weekly sampling leaves 6 unobserved days between measurements. For training only,
we estimate those days and create extra (base_date, +7d) pairs:

- `loglin`: log10 cells change at a constant rate between two measurements
  (constant exponential growth/decay).
- `domain`: the same total change, distributed by daily drivers. Rises follow a
  cardinal-temperature growth curve for Microcystis (Tmin 10, Topt 29, Tmax 40 C;
  growth starts accelerating ~25 C and peaks ~29 C in culture studies). Falls follow
  daily river flow at the adjacent WAMIS gauge (flushing lowers cell density).
  Daily water temperature is linearly interpolated between measurements.

Interpolation uses the next real measurement, so each pseudo pair records
`info_end` (date of the last real value used) and CV keeps only pairs with
info_end before the evaluation year. Pseudo rows never enter evaluation.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from algae.experiment import ROOT
from algae.features import TARGET, WATER, GENERA, build_features, load_bundle

OUT = ROOT/'artifacts/augment'
OFFSETS = [2, 3, 4]   # pseudo base = measurement day + offset
BATCHES = 6           # pseudo rows in one feature build are >= 6 gaps (~6 weeks) apart


def ctmi(t, tmin=10., topt=29., tmax=40.):
    """Rosso cardinal temperature model, scaled to 1 at topt."""
    t = np.asarray(t, float)
    num = (t-tmax)*(t-tmin)**2
    den = (topt-tmin)*((topt-tmin)*(t-topt)-(topt-tmax)*(topt+tmin-2*t))
    out = np.where((t > tmin) & (t < tmax), num/den, 0.)
    return np.nan_to_num(out, nan=0.)


def daily_curve(meas, wt, flow, method):
    """Daily log10 cells between consecutive measurements of one station."""
    days = pd.date_range(meas.index[0], meas.index[-1], freq='D')
    curve = pd.Series(np.nan, index=days)
    wt_daily = wt.reindex(days).interpolate(limit_area='inside')
    for (t0, v0), (t1, v1) in zip(meas.items(), list(meas.items())[1:]):
        span = pd.date_range(t0, t1, freq='D')
        steps = span[1:]
        if method == 'loglin' or len(steps) == 0:
            w = np.ones(len(steps))
        elif v1 >= v0:
            w = ctmi(wt_daily.reindex(steps).to_numpy())+.05
        else:
            f = flow.reindex(steps).to_numpy()
            w = np.where(np.isfinite(f), f, np.nanmean(f) if np.isfinite(f).any() else 1.)+1e-3
        frac = np.concatenate([[0.], np.cumsum(w)/w.sum()]) if len(steps) else np.array([0.])
        curve.loc[span] = v0+(v1-v0)*frac
    return curve


def build(method):
    from algae.wamis import SLOTS, load as load_gauge
    tables = load_bundle(ROOT/'02_dataset_bundle_algae2026.zip')
    a = tables['algae_measure'].copy()
    a['date'] = pd.to_datetime(a.chck_de)
    a[TARGET] = pd.to_numeric(a[TARGET], errors='coerce')
    a['iem_wtrtp'] = pd.to_numeric(a['iem_wtrtp'], errors='coerce')
    pseudo = []
    for site, g in a.groupby('station_id'):
        g = g.sort_values('date').set_index('date')
        meas = np.log10(g[TARGET].dropna().clip(lower=0)+1)
        meas = meas[~meas.index.duplicated()]
        wt = g['iem_wtrtp'].dropna()
        wt = wt[~wt.index.duplicated()]
        curve = daily_curve(meas, wt, load_gauge(SLOTS[site][0], 'flw'), method)
        times = list(meas.index)
        occupied = set(g.index)  # real rows, including ones whose cell count is missing
        for i, (t0, t1) in enumerate(zip(times, times[1:])):
            for o in OFFSETS:
                d = t0+pd.Timedelta(days=o)
                if d >= t1 or d in occupied:
                    continue
                target_day = d+pd.Timedelta(days=7)
                if target_day > times[-1]:
                    continue
                info_end = next(t for t in times if t >= target_day)
                pseudo.append({'station_id': site, 'base_date': d, 'target_date': target_day,
                               'log_base': curve[d], 'log_target': curve[target_day],
                               'info_end': info_end, 'gap_index': i, 'offset': o})
    pseudo = pd.DataFrame(pseudo)
    parts = []
    for o in OFFSETS:
        for b in range(BATCHES):
            batch = pseudo[(pseudo.offset == o) & (pseudo.gap_index % BATCHES == b)]
            if batch.empty:
                continue
            extra = pd.DataFrame({'station_id': batch.station_id, 'chck_de': batch.base_date.dt.strftime('%Y-%m-%d'),
                                  TARGET: np.power(10., batch.log_base)-1})
            for col in WATER+GENERA:
                extra[col] = np.nan
            mod = dict(tables)
            mod['algae_measure'] = pd.concat([tables['algae_measure'], extra], ignore_index=True)
            feats, _ = build_features(mod, batch[['station_id', 'base_date']])
            feats.index = batch.index
            parts.append(batch.join(feats))
            print('batch', o, b, len(batch), flush=True)
    out = pd.concat(parts).sort_index()
    out['target'] = np.power(10., out.log_target)-1
    out['split'] = 'pseudo'
    assert np.allclose(out.cells_log_lag0, out.log_base, atol=1e-9)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_pickle(OUT/f'{method}.pkl')
    print(method, len(out), 'pseudo pairs')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['loglin', 'domain'], nargs='+', default=['loglin', 'domain'])
    for m in ap.parse_args().method:
        build(m)


# ---- CV experiment ----------------------------------------------------------

def run_cv(method, weight, model, since, years=range(2018, 2025), max_pseudo=None):
    import json
    import time
    from algae.experiment import design, folds, get_data, make_model
    from algae.exp5 import OUT as CV5, score, warm
    data, names = get_data('full')
    pseudo = pd.read_pickle(OUT/f'{method}.pkl')
    tag = f'aug_{method}_w{weight}_{model}_{since}'
    for year in years:
        out = CV5/f'{year}_{tag}'
        if (out/'predictions.csv').exists():
            continue
        start = pd.Timestamp(year=year, month=1, day=1)
        train, test = folds(data, year, 'all', since)
        extra = pseudo[(pseudo.info_end < start) & (pseudo.base_date.dt.year >= since)]
        if max_pseudo and len(extra) > max_pseudo:
            extra = extra.sample(max_pseudo, random_state=42)
        both = pd.concat([train.assign(_w=1.), extra.assign(_w=weight)]).sort_values(['base_date', 'station_id'])
        y = np.log10(both.target.to_numpy(float)+1)-both.cells_log_lag0.to_numpy(float)
        est = make_model(model)
        t = time.time()
        if model == 'lightgbm':
            est.fit(design(both, names, model), y, sample_weight=both._w.to_numpy())
        else:
            est.fit(design(both, names, model), y)
        z = np.maximum(est.predict(design(test, names, model))+test.cells_log_lag0.to_numpy(float), 0)
        pred = test[['station_id', 'base_date', 'target_date', 'target', 'cells_log_lag0']].copy()
        pred['pred_log10'] = z
        pred['pred_cells'] = np.power(10., z)-1
        out.mkdir(parents=True, exist_ok=True)
        pred.to_csv(out/'predictions.csv', index=False)
        (out/'config.json').write_text(json.dumps({'method': method, 'weight': weight, 'model': model,
                                                   'since': since, 'n_real': len(train), 'n_pseudo': len(extra),
                                                   'seconds': time.time()-t}, indent=2))
        w = warm(pred)
        print('DONE', out.name, len(train), len(extra), json.dumps(score(w.target, w.pred_cells)), flush=True)
    return tag
