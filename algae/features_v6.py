"""Causal event, biology, thermal-history and latent-cell features.

Daily drivers end at t-1; cell observations may be available on t, matching v1.
Thermal regressions are refit at calendar-year boundaries using earlier years
only. The two Kalman filters have fixed, documented noise parameters.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from algae.experiment import ROOT, get_data
from algae.features import GENERA, TARGET, WATER, build_features, load_bundle
from algae.wamis import SLOTS, load as load_gauge

OUT = ROOT / 'artifacts/features_v6'
EVENTS = ['ev_flow_peak_age', 'ev_flow_after_peak', 'ev_flow_excess28',
          'ev_recovery_heat', 'ev_recovery_sun', 'ev_flow_coverage28',
          'ev_calm_run', 'ev_wind_event_age', 'ev_air_cooling3',
          'ev_high_observed_age', 'ev_high_observed_count']
BIO = ['bio_micro_share_change', 'bio_composition_turnover', 'bio_genus_switch',
       'bio_cells_chla_logratio_change', 'bio_ph_change', 'bio_do_change']
THERMAL = ['th_gdd20_7d', 'th_gdd25_14d', 'th_warm_run', 'th_cooling3',
           'th_sd_7d', 'th_last_innovation', 'th_growth_light7', 'th_fit_n_log']
STATE = ['st_level', 'st_trend', 'st_sd', 'st_innovation', 'st_forecast_delta7']
GROUPS = {'events': EVENTS, 'bio': BIO, 'thermal': THERMAL, 'state': STATE}


def prepare(tables):
    a = tables['algae_measure'].copy()
    a['date'] = pd.to_datetime(a.chck_de)
    for c in [TARGET] + GENERA + WATER:
        a[c] = pd.to_numeric(a[c], errors='coerce')
    wx = tables['asos_daily'].copy()
    wx['date'] = pd.to_datetime(wx.tm)
    sites = tables['stations'].set_index('station_id')
    algae = {s: g.sort_values('date').drop_duplicates('date').set_index('date')
             for s, g in a.groupby('station_id')}
    weather = {s: g.sort_values('date').drop_duplicates('date').set_index('date').asfreq('D')
               for s, g in wx.groupby('stnId')}
    return algae, {s: weather[int(sites.loc[s, 'asos_stn_id'])] for s in algae}


def run_length(s, threshold, above=True):
    n = 0
    for v in reversed(s.to_numpy(float)):
        if not np.isfinite(v) or not ((v >= threshold) if above else (v <= threshold)):
            break
        n += 1
    return n


def event_row(own, wx, flow, base):
    base = pd.Timestamp(base)
    end = base - pd.Timedelta(days=1)
    w = wx.loc[end-pd.Timedelta(days=27):end].reindex(pd.date_range(end-pd.Timedelta(days=27), end))
    q = flow.loc[:end]
    q28 = q.reindex(w.index)
    # Compare each flow with the median from its preceding 30 calendar days.
    med = q.shift(1).rolling(30, min_periods=15).median()
    anomaly = np.log10(q28+1)-np.log10(med.reindex(w.index)+1)
    strong = anomaly[anomaly > np.log10(2)]
    peak = strong.idxmax() if len(strong) else None
    age = (end-peak).days if peak is not None else 28
    after = w.loc[peak:] if peak is not None else w.iloc[0:0]
    f = {'ev_flow_peak_age': age,
         'ev_flow_after_peak': float(np.log10(q28.iloc[-1]+1)-np.log10(q28.loc[peak]+1)) if peak is not None else np.nan,
         'ev_flow_excess28': float(anomaly.clip(lower=0).sum(min_count=1)),
         'ev_recovery_heat': float((after.avgTa-20).clip(lower=0).sum(min_count=1)) if len(after) else 0.,
         'ev_recovery_sun': float(after.sumSsHr.sum(min_count=1)) if len(after) else 0.,
         'ev_flow_coverage28': q28.notna().mean(),
         'ev_calm_run': run_length(w.avgWs, 1.5, above=False),
         'ev_air_cooling3': w.avgTa.iloc[-3:].mean()-w.avgTa.iloc[-10:-3].mean()}
    windy = w.index[w.avgWs >= 3]
    f['ev_wind_event_age'] = (end-windy[-1]).days if len(windy) else 28
    valid = own.loc[:base, TARGET].dropna()
    n = 0
    for t, v in reversed(list(valid.items())):
        if v <= 10000 or (base-t).days > 56:
            break
        n += 1
    f['ev_high_observed_count'] = n
    f['ev_high_observed_age'] = (base-valid.index[-n]).days if n else 0
    return f


def biology_row(own, base):
    own = own.loc[:base]
    f = dict.fromkeys(BIO, np.nan)
    comp = own.dropna(subset=GENERA)
    comp = comp[comp[GENERA].sum(axis=1) > 0]
    if len(comp) >= 2 and (base-comp.index[-2]).days <= 28:
        shares = comp[GENERA].iloc[-2:].div(comp[GENERA].iloc[-2:].sum(axis=1), axis=0)
        f['bio_micro_share_change'] = float(shares.iloc[-1, 0]-shares.iloc[-2, 0])
        f['bio_composition_turnover'] = float(abs(shares.iloc[-1]-shares.iloc[-2]).sum()/2)
        f['bio_genus_switch'] = float(shares.iloc[-1].argmax() != shares.iloc[-2].argmax())
    joint = own.dropna(subset=[TARGET, 'iem_chla'])
    if len(joint) >= 2 and (base-joint.index[-2]).days <= 28:
        ratio = np.log10(joint[TARGET].iloc[-2:]+1)-np.log10(joint.iem_chla.iloc[-2:].clip(lower=0)+1)
        f['bio_cells_chla_logratio_change'] = float(ratio.iloc[-1]-ratio.iloc[-2])
    for col, alias in [('iem_ph', 'ph'), ('iem_doc', 'do')]:
        v = own[col].dropna()
        if len(v) >= 2 and (base-v.index[-2]).days <= 28:
            f[f'bio_{alias}_change'] = float(v.iloc[-1]-v.iloc[-2])
    return f


def build_static(tables, pairs, flows=None):
    algae, weather = prepare(tables)
    flows = flows if flows is not None else {s: load_gauge(SLOTS[s][0], 'flw') for s in algae}
    rows = []
    for p in pairs.itertuples():
        t = pd.Timestamp(p.base_date)
        rows.append({**event_row(algae[p.station_id], weather[p.station_id], flows[p.station_id], t),
                     **biology_row(algae[p.station_id], t)})
    return pd.DataFrame(rows, index=pairs.index).replace([np.inf, -np.inf], np.nan)


def thermal_filter(own, wx, end):
    """Annual forward regression + assimilated residual, never backwards fill.

    Fixed residual persistence .92, process variance .25 C^2/day, observation
    variance 1 C^2. Prediction at a water measurement is recorded BEFORE update.
    """
    end = pd.Timestamp(end)
    start = min(own.index.min(), wx.index.min())
    dates = pd.date_range(start, end)
    # State on day d is driven by daily weather through d-1.
    w = wx.reindex(dates).shift(1)
    x = pd.DataFrame({'air3': w.avgTa.ewm(span=3, adjust=False).mean(),
                      'air10': w.avgTa.ewm(span=10, adjust=False).mean(),
                      'sin': np.sin(2*np.pi*dates.dayofyear/365.25),
                      'cos': np.cos(2*np.pi*dates.dayofyear/365.25)}, index=dates)
    obs = own.iem_wtrtp.reindex(dates)
    result = pd.DataFrame(index=dates, columns=['mean', 'sd', 'innovation', 'fit_n'], dtype=float)
    checks = []
    residual, variance, last_innovation = 0., 4., np.nan
    for year in sorted(set(dates.year)):
        history = (dates.year < year) & obs.notna().to_numpy() & x.air10.notna().to_numpy()
        n = int(history.sum())
        future = dates.year == year
        if n >= 25:
            m = make_pipeline(SimpleImputer(strategy='median'), Ridge(alpha=20))
            m.fit(x.loc[history], obs.loc[history])
            equilibrium = m.predict(x.loc[future])
        else:
            # Explicit untrained fallback, not an estimate from later years.
            equilibrium = (x.loc[future, 'air10'].fillna(10)+3).clip(0, 35).to_numpy()
        for day, mu in zip(dates[future], equilibrium):
            residual *= .92
            variance = .92**2*variance+.25
            prior, prior_sd = mu+residual, np.sqrt(variance+1.)
            value = obs.loc[day]
            if pd.notna(value):
                innovation = float(value-prior)
                if n >= 25:
                    checks.append({'date': day, 'actual': value, 'pred': prior,
                                   'sd': prior_sd, 'fit_n': n})
                gain = variance/(variance+1.)
                residual += gain*innovation
                variance *= 1-gain
                last_innovation = innovation
            result.loc[day] = [np.clip(mu+residual, 0, 40), np.sqrt(variance), last_innovation, n]
    result['sun'] = wx.reindex(dates).sumSsHr
    return result, pd.DataFrame(checks)


def cell_filter(own, end):
    """Fixed local linear filter, log10 units; .16 observation variance.

    Irregular observation intervals increase process uncertainty; assimilates
    only observations on or before the requested base date.
    """
    v = own.loc[:end, TARGET].dropna().clip(lower=0)
    out = []
    state, cov, previous = np.zeros(2), np.diag([1., .01]), None
    for t, value in v.items():
        dt = (t-previous).days if previous is not None else 0
        transition = np.array([[1., dt], [0., .95**dt]])
        state = transition@state
        cov = transition@cov@transition.T + np.diag([.025*dt, .0002*dt])
        innovation = float(np.log10(value+1)-state[0])
        gain = cov[:, 0]/(cov[0, 0]+.16)
        state += gain*innovation
        cov = cov-np.outer(gain, cov[0].copy())
        out.append({'date': t, 'st_level': state[0], 'st_trend': state[1],
                    'st_sd': np.sqrt(max(cov[0, 0], 0)), 'st_innovation': innovation,
                    'st_forecast_delta7': 7*state[1]})
        previous = t
    return pd.DataFrame(out).set_index('date') if out else pd.DataFrame(columns=STATE)


def build_dynamic(tables, pairs):
    algae, weather = prepare(tables)
    cache, checks = {}, []
    for site, g in pairs.groupby('station_id'):
        end = pd.to_datetime(g.base_date).max()
        thermal, audit = thermal_filter(algae[site], weather[site], end-pd.Timedelta(days=1))
        if len(audit):
            checks.append(audit.assign(station_id=site))
        cache[site] = thermal, cell_filter(algae[site], end)
    rows = []
    for p in pairs.itertuples():
        base = pd.Timestamp(p.base_date)
        end = base-pd.Timedelta(days=1)
        th, st = cache[p.station_id]
        w = th.loc[end-pd.Timedelta(days=13):end]
        last = w.iloc[-1] if len(w) else pd.Series(dtype=float)
        f = {'th_gdd20_7d': float((w['mean'].iloc[-7:]-20).clip(lower=0).sum(min_count=1)),
             'th_gdd25_14d': float((w['mean']-25).clip(lower=0).sum(min_count=1)),
             'th_warm_run': run_length(w['mean'], 25),
             'th_cooling3': w['mean'].iloc[-3:].mean()-w['mean'].iloc[-10:-3].mean(),
             'th_sd_7d': w.sd.iloc[-7:].mean(),
             'th_last_innovation': last.get('innovation', np.nan),
             'th_growth_light7': float(((w['mean'].iloc[-7:]-20).clip(lower=0)*w.sun.iloc[-7:]/12).sum(min_count=1)),
             'th_fit_n_log': np.log10(last.get('fit_n', 0)+1)}
        past = st.loc[:base]
        f.update(past.iloc[-1].to_dict() if len(past) else dict.fromkeys(STATE, np.nan))
        rows.append(f)
    return pd.DataFrame(rows, index=pairs.index), pd.concat(checks, ignore_index=True) if checks else pd.DataFrame()


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    data, names = get_data()
    tables = load_bundle(ROOT/'02_dataset_bundle_algae2026.zip')
    print('BUILD static', flush=True)
    static = build_static(tables, data)
    print('BUILD dynamic', flush=True)
    dynamic, audit = build_dynamic(tables, data)
    extra = static.join(dynamic)
    assert not set(names) & set(extra)
    data.join(extra).to_pickle(OUT/'dataset.pkl')
    audit.to_csv(OUT/'thermal_forward_checks.csv', index=False)
    manifest = {'v1_features': names, 'feature_groups': GROUPS,
                'bundle_sha256': hashlib.sha256((ROOT/'02_dataset_bundle_algae2026.zip').read_bytes()).hexdigest(),
                'timing': 'algae<=base; daily drivers<base; thermal fits use preceding calendar years',
                'rain': 'not zero-imputed; no rainfall-based event classification',
                'note': 'Thermal model is dynamic ridge+residual Kalman, not original Air2Water. Cell filter uses fixed priors.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE features', len(data), len(names), '+', len(extra.columns), flush=True)


def mask_history():
    """Remove the preceding actual sample for selected REAL training base dates.

    Current observation and actual target remain intact. Recompute ALL v1/event/
    biology features from raw data, so lag-derived variables remain consistent.
    This variant is evaluated with v1+static features only, no dynamic filters.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    data, names = get_data()
    tables = load_bundle(ROOT/'02_dataset_bundle_algae2026.zip')
    algae, weather = prepare(tables)
    flows = {s: load_gauge(SLOTS[s][0], 'flw') for s in algae}
    pool = data[data.split.eq('train') & data.base_date.dt.year.ge(2017)].sort_values(['station_id', 'base_date'])
    rows = []
    for n, (idx, p) in enumerate(pool.iterrows()):
        if n % 5 or p.base_cells_missing:
            continue
        own = algae[p.station_id]
        before = own.loc[own.index < p.base_date]
        if len(before) < 3:
            continue
        removed = before.index[-1]
        modified = dict(tables)
        a = tables['algae_measure']
        modified['algae_measure'] = a[~(a.station_id.eq(p.station_id) & a.chck_de.eq(removed.strftime('%Y-%m-%d')))]
        one = pd.DataFrame([{'station_id': p.station_id, 'base_date': p.base_date}])
        x, _ = build_features(modified, one)
        reduced = own.drop(removed)
        f = {**event_row(reduced, weather[p.station_id], flows[p.station_id], p.base_date),
             **biology_row(reduced, p.base_date)}
        row = p[['station_id', 'base_date', 'target_date', 'target', 'split']].to_dict()
        row.update(x.iloc[0].to_dict())
        row.update(f)
        row.update(parent_index=idx, removed_date=removed)
        rows.append(row)
    pd.DataFrame(rows).to_pickle(OUT/'history_masked.pkl')
    print('DONE masked real pairs', len(rows), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['build', 'mask'], default='build', nargs='?')
    args = parser.parse_args()
    build() if args.command == 'build' else mask_history()
