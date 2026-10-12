"""Version 2 feature extension. Version 1 modules stay frozen.

Adds thermal accumulation, water-quality trends, longer algae memory, and the
residence-time proxies the bundle README asks for (reservoir level combined with
inflow and release). Timing rules are identical to version 1: algae and water
quality up to base_date, daily weather and dam drivers up to base_date - 1 day.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from algae.features import (DAMS, GENERA, SITES, TARGET, WATER, build_features,
                            load_bundle, log_value)

ROOT = Path(__file__).resolve().parents[1]


def _mean(series):
    return float(series.mean()) if len(series) and series.notna().any() else np.nan


def _diff(a, b):
    return a - b if pd.notna(a) and pd.notna(b) else np.nan


def build_extra(tables, pairs):
    """New columns only. Same causality cutoffs as version 1."""
    a = tables['algae_measure'].copy()
    a['date'] = pd.to_datetime(a.chck_de)
    for col in [TARGET] + WATER + GENERA:
        a[col] = pd.to_numeric(a[col], errors='coerce')
    weather = tables['asos_daily'].copy()
    weather['date'] = pd.to_datetime(weather.tm)
    dams = tables['dam_daily'].copy()
    dams['date'] = pd.to_datetime(dams.obs_date)
    algae_groups = {k: g.sort_values('date').set_index('date') for k, g in a.groupby('station_id')}
    weather_groups = {k: g.sort_values('date').set_index('date') for k, g in weather.groupby('stnId')}
    dam_groups = {k: g.sort_values('date').set_index('date') for k, g in dams.groupby('dam_code')}
    stations = tables['stations'].set_index('station_id')
    rows = []
    for pair in pairs.itertuples(index=False):
        site, base = pair.station_id, pd.Timestamp(pair.base_date)
        cutoff = base - pd.Timedelta(days=1)
        own = algae_groups[site].loc[:base]
        valid = own[own[TARGET].notna()]
        f = {}

        # --- Algae memory beyond 28 days -------------------------------------
        for days in [56]:
            recent = valid.loc[base-pd.Timedelta(days=days-1):base, TARGET]
            logged = np.log10(1 + recent)
            f[f'cells_log_mean_{days}d'] = _mean(logged)
            f[f'cells_log_max_{days}d'] = float(logged.max()) if len(logged) else np.nan
            f[f'cells_count_{days}d'] = len(logged)
        year_window = valid.loc[base-pd.Timedelta(days=364):base, TARGET]
        f['cells_log_max_365d'] = log_value(year_window.max()) if len(year_window) else np.nan
        f['cells_log_mean_365d'] = _mean(np.log10(1 + year_window))
        recent28 = valid.loc[base-pd.Timedelta(days=27):base, TARGET]
        f['cells_over_caution_fraction_28d'] = float((recent28 > 10000).mean()) if len(recent28) else np.nan
        f['cells_log_vs_year_mean'] = _diff(log_value(valid[TARGET].iloc[-1]) if len(valid) else np.nan,
                                            f['cells_log_mean_365d'])

        # --- Water quality trends (v1 keeps only the latest reading) ---------
        for col, alias in [('iem_wtrtp', 'wtrtp'), ('iem_chla', 'chla'), ('iem_tur', 'tur')]:
            series = own[col].dropna()
            latest = float(series.iloc[-1]) if len(series) else np.nan
            for days in ([28, 56] if alias == 'wtrtp' else [28]):
                window = series.loc[base-pd.Timedelta(days=days-1):base]
                f[f'wq_{alias}_mean_{days}d'] = _mean(window)
            f[f'wq_{alias}_change_28d'] = _diff(latest, f[f'wq_{alias}_mean_28d'])
        f['wq_chla_log'] = log_value(own.iem_chla.dropna().iloc[-1]) if own.iem_chla.notna().any() else np.nan
        f['wq_obs_count_28d'] = int(own.loc[base-pd.Timedelta(days=27):base, 'iem_wtrtp'].notna().sum())

        # --- Weather: thermal accumulation and longer windows ----------------
        wx = weather_groups[int(stations.loc[site, 'asos_stn_id'])].loc[:cutoff]
        for days in [28]:
            window = wx.loc[cutoff-pd.Timedelta(days=days-1):cutoff]
            for col in ['avgTa', 'minTa', 'avgTs', 'avgRhm', 'sumSsHr']:
                f[f'wx_{col}_{days}d'] = _mean(window[col])
            f[f'wx_rain_observed_logsum_{days}d'] = log_value(window.sumRn.sum(min_count=1))
            f[f'wx_rain_observed_fraction_{days}d'] = window.sumRn.notna().sum()/days
            f[f'wx_temp_observed_fraction_{days}d'] = window.avgTa.notna().sum()/days
        for days, threshold in [(28, 20), (28, 25), (56, 20)]:
            window = wx.loc[cutoff-pd.Timedelta(days=days-1):cutoff, 'avgTa'].dropna()
            f[f'wx_gdd{threshold}_{days}d'] = float(np.clip(window-threshold, 0, None).sum()) if len(window) else np.nan
        hot = wx.loc[cutoff-pd.Timedelta(days=13):cutoff, 'maxTa'].dropna()
        f['wx_hot_days_fraction_14d'] = float((hot >= 30).mean()) if len(hot) else np.nan
        cloud = wx.loc[cutoff-pd.Timedelta(days=13):cutoff, 'avgTca']
        f['wx_avgTca_14d'] = _mean(cloud)
        f['wx_avgTca_observed_fraction_14d'] = cloud.notna().sum()/14
        warm14 = wx.loc[cutoff-pd.Timedelta(days=13):cutoff]
        f['wx_ts_minus_ta_14d'] = _diff(_mean(warm14.avgTs), _mean(warm14.avgTa))
        f['wx_temp_trend_28d'] = _diff(_mean(wx.loc[cutoff-pd.Timedelta(days=6):cutoff, 'avgTa']),
                                       f['wx_avgTa_28d'])

        # --- Hydrology: residence-time proxies per dam slot ------------------
        for slot, code in enumerate(DAMS[site]):
            hydro = dam_groups[code].loc[:cutoff]
            rwl = hydro.rwl.dropna()
            latest_rwl = float(rwl.iloc[-1]) if len(rwl) else np.nan
            year_rwl = _mean(rwl.loc[cutoff-pd.Timedelta(days=364):cutoff])
            f[f'dam{slot}_rwl_last'] = latest_rwl
            f[f'dam{slot}_rwl_anomaly_365d'] = _diff(latest_rwl, year_rwl)
            for days in [28]:
                window = hydro.loc[cutoff-pd.Timedelta(days=days-1):cutoff]
                for col in ['iqty', 'tdqty']:
                    f[f'dam{slot}_{col}_logmean_{days}d'] = log_value(_mean(window[col]))
            window14 = hydro.loc[cutoff-pd.Timedelta(days=13):cutoff]
            for col, alias in [('spdqty', 'spill'), ('itqty', 'intake')]:
                f[f'dam{slot}_{alias}_logmean_14d'] = log_value(_mean(window14[col]))
            release14 = log_value(_mean(window14.tdqty))
            inflow14 = log_value(_mean(window14.iqty))
            f[f'dam{slot}_release_ratio_14d'] = _diff(release14, inflow14)
            f[f'dam{slot}_tdqty_trend'] = _diff(log_value(_mean(hydro.loc[cutoff-pd.Timedelta(days=6):cutoff, 'tdqty'])),
                                                f[f'dam{slot}_tdqty_logmean_28d'])
            f[f'dam{slot}_residence_proxy_14d'] = _diff(f[f'dam{slot}_rwl_anomaly_365d'], release14)
            rain14 = window14.rf
            f[f'dam{slot}_rain_observed_logsum_14d'] = log_value(rain14.sum(min_count=1))
            f[f'dam{slot}_rain_observed_fraction_14d'] = rain14.notna().sum()/14
        rows.append(f)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def build_dataset_v2(bundle: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    tables = load_bundle(bundle)
    pairs = tables['eval_pairs'].copy()
    base_features, provenance = build_features(tables, pairs)
    extra = build_extra(tables, pairs)
    overlap = set(base_features.columns) & set(extra.columns)
    assert not overlap, f'v2 must not redefine v1 columns: {sorted(overlap)}'
    features = pd.concat([base_features, extra], axis=1)

    labels = pairs[pairs.split.isin(['train', 'valid'])][['station_id', 'base_date', 'target_date']].merge(
        tables['algae_measure'][['station_id', 'chck_de', TARGET]],
        left_on=['station_id', 'target_date'], right_on=['station_id', 'chck_de'], validate='many_to_one')
    metadata = pairs.merge(labels[['station_id', 'base_date', TARGET]], on=['station_id', 'base_date'],
                           how='left', validate='one_to_one').rename(columns={TARGET: 'target'})
    assert metadata.loc[metadata.split.isin(['test', 'excluded']), 'target'].isna().all()
    assert metadata.loc[metadata.split.isin(['train', 'valid']), 'target'].notna().all()
    assert (provenance.algae_latest_source <= provenance.base_date).all()
    assert (provenance.weather_latest_source < provenance.base_date).all()
    assert (provenance.dam_latest_source < provenance.base_date).all()
    pd.concat([metadata, features], axis=1).to_pickle(output/'dataset.pkl')
    provenance.to_csv(output/'provenance.csv', index=False)

    v1_manifest = json.loads((ROOT/'artifacts/features/manifest.json').read_text(encoding='utf-8'))
    groups = dict(v1_manifest['feature_groups'])
    new = list(extra.columns)
    groups['v2_algae'] = [c for c in new if c.startswith('cells_')]
    groups['v2_water'] = [c for c in new if c.startswith('wq_')]
    groups['v2_weather'] = [c for c in new if c.startswith('wx_')]
    groups['v2_hydro'] = [c for c in new if c.startswith('dam')]
    assert sum(len(groups[k]) for k in groups if k.startswith('v2_')) == len(new)
    manifest = {'features': list(features), 'n_features': len(features.columns),
                'n_features_v1': len(base_features.columns), 'n_features_new': len(new),
                'rows': len(pairs), 'version': 2,
                'bundle_sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(),
                'daily_lag_days': 1, 'dam_slots': DAMS,
                'rain_policy': v1_manifest['rain_policy'],
                'test_labels': 'Not loaded into supervised dataset.',
                'v1_features': list(base_features.columns), 'new_features': new,
                'feature_groups': groups}
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    result = build_dataset_v2(ROOT/'02_dataset_bundle_algae2026.zip', ROOT/'artifacts/features_v2')
    print('Built', result['rows'], 'rows,', result['n_features'], 'features',
          f"({result['n_features_v1']} v1 + {result['n_features_new']} new)")
