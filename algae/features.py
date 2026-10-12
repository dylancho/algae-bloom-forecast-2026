from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = 'iem_bgalage_cell_co'
WATER = ['iem_wtrtp', 'iem_ph', 'iem_doc', 'iem_trp', 'iem_tur', 'iem_chla']
GENERA = ['iem_bgalage_microsts', 'iem_bgalage_anba', 'iem_bgalage_osrtria', 'iem_bgalage_apzo']
# Separate slots, never summed: Daecheong/Yongdam are serial reservoirs.
# Conservative common upstream signals for Nakdong. Not a travel-time model.
DAMS = {'HP': [2001110, 2002110], 'GJGR': [2001110, 2002110],
        'CS': [2001110, 2002110], 'MGMR': [2001110, 2002110],
        'DCHN': [3008110, 3001110]}
SITES = ['HP', 'GJGR', 'CS', 'MGMR', 'DCHN']


def load_bundle(path: Path) -> dict[str, pd.DataFrame]:
    with zipfile.ZipFile(path) as bundle:
        return {Path(n).stem: pd.read_csv(io.BytesIO(bundle.read(n)))
                for n in bundle.namelist() if n.endswith('.csv')}


def log_value(value):
    return float(np.log10(1 + value)) if pd.notna(value) and value >= 0 else np.nan


def build_features(tables, pairs=None):
    """Only input timestamps <= base_date (daily drivers <= base_date-1d).

    Target_date, lag_days, split, and all target labels are excluded from features.
    Missing rainfall stays unknown; observed sums and observed fractions are distinct.
    """
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
    pairs = tables['eval_pairs'] if pairs is None else pairs
    rows, provenance = [], []
    for pair in pairs.itertuples(index=False):
        site, base = pair.station_id, pd.Timestamp(pair.base_date)
        cutoff = base - pd.Timedelta(days=1)
        own = algae_groups[site].loc[:base]
        valid = own[own[TARGET].notna()]
        f = {'station': SITES.index(site), 'doy_sin': np.sin(2 * np.pi * base.dayofyear / 365.25),
             'doy_cos': np.cos(2 * np.pi * base.dayofyear / 365.25)}
        f['base_cells_missing'] = float(base not in own.index or pd.isna(own.loc[base, TARGET]))
        for lag in range(3):
            f[f'cells_log_lag{lag}'] = log_value(valid[TARGET].iloc[-1-lag]) if len(valid) > lag else np.nan
            f[f'cells_age_lag{lag}'] = (base - valid.index[-1-lag]).days if len(valid) > lag else np.nan
        f['growth_per_day'] = ((f['cells_log_lag0'] - f['cells_log_lag1']) /
                               (f['cells_age_lag1'] - f['cells_age_lag0'])) if len(valid) >= 2 else np.nan
        for horizon in [7, 14, 28]:
            past = valid.loc[:base-pd.Timedelta(days=horizon)]
            f[f'cells_log_at_{horizon}d'] = log_value(past[TARGET].iloc[-1]) if len(past) else np.nan
            f[f'cells_age_at_{horizon}d'] = (base-past.index[-1]).days if len(past) else np.nan
        for days in [14, 28]:
            recent = valid.loc[base-pd.Timedelta(days=days-1):base, TARGET]
            logged = np.log10(1 + recent)
            for stat in ['mean', 'max', 'std']:
                f[f'cells_log_{stat}_{days}d'] = float(getattr(logged, stat)()) if len(logged) else np.nan
            f[f'cells_count_{days}d'] = len(logged)
        for col in WATER:
            values = own[col].dropna()
            f[f'wq_{col}'] = values.iloc[-1] if len(values) else np.nan
            f[f'wq_{col}_age'] = (base-values.index[-1]).days if len(values) else np.nan
        complete_genera = own.dropna(subset=GENERA)
        species = complete_genera.iloc[-1][GENERA] if len(complete_genera) else None
        total = species.sum() if species is not None else np.nan
        f['genus_age'] = (base-complete_genera.index[-1]).days if len(complete_genera) else np.nan
        f['genus_zero'] = float(total == 0) if pd.notna(total) else np.nan
        for i, col in enumerate(GENERA):
            f[f'genus_share_{i}'] = float(species[col]/total) if pd.notna(total) and total > 0 else np.nan
        wx = weather_groups[int(stations.loc[site, 'asos_stn_id'])].loc[:cutoff]
        for days in [7, 14]:
            window = wx.loc[cutoff-pd.Timedelta(days=days-1):cutoff]
            for col in ['avgTa', 'maxTa', 'avgWs', 'avgRhm', 'sumSsHr']:
                f[f'wx_{col}_{days}d'] = window[col].mean()
            f[f'wx_rain_observed_logsum_{days}d'] = log_value(window.sumRn.sum(min_count=1))
            f[f'wx_rain_observed_fraction_{days}d'] = window.sumRn.notna().sum()/days
            f[f'wx_temp_observed_fraction_{days}d'] = window.avgTa.notna().sum()/days
        f['wx_temp_recent_change'] = f['wx_avgTa_7d'] - f['wx_avgTa_14d']
        dam_dates = []
        for slot, code in enumerate(DAMS[site]):
            hydro = dam_groups[code].loc[:cutoff]
            dam_dates.append(hydro.index.max())
            for days in [7, 14]:
                window = hydro.loc[cutoff-pd.Timedelta(days=days-1):cutoff]
                for col in ['iqty', 'tdqty']:
                    f[f'dam{slot}_{col}_logmean_{days}d'] = log_value(window[col].mean())
            window = hydro.loc[cutoff-pd.Timedelta(days=6):cutoff]
            f[f'dam{slot}_rwl_change_7d'] = window.rwl.iloc[-1]-window.rwl.iloc[0] if len(window) >= 2 else np.nan
            f[f'dam{slot}_observed_fraction_7d'] = len(window)/7
            f[f'dam{slot}_rain_observed_logsum_7d'] = log_value(window.rf.sum(min_count=1))
            f[f'dam{slot}_rain_observed_fraction_7d'] = window.rf.notna().sum()/7
        rows.append(f)
        provenance.append({'station_id': site, 'base_date': base,
                           'algae_latest_source': own.index.max(),
                           'cells_latest_source': valid.index.max(),
                           'weather_latest_source': wx.index.max(),
                           'dam_latest_source': max(dam_dates)})
    result = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    return result, pd.DataFrame(provenance)


def build_dataset(bundle, output):
    output.mkdir(parents=True, exist_ok=True)
    tables = load_bundle(bundle)
    pairs = tables['eval_pairs'].copy()
    features, provenance = build_features(tables, pairs)
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
    manifest = {'features': list(features), 'n_features': len(features.columns), 'rows': len(pairs),
                'bundle_sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(),
                'daily_lag_days': 1, 'dam_slots': DAMS,
                'rain_policy': 'Observed sums plus observed fractions; unknown is not imputed to zero.',
                'test_labels': 'Not loaded into supervised dataset.',
                'feature_groups': {
                    'algae': [c for c in features if not c.startswith(('wq_', 'genus_', 'wx_', 'dam'))],
                    'water': [c for c in features if c.startswith(('wq_', 'genus_'))],
                    'weather': [c for c in features if c.startswith('wx_')],
                    'hydro': [c for c in features if c.startswith('dam')]}}
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest
