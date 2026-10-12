"""Version 8: base-date (same-day) daily drivers.

v1 uses daily weather/dam values up to base_date-1. The contest rule forbids only
observations *after* base_date, so day-d daily aggregates are admissible, although
about half of each 00-24 KST day falls after the ~11h sampling time.

- shifted_tables: relabel each daily row one day earlier, so the unchanged v1
  builder (cutoff = base-1) ends its windows on the real base_date.
- extras: a small set of day-d values alongside the untouched v1 features.
Nothing after base_date is read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from algae.features import DAMS, build_features

KEYS = ['station_id', 'base_date', 'target_date']
EXTRA = ['sd_rain_log', 'sd_minTa', 'sd_avgTa_vs_7d', 'sd_avgWs', 'sd_sumSsHr',
         'sd_dam0_rf_log', 'sd_dam0_iqty_jump', 'sd_dam0_tdqty_jump',
         'sd_dam1_rf_log', 'sd_dam1_iqty_jump', 'sd_dam1_tdqty_jump']
MINTA = ['sd_minTa']


def shifted_tables(tables):
    out = dict(tables)
    w = tables['asos_daily'].copy()
    w['tm'] = (pd.to_datetime(w.tm)-pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')
    d = tables['dam_daily'].copy()
    d['obs_date'] = (pd.to_datetime(d.obs_date)-pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')
    out['asos_daily'], out['dam_daily'] = w, d
    return out


def full_sameday(tables, pairs):
    """v1 feature set with every daily window ending on base_date."""
    x, _ = build_features(shifted_tables(tables), pairs)
    return x


def _log(v):
    return np.log10(1+v) if pd.notna(v) and v >= 0 else np.nan


def extras(tables, pairs):
    w = tables['asos_daily'].copy()
    w['date'] = pd.to_datetime(w.tm)
    d = tables['dam_daily'].copy()
    d['date'] = pd.to_datetime(d.obs_date)
    wg = {k: g.set_index('date').sort_index() for k, g in w.groupby('stnId')}
    dg = {k: g.set_index('date').sort_index() for k, g in d.groupby('dam_code')}
    stn = tables['stations'].set_index('station_id').asos_stn_id.astype(int)
    rows = []
    for p in pairs.itertuples(index=False):
        base = pd.Timestamp(p.base_date)
        prev7 = slice(base-pd.Timedelta(days=7), base-pd.Timedelta(days=1))
        f = {}
        wx = wg[stn[p.station_id]].loc[:base]
        today = wx.loc[base] if base in wx.index else None
        get = (lambda c: today[c] if today is not None else np.nan)
        f['sd_rain_log'] = _log(get('sumRn'))
        f['sd_minTa'] = get('minTa')
        f['sd_avgTa_vs_7d'] = get('avgTa')-wx.loc[prev7, 'avgTa'].mean()
        f['sd_avgWs'] = get('avgWs')
        f['sd_sumSsHr'] = get('sumSsHr')
        for slot, code in enumerate(DAMS[p.station_id]):
            h = dg[code].loc[:base]
            t = h.loc[base] if base in h.index else None
            f[f'sd_dam{slot}_rf_log'] = _log(t.rf) if t is not None else np.nan
            for col in ['iqty', 'tdqty']:
                past = _log(h.loc[prev7, col].mean())
                f[f'sd_dam{slot}_{col}_jump'] = (_log(t[col])-past) if t is not None else np.nan
        rows.append(f)
    return pd.DataFrame(rows, columns=EXTRA).astype(float)


SIGNED = ['sd_rain1_river', 'sd_rain1_lake', 'sd_rain3_river', 'sd_rain3_lake']
LAKE = {'DCHN'}


def signed(tables, pairs):
    """Base-day and 3-day (d-2..d) ASOS rain, split by river vs lake station.

    Literature (MoE 2016 p.20): storms flush rivers but feed Daecheong blooms, so a
    pooled rain feature cancels out. ASOS leaves sumRn blank on dry days; blank -> 0 here.
    """
    w = tables['asos_daily'].copy()
    w['date'] = pd.to_datetime(w.tm)
    w['rain'] = pd.to_numeric(w.sumRn, errors='coerce').fillna(0)
    wg = {k: g.set_index('date').rain.sort_index() for k, g in w.groupby('stnId')}
    stn = tables['stations'].set_index('station_id').asos_stn_id.astype(int)
    rows = []
    for p in pairs.itertuples(index=False):
        base = pd.Timestamp(p.base_date)
        r = wg[stn[p.station_id]].loc[:base]
        one = np.log10(1+r.get(base, 0.))
        three = np.log10(1+r.loc[base-pd.Timedelta(days=2):base].sum())
        lake = p.station_id in LAKE
        rows.append({'sd_rain1_river': 0. if lake else one, 'sd_rain1_lake': one if lake else 0.,
                     'sd_rain3_river': 0. if lake else three, 'sd_rain3_lake': three if lake else 0.})
    return pd.DataFrame(rows, columns=SIGNED).astype(float)


def build(tables, pairs):
    """Keyed frame: v1-names with prefix 'sdf_' (full same-day) plus extras."""
    full = full_sameday(tables, pairs).add_prefix('sdf_')
    keys = pairs[KEYS].reset_index(drop=True).copy()
    keys['base_date'] = pd.to_datetime(keys.base_date)
    keys['target_date'] = pd.to_datetime(keys.target_date)
    return pd.concat([keys, full.reset_index(drop=True), extras(tables, pairs), signed(tables, pairs)], axis=1)
