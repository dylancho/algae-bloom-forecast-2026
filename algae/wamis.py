"""River gauge features from the WAMIS snapshot (data_external/wamis).

Same availability rule as the bundle's daily drivers: only days <= base_date-1.
Water level datums changed over time at some gauges, so level enters only as
changes against its own recent past, never as an absolute value.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from algae.experiment import ROOT

FOLDER = ROOT/'data_external/wamis'
# two slots per station: (near/at the site, further upstream)
SLOTS = {'HP': ['2011640', '2011648'], 'GJGR': ['2014640', '2011660'],
         'CS': ['2020647', '2017620'], 'MGMR': ['2022610', '2020647'],
         'DCHN': ['3008630', '3004650']}


def load(code, kind):
    rows = []
    for f in sorted(FOLDER.glob(f'{code}_{kind}_*.json')):
        for r in json.loads(f.read_text(encoding='utf-8')).get('list', []):
            v = pd.to_numeric(r.get('wl' if kind == 'wl' else 'fw'), errors='coerce')
            rows.append((pd.Timestamp(r['ymd']), v))
    s = pd.Series(dict(rows), dtype=float).sort_index()
    if kind == 'flw':
        s = s.where(s >= 0)
    return s.asfreq('D')


def window(s, end, days):
    return s.loc[end-pd.Timedelta(days=days-1):end]


def gauge_features(flow, level, base):
    end = base-pd.Timedelta(days=1)
    out = {}
    lf = np.log10(flow.clip(lower=0)+1)
    for d in [1, 3, 7, 14, 30]:
        w = window(lf, end, d)
        out[f'flow_log_{d}d'] = w.mean() if w.notna().sum() >= max(1, d//2) else np.nan
    out['flow_ratio_3_30'] = out['flow_log_3d']-out['flow_log_30d']
    out['flow_ratio_7_30'] = out['flow_log_7d']-out['flow_log_30d']
    w7 = window(lf, end, 7)
    out['flow_max_7d'] = w7.max() if w7.notna().any() else np.nan
    out['flow_jump_7d'] = out['flow_max_7d']-out['flow_log_30d']
    lv14, lv3 = window(level, end, 14), window(level, end, 3)
    out['wl_3_minus_14'] = lv3.mean()-lv14.mean() if lv3.notna().any() and lv14.notna().sum() >= 7 else np.nan
    out['wl_range_7d'] = np.ptp(window(level, end, 7).dropna()) if window(level, end, 7).notna().sum() >= 3 else np.nan
    return out


def build(data):
    cache = {}
    for codes in SLOTS.values():
        for c in codes:
            if c not in cache:
                cache[c] = (load(c, 'flw'), load(c, 'wl'))
    rows = []
    for site, base in zip(data.station_id, pd.to_datetime(data.base_date)):
        row = {}
        for k, code in enumerate(SLOTS[site]):
            for name, v in gauge_features(*cache[code], base).items():
                row[f'g{k}_{name}'] = v
        rows.append(row)
    return pd.DataFrame(rows, index=data.index)


def coverage():
    for site, codes in SLOTS.items():
        for c in codes:
            f, w = load(c, 'flw'), load(c, 'wl')
            print(site, c, 'flow', f.first_valid_index(), f.last_valid_index(), f.notna().sum(),
                  '| wl', w.first_valid_index(), w.last_valid_index(), w.notna().sum())


if __name__ == '__main__':
    coverage()
