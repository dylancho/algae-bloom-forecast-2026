"""Nutrient features from the NIER water-quality network snapshot (data_external/nier_wq).

Two slots per algae station: a near network site and an upstream/inflow site.
Availability rule: a sample counts only when its sampling date <= base_date - `lag_days`
(default 1, same as the bundle's daily drivers). `lag_days` is a parameter so the
publication-delay sensitivity (lab results appear weeks later) can be checked.
Lake sites sample several depths per day; values are the per-day median.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from algae.experiment import ROOT

SOURCE = ROOT/'data_external/nier_wq/wq_nakdong_geum.csv'
SLOTS = {'HP': ['칠곡', '선산'], 'GJGR': ['다사', '칠곡'], 'CS': ['남지', '현풍'],
         'MGMR': ['물금', '임해진'], 'DCHN': ['대청댐5', '옥천']}
VARS = ['tn', 'tp', 'po4p', 'no3n', 'nh3n', 'dtp', 'chla']


def load():
    df = pd.read_csv(SOURCE, parse_dates=['date'], encoding='utf-8-sig')
    sites = {s for v in SLOTS.values() for s in v}
    df = df[df.site.isin(sites)]
    cols = [c for c in VARS+['wtemp'] if c in df]
    g = df.groupby(['site', 'date'])[cols].median().sort_index()
    return {s: g.loc[s] for s in sites if s in g.index.get_level_values(0)}


def slot_features(t, base, lag_days):
    out = {}
    cut = base-pd.Timedelta(days=lag_days)
    past = t.loc[:cut]
    p = past.dropna(subset=['tp'])
    if len(p) == 0:
        return out
    last = p.iloc[-1]
    out['age_days'] = (base-p.index[-1]).days
    for v in ['tn', 'tp', 'po4p', 'no3n', 'nh3n', 'chla']:
        if v in p:
            s = past[v].dropna()
            out[f'{v}_log'] = np.log10(s.iloc[-1]+1e-3) if len(s) else np.nan
    out['tn_tp_log'] = np.log10((last.tn+1e-3)/(last.tp+1e-4)) if pd.notna(last.tn) else np.nan
    if len(p) >= 2:
        prev = p.iloc[-2]
        out['tp_change'] = np.log10(last.tp+1e-3)-np.log10(prev.tp+1e-3)
        out['tn_change'] = np.log10(last.tn+1e-3)-np.log10(prev.tn+1e-3) if pd.notna(prev.tn) and pd.notna(last.tn) else np.nan
    w = p.loc[cut-pd.Timedelta(days=45):]
    out['tp_45d_mean_log'] = np.log10(w.tp.mean()+1e-3) if len(w) else np.nan
    return out


def build(data, lag_days=1):
    tables = load()
    rows = []
    for site, base in zip(data.station_id, pd.to_datetime(data.base_date)):
        row = {}
        for k, name in enumerate(SLOTS[site]):
            if name in tables:
                for f, v in slot_features(tables[name], base, lag_days).items():
                    row[f'nu{k}_{f}'] = v
        rows.append(row)
    frame = pd.DataFrame(rows, index=data.index)
    return frame


def names(frame, slots=(0, 1)):
    return [c for c in frame.columns if int(c[2]) in slots]


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    d = pd.read_pickle(ROOT/'artifacts/features_v6/dataset.pkl')
    f = build(d)
    warm = d.base_date.dt.month.between(5, 10)
    print(f.shape)
    print(f[warm].notna().groupby(d.station_id[warm]).mean().round(2).T.to_string())
    print(f[warm].groupby(d.station_id[warm])[['nu0_age_days', 'nu1_age_days']].median())
