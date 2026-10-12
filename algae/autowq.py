"""Daily features from the NIER automatic water-quality network snapshot (data_external/nier_auto).

Two slots per algae station, chosen so that the item is still measured in 2026:
  bio  - nearest site with a working Chl-a sensor in 2026 (daily biomass signal)
  phys - nearest site for pH / DO / turbidity / EC / TOC (photosynthesis and dilution)
Only days <= base_date - 1 are used (same rule as the bundle's daily drivers).
Confirmed files are used where present; the unconfirmed snapshot fills the latest months.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from algae.experiment import ROOT

FOLDER = ROOT/'data_external/nier_auto'
BIO = {'HP': 'S02015', 'GJGR': 'S02004', 'CS': 'S02005', 'MGMR': 'S02017', 'DCHN': 'S03003'}
PHYS = {'HP': 'S02012', 'GJGR': 'S02004', 'CS': 'S02007', 'MGMR': 'S02017', 'DCHN': 'S03003'}
ITEMS = {'IEM_WTRTP': 'wt', 'IEM_PH': 'ph', 'IEM_DOC': 'do', 'IEM_EC': 'ec', 'IEM_TUR': 'tur',
         'IEM_TOC': 'toc', 'IEM_TN': 'tn', 'IEM_TP': 'tp', 'IEM_CHLA': 'chla'}


def load(code):
    frames = []
    for f in sorted(FOLDER.glob(f'{code}_*.json')):
        d = pd.DataFrame(json.loads(f.read_text(encoding='utf-8')))
        if d.empty:
            continue
        d['date'] = pd.to_datetime(d.MESURE_DT.str[:10])
        d['confirmed'] = not f.stem.endswith('recent')
        keep = ['date', 'confirmed']+[c for c in ITEMS if c in d]
        frames.append(d[keep])
    d = pd.concat(frames).rename(columns=ITEMS)
    for c in ITEMS.values():
        if c in d:
            d[c] = pd.to_numeric(d[c], errors='coerce')
        else:
            d[c] = np.nan
    vals = list(ITEMS.values())
    d = d[d[vals].notna().any(axis=1)]
    # prefer a confirmed value for a day; unconfirmed only fills days without one
    d = d.sort_values(['date', 'confirmed']).drop_duplicates('date', keep='last')
    return d.set_index('date')[vals].asfreq('D')


def window(s, end, days):
    return s.loc[end-pd.Timedelta(days=days-1):end]


def mean_if(s, need):
    return s.mean() if s.notna().sum() >= need else np.nan


def bio_features(t, base):
    end = base-pd.Timedelta(days=1)
    out = {}
    c = np.log10(t.chla.clip(lower=0)+1)
    valid = c.loc[:end].dropna()
    if len(valid):
        out['chla_age'] = (base-valid.index[-1]).days
        out['chla_last'] = valid.iloc[-1] if out['chla_age'] <= 7 else np.nan
    out['chla_3d'] = mean_if(window(c, end, 3), 1)
    out['chla_7d'] = mean_if(window(c, end, 7), 3)
    out['chla_max7'] = window(c, end, 7).max()
    early = mean_if(window(c, end-pd.Timedelta(days=4), 4), 2)
    out['chla_trend'] = out['chla_3d']-early
    out['chla_vs_30d'] = out['chla_7d']-mean_if(window(c, end, 30), 10)
    out['tp_7d'] = mean_if(np.log10(window(t.tp, end, 7)+1e-3), 3)
    out['tn_7d'] = mean_if(np.log10(window(t.tn, end, 7)+1e-3), 3)
    return out


def phys_features(t, base):
    end = base-pd.Timedelta(days=1)
    out = {}
    out['ph_3d'] = mean_if(window(t.ph, end, 3), 1)
    out['ph_max7'] = window(t.ph, end, 7).max()
    out['ph_trend'] = out['ph_3d']-mean_if(window(t.ph, end-pd.Timedelta(days=4), 7), 3)
    out['do_3d'] = mean_if(window(t.do, end, 3), 1)
    out['do_trend'] = out['do_3d']-mean_if(window(t.do, end-pd.Timedelta(days=4), 7), 3)
    out['tur_3d'] = mean_if(np.log10(window(t.tur, end, 3).clip(lower=0)+.1), 1)
    out['ec_ratio'] = mean_if(window(t.ec, end, 3), 1)/mean_if(window(t.ec, end, 21), 7)
    out['toc_7d'] = mean_if(window(t.toc, end, 7), 3)
    out['wt_3d'] = mean_if(window(t.wt, end, 3), 1)
    return out


def build(data):
    cache = {c: load(c) for c in set(BIO.values()) | set(PHYS.values())}
    rows = []
    for site, base in zip(data.station_id, pd.to_datetime(data.base_date)):
        row = {f'au_bio_{k}': v for k, v in bio_features(cache[BIO[site]], base).items()}
        row.update({f'au_phy_{k}': v for k, v in phys_features(cache[PHYS[site]], base).items()})
        rows.append(row)
    return pd.DataFrame(rows, index=data.index)


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    d = pd.read_pickle(ROOT/'artifacts/features_v6/dataset.pkl')
    f = build(d)
    warm = d.base_date.dt.month.between(5, 10)
    year = d.base_date.dt.year
    print(f.shape)
    print(f[warm].notna().groupby([d.station_id[warm]]).mean().round(2).T.to_string())
    print(f[warm].notna().groupby(year[warm]).mean().round(2)[['au_bio_chla_3d', 'au_phy_ph_3d', 'au_bio_tp_7d']].T.to_string())
