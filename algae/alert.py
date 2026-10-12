"""Version 9: algae-alert state and bloom-episode position (strategy R1).

MoE 2016 pp.42-44: an alert is issued after two consecutive samples over a level
(source-water levels 1,000 / 10,000 / 1,000,000 cells/mL); at 'caution' and above,
sampling goes to twice a week and dam/weir releases may be adjusted. In the bundle,
two consecutive samples >= 10,000 precede a <=4-day next sample 94% (2012-19) and
89% (2020-26) of the time, so one rule is used for all years.

Only cell counts measured on or before base_date and daily dam data up to base_date-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from algae.features import DAMS, TARGET

LEVELS = [1_000, 10_000, 1_000_000]
STATE = ['al_state', 'al_days_in_state', 'al_consec_over10k',
         'ep_duration_days', 'ep_n_samples', 'ep_max_log', 'ep_from_max', 'ep_days_since_max']
FLOW = ['al_release_anom3', 'al_inflow_anom3', 'al_release_x_alert']


def level(c):
    return int(sum(c >= t for t in LEVELS))


def _state_path(cells):
    """Alert state after each sample: raise/lower only on two consecutive samples."""
    states, s = [], 0
    for i, c in enumerate(cells):
        if i:
            pair = min(level(c), level(cells[i-1]))
            top = max(level(c), level(cells[i-1]))
            if pair > s:
                s = pair
            elif top < s:
                s = top
        states.append(s)
    return states


def state_features(valid, base):
    """valid: this station's non-missing counts indexed by date, all <= base."""
    f = dict.fromkeys(STATE, np.nan)
    if not len(valid):
        return f
    dates, cells = list(valid.index), valid[TARGET].to_list()
    path = _state_path(cells)
    s = path[-1]
    since = len(path)-1
    while since > 0 and path[since-1] == s:
        since -= 1
    f['al_state'] = s
    f['al_days_in_state'] = (base-dates[since]).days
    k = 0
    while k < len(cells) and cells[-1-k] >= LEVELS[1]:
        k += 1
    f['al_consec_over10k'] = k
    # episode: trailing run of samples >= 1,000
    e = 0
    while e < len(cells) and cells[-1-e] >= LEVELS[0]:
        e += 1
    f['ep_n_samples'] = e
    if e:
        ep = valid.iloc[-e:]
        logs = np.log10(ep[TARGET].to_numpy(float)+1)
        j = int(np.argmax(logs))
        f['ep_duration_days'] = (base-ep.index[0]).days
        f['ep_max_log'] = logs[j]
        f['ep_from_max'] = logs[-1]-logs[j]
        f['ep_days_since_max'] = (base-ep.index[j]).days
    else:
        f['ep_duration_days'] = 0
    return f


def flow_features(hydro, base, state):
    cut = base-pd.Timedelta(days=1)
    recent = hydro.loc[cut-pd.Timedelta(days=2):cut]
    ref = hydro.loc[cut-pd.Timedelta(days=32):cut-pd.Timedelta(days=3)]
    f = {}
    for col, name in [('tdqty', 'al_release_anom3'), ('iqty', 'al_inflow_anom3')]:
        a, b = recent[col].mean(), ref[col].mean()
        f[name] = np.log10(1+a)-np.log10(1+b) if len(recent) and len(ref) and a >= 0 and b >= 0 else np.nan
    f['al_release_x_alert'] = f['al_release_anom3']*float(state >= 2) if pd.notna(state) else np.nan
    return f


def build(tables, pairs):
    a = tables['algae_measure'].copy()
    a['date'] = pd.to_datetime(a.chck_de)
    a[TARGET] = pd.to_numeric(a[TARGET], errors='coerce')
    groups = {k: g.dropna(subset=[TARGET]).sort_values('date').set_index('date')[[TARGET]]
              for k, g in a.groupby('station_id')}
    d = tables['dam_daily'].copy()
    d['date'] = pd.to_datetime(d.obs_date)
    dams = {k: g.sort_values('date').set_index('date') for k, g in d.groupby('dam_code')}
    rows = []
    for p in pairs.itertuples(index=False):
        base = pd.Timestamp(p.base_date)
        f = state_features(groups[p.station_id].loc[:base], base)
        hydro = dams[DAMS[p.station_id][0]].loc[:base-pd.Timedelta(days=1)]
        f.update(flow_features(hydro, base, f['al_state']))
        rows.append(f)
    out = pd.DataFrame(rows, columns=STATE+FLOW).astype(float)
    keys = pairs[['station_id', 'base_date', 'target_date']].reset_index(drop=True).copy()
    keys['base_date'] = pd.to_datetime(keys.base_date)
    keys['target_date'] = pd.to_datetime(keys.target_date)
    return pd.concat([keys, out], axis=1)
