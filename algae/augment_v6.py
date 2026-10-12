"""Constrained raw-history mixup for rare REAL forecasting examples.

This is NOT the published SMOGN algorithm. Unlike independent feature-vector
interpolation, it mixes aligned raw historical blocks then recomputes features.
Admission to a fold requires BOTH parents' target dates before its boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from algae.experiment import ROOT, get_data
from algae.features import DAMS, GENERA, TARGET, WATER, build_features, load_bundle
from algae.features_v6 import OUT, biology_row, event_row, prepare
from algae.wamis import SLOTS, load as load_gauge


def mix_value(a, b, fraction, logged=False):
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    if logged:
        return 10**((1-fraction)*np.log10(max(a, 0)+1)+fraction*np.log10(max(b, 0)+1))-1
    return (1-fraction)*a+fraction*b


def build(limit=180):
    OUT.mkdir(exist_ok=True)
    data, _ = get_data()
    pool = data[data.split.eq('train') & data.base_date.dt.year.ge(2017) & data.base_cells_missing.eq(0)].copy()
    delta = np.log10(pool.target+1)-pool.cells_log_lag0
    pool['regime'] = np.where(delta < -.7, 0, np.where(delta > .7, 2, 1))
    rare = pool[(pool.target > 10000) | (abs(delta) > .7)]
    if len(rare) > limit:
        rare = rare.sample(limit, random_state=42).sort_index()
    tables = load_bundle(ROOT/'02_dataset_bundle_algae2026.zip')
    algae, weather = prepare(tables)
    station_wx = tables['stations'].set_index('station_id').asos_stn_id.to_dict()
    flows = {s: load_gauge(c[0], 'flw') for s, c in SLOTS.items()}
    rng = np.random.default_rng(42)
    rows = []
    for idx, anchor in rare.iterrows():
        cyc = abs(pool.base_date.dt.dayofyear-anchor.base_date.dayofyear)
        cyc = np.minimum(cyc, 366-cyc)
        eligible = pool[(pool.index != idx) & pool.station_id.eq(anchor.station_id)
                        & pool.regime.eq(anchor.regime) & pool.lag_days.eq(anchor.lag_days)
                        & (cyc <= 45) & (abs(pool.cells_log_lag0-anchor.cells_log_lag0) <= .7)]
        if eligible.empty:
            continue
        distance = abs(eligible.cells_log_lag0-anchor.cells_log_lag0)
        distance += abs(eligible.wq_iem_wtrtp-anchor.wq_iem_wtrtp).fillna(10)/10
        donor = eligible.loc[rng.choice(distance.nsmallest(3).index.to_numpy())]
        fraction = float(rng.uniform(.2, .5))
        site, base, other_base = anchor.station_id, anchor.base_date, donor.base_date
        offset = other_base-base
        own = algae[site].copy()
        source = algae[site]
        recent = own.index[(own.index >= base-pd.Timedelta(days=56)) & (own.index <= base)]
        for day in recent:
            other = source.loc[:day+offset]
            if other.empty or (day+offset-other.index[-1]).days > 14:
                continue
            counterpart = other.iloc[-1]
            for col in [TARGET]+WATER+GENERA:
                own.loc[day, col] = mix_value(own.loc[day, col], counterpart[col], fraction,
                                             logged=col in [TARGET, 'iem_chla']+GENERA)
            # Preserve genus composition nonnegativity and its total cell scale.
            species = own.loc[day, GENERA]
            total = own.loc[day, TARGET]
            if species.notna().all() and species.sum() > 0 and pd.notna(total):
                own.loc[day, GENERA] = species/species.sum()*total
        modified = dict(tables)
        a = tables['algae_measure'].copy()
        for col in [TARGET]+WATER+GENERA:
            a[col] = pd.to_numeric(a[col], errors='coerce').astype(float)
        actual_days = pd.to_datetime(a.chck_de)
        for day in recent:
            mask = a.station_id.eq(site) & actual_days.eq(day)
            a.loc[mask, [TARGET]+WATER+GENERA] = own.loc[day, [TARGET]+WATER+GENERA].to_numpy()
        modified['algae_measure'] = a
        # Both parent daily histories end at their own base-1. Missing drivers
        # remain missing if either parent is unknown.
        days = pd.date_range(base-pd.Timedelta(days=28), base-pd.Timedelta(days=1))
        wx = weather[site].copy()
        for col in wx.select_dtypes('number'):
            if col == 'stnId':
                continue
            left = weather[site][col].reindex(days).to_numpy(float)
            right = weather[site][col].reindex(days+offset).to_numpy(float)
            wx.loc[days, col] = (1-fraction)*left+fraction*right
        frame = tables['asos_daily'].copy()
        for col in [c for c in wx.select_dtypes('number') if c != 'stnId']:
            frame[col] = pd.to_numeric(frame[col], errors='coerce').astype(float)
        for day in days:
            mask = frame.stnId.eq(station_wx[site]) & frame.tm.eq(day.strftime('%Y-%m-%d'))
            cols = [c for c in wx.select_dtypes('number') if c != 'stnId']
            frame.loc[mask, cols] = wx.loc[day, cols].to_numpy()
        modified['asos_daily'] = frame
        dam = tables['dam_daily'].copy()
        for col in [c for c in dam.select_dtypes('number') if c != 'dam_code']:
            dam[col] = dam[col].astype(float)
        for code in DAMS[site]:
            source_dam = tables['dam_daily'][tables['dam_daily'].dam_code.eq(code)].copy()
            source_dam.index = pd.to_datetime(source_dam.obs_date)
            cols = [c for c in source_dam.select_dtypes('number') if c != 'dam_code']
            for day in days:
                if day not in source_dam.index or day+offset not in source_dam.index:
                    continue
                left, right = source_dam.loc[day, cols], source_dam.loc[day+offset, cols]
                mask = dam.dam_code.eq(code) & dam.obs_date.eq(day.strftime('%Y-%m-%d'))
                dam.loc[mask, cols] = ((1-fraction)*left+fraction*right).to_numpy()
        modified['dam_daily'] = dam
        flow = flows[site].copy()
        left = flows[site].reindex(days).to_numpy(float)
        right = flows[site].reindex(days+offset).to_numpy(float)
        flow = flow.reindex(flow.index.union(days)).sort_index()
        flow.loc[days] = (1-fraction)*left+fraction*right
        x, _ = build_features(modified, pd.DataFrame([{'station_id': site, 'base_date': base}]))
        row = anchor[['station_id', 'base_date', 'target_date', 'split']].to_dict()
        row.update(x.iloc[0].to_dict())
        row.update(event_row(own, wx, flow, base))
        row.update(biology_row(own, base))
        row.update(target=mix_value(anchor.target, donor.target, fraction, logged=True),
                   parent_index=idx, donor_index=donor.name,
                   info_end=max(anchor.target_date, donor.target_date), fraction=fraction)
        rows.append(row)
    output = pd.DataFrame(rows)
    output.to_pickle(OUT/'history_mixup.pkl')
    print('DONE raw-history mixup', len(output), flush=True)


if __name__ == '__main__':
    build()
