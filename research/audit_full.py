"""Audit the supplied full bundle and a fixed persistence baseline.

No test/excluded targets are joined or scored. Persistence uses only available
same-station observations on/before each base date; there is no parameter fit.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'iem_bgalage_cell_co'


def records(frame):
    return json.loads(frame.to_json(orient='records', date_format='iso', force_ascii=False))


def metrics(frame):
    actual = frame['target'].to_numpy(float)
    pred = frame['prediction'].to_numpy(float)
    err = np.log10(pred + 1) - np.log10(actual + 1)
    denom = ((actual - actual.mean()) ** 2).sum()
    return {
        'n': len(frame), 'rmse_log10': float(np.sqrt(np.mean(err ** 2))),
        'mae_log10': float(np.mean(np.abs(err))),
        'r2_nse': float(1 - ((actual - pred) ** 2).sum() / denom) if denom else None,
    }


def main():
    with zipfile.ZipFile(ROOT / '02_dataset_bundle_algae2026.zip') as bundle:
        frames = {Path(n).name: pd.read_csv(io.BytesIO(bundle.read(n)))
                  for n in bundle.namelist() if n.endswith('.csv')}
        readme_name = next(n for n in bundle.namelist() if n.endswith('/README.md'))
        (ROOT / 'research' / 'FULL_DATA_README.md').write_text(
            bundle.read(readme_name).decode('utf-8-sig').replace('\r\n', '\n').replace('\r', ''),
            encoding='utf-8')
    a = frames['algae_measure.csv'].copy()
    a['date'] = pd.to_datetime(a.chck_de)
    p = frames['eval_pairs.csv'].copy()
    p['base_date'] = pd.to_datetime(p.base_date)
    p['target_date'] = pd.to_datetime(p.target_date)
    historical = a[a.date < '2026-01-01'].copy()
    historical['year'] = historical.date.dt.year
    summary = {'tables': {}, 'scope': 'Train/valid diagnostics; no test-target scoring or model tuning.'}
    for name, frame in frames.items():
        summary['tables'][name] = {'rows': len(frame), 'columns': list(frame),
                                  'missing_counts': frame.isna().sum().to_dict()}
    summary['split'] = records(p.groupby('split').agg(
        n=('station_id', 'size'), base_start=('base_date', 'min'), base_end=('base_date', 'max'),
        target_end=('target_date', 'max')).reset_index())
    summary['lag_days'] = pd.crosstab(p.split, p.lag_days).to_dict()
    summary['station_year_counts'] = records(pd.crosstab(historical.year, historical.station_id).reset_index())
    summary['historical_missing_percent_by_year'] = records(
        historical.groupby('year')[[TARGET, 'iem_wtrtp', 'iem_chla', 'iem_bgalage_microsts', 'iem_microstlr']]
        .agg(lambda s: 100 * s.isna().mean()).reset_index())
    summary['historical_nonnumeric_tokens'] = {}
    for col in historical.filter(regex='^iem_'):
        bad = historical[col].notna() & pd.to_numeric(historical[col], errors='coerce').isna()
        if bad.any():
            summary['historical_nonnumeric_tokens'][col] = historical.loc[bad, col].value_counts().to_dict()
    genera = ['iem_bgalage_microsts', 'iem_bgalage_anba', 'iem_bgalage_osrtria', 'iem_bgalage_apzo']
    complete = historical[genera + [TARGET]].notna().all(axis=1)
    summary['historical_genus_sum_mismatches'] = int((historical.loc[complete, genera].sum(axis=1) != historical.loc[complete, TARGET]).sum())
    summary['historical_dominant_missing_with_zero_target'] = int((historical.dominant_genus.isna() & historical[TARGET].eq(0)).sum())
    summary['historical_dominant_missing_with_positive_target'] = int((historical.dominant_genus.isna() & historical[TARGET].gt(0)).sum())
    summary['duplicate_base_keys'] = int(p.duplicated(['station_id', 'base_date']).sum())
    summary['duplicate_algae_keys'] = int(a.duplicated(['station_id', 'chck_de']).sum())
    summary['duplicate_target_keys_by_split'] = {
        str(k): int(g.duplicated(['station_id', 'target_date']).sum()) for k, g in p.groupby('split')}
    submission = frames['sample_submission.csv'].copy()
    for col in ['base_date', 'target_date']:
        submission[col] = pd.to_datetime(submission[col])
    keys = ['station_id', 'base_date', 'target_date']
    summary['submission_matches_test_order'] = p.loc[p.split.eq('test'), keys].reset_index(drop=True).equals(submission[keys])

    evaluated = p[p.split.isin(['train', 'valid'])].copy()
    # Restrict label retrieval to exactly the declared train/valid target keys.
    labels = evaluated[['station_id', 'target_date']].drop_duplicates().merge(
        a[['station_id', 'date', TARGET]], left_on=['station_id', 'target_date'],
        right_on=['station_id', 'date'], validate='one_to_one').rename(columns={TARGET: 'target'})
    evaluated = evaluated.merge(labels[['station_id', 'target_date', 'target']], on=['station_id', 'target_date'], validate='many_to_one')
    base_values = a[['station_id', 'date', TARGET]].rename(columns={'date': 'base_date', TARGET: 'base_value'})
    evaluated = evaluated.merge(base_values, on=['station_id', 'base_date'], validate='one_to_one')
    chunks = []
    for station, group in evaluated.groupby('station_id'):
        available = a[(a.station_id == station) & a[TARGET].notna()]
        available = available[['date', TARGET]].sort_values('date').rename(columns={TARGET: 'prediction', 'date': 'source_date'})
        matched = pd.merge_asof(group.sort_values('base_date'), available,
                               left_on='base_date', right_on='source_date', direction='backward')
        chunks.append(matched)
    evaluated = pd.concat(chunks, ignore_index=True)
    assert evaluated.target.notna().all(), 'Missing official train/valid target'
    assert evaluated.prediction.notna().all(), 'Missing baseline source'
    assert (evaluated.source_date <= evaluated.base_date).all(), 'Future observation in persistence'
    evaluated['age_days'] = (evaluated.base_date - evaluated.source_date).dt.days
    evaluated['year'] = evaluated.base_date.dt.year
    evaluated['season'] = evaluated.base_date.dt.month.between(5, 10)
    evaluated['zero'] = evaluated.target.eq(0)
    evaluated['high'] = evaluated.target.ge(10000)
    evaluated['base_missing'] = evaluated.base_value.isna()
    summary['target_stats_by_split_station'] = records(evaluated.groupby(['split', 'station_id']).agg(
        n=('target', 'size'), zero_fraction=('zero', 'mean'), high_fraction=('high', 'mean'),
        median=('target', 'median'), max=('target', 'max'), missing_base=('base_missing', 'sum')).reset_index())
    summary['target_stats_by_season'] = records(evaluated.groupby(['split', 'season']).agg(
        n=('target', 'size'), zero_fraction=('zero', 'mean'), high_fraction=('high', 'mean')).reset_index())
    summary['train_crossing_2025'] = records(p[p.split.eq('train') & p.target_date.ge('2025-01-01')][keys])
    summary['valid_crossing_2026'] = records(p[p.split.eq('valid') & p.target_date.ge('2026-01-01')][keys])
    summary['persistence'] = []
    for year in range(2021, 2026):
        for label, mask in [('all', evaluated.year.eq(year)),
                            ('May_Oct', evaluated.year.eq(year) & evaluated.season)]:
            group = evaluated[mask]
            summary['persistence'].append({'year': year, 'window': label, **metrics(group)})
    summary['persistence_2025_season_by_station'] = [
        {'station_id': station, **metrics(group)}
        for station, group in evaluated[evaluated.year.eq(2025) & evaluated.season].groupby('station_id')]
    summary['missing_base_age_days'] = records(evaluated[evaluated.base_missing].groupby('split').age_days.agg(['count', 'median', 'max']).reset_index())
    warm_valid = evaluated[evaluated.year.eq(2025) & evaluated.season].copy()
    raw_squared_error = (warm_valid.target - warm_valid.prediction) ** 2
    summary['persistence_2025_season_top5_raw_squared_error_fraction'] = float(raw_squared_error.nlargest(5).sum() / raw_squared_error.sum())
    evaluated.to_csv(ROOT / 'research' / 'persistence_train_valid.csv', index=False, encoding='utf-8-sig')

    weather = frames['asos_daily.csv'].copy()
    weather['year'] = pd.to_datetime(weather.tm).dt.year
    summary['radiation_nonmissing_fraction_by_year'] = records(pd.crosstab(
        weather.year, weather.stnNm, values=weather.sumGsr.notna().astype(int), aggfunc='mean').reset_index())
    summary['weather_quality_flags_present'] = [c for c in weather if 'flag' in c.lower() or 'qc' in c.lower()]
    summary['radiation_by_station'] = records(weather.groupby('stnNm').sumGsr.agg(
        n='size', observed='count', missing_fraction=lambda s: s.isna().mean()).reset_index())
    dams = frames['dam_daily.csv']
    summary['dam_duplicate_keys'] = int(dams.duplicated(['dam_code', 'obs_date']).sum())
    summary['dam_constant_numeric_columns'] = {
        str(k): [c for c in ['rwl', 'iqty', 'tdqty', 'edqty', 'spdqty', 'otltdqty', 'itqty', 'rf'] if g[c].nunique(dropna=True) <= 1]
        for k, g in dams.groupby('dam_name')}
    summary['dam_coverage'] = records(dams.groupby('dam_name').obs_date.agg(['min', 'max', 'count']).reset_index())
    operations = frames['dam_operation_api.csv']
    summary['operations_date_range'] = [operations.obs_date.min(), operations.obs_date.max()]
    summary['operations_days'] = int(operations.obs_date.nunique())
    # Matching diagnostics only on train/valid. Do not reconstruct test labels.
    nearest_mismatches, nearest_ties = [], 0
    for row in evaluated.itertuples():
        dates = a.loc[a.station_id.eq(row.station_id), 'date']
        diff = (dates - (row.base_date + pd.Timedelta(days=7))).abs()
        best = dates[diff.eq(diff.min())]
        nearest_ties += int(len(best) > 1)
        if row.target_date not in best.values:
            nearest_mismatches.append({'station_id': row.station_id, 'base_date': str(row.base_date.date())})
    summary['train_valid_nearest_date_mismatches'] = nearest_mismatches
    summary['train_valid_nearest_date_ties'] = nearest_ties
    destination = ROOT / 'research' / 'full_audit.json'
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: summary[key] for key in ['split', 'target_stats_by_season', 'persistence',
          'persistence_2025_season_by_station', 'missing_base_age_days',
          'train_valid_nearest_date_mismatches', 'train_valid_nearest_date_ties']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
