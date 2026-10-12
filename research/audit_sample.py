"""Read the supplied sample bundle without training on its test-period labels."""
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd


def audit(source: Path) -> dict:
    with zipfile.ZipFile(source) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read('05_sample_data_public_csv.zip'))) as inner:
            frames = {
                Path(name).name: pd.read_csv(io.BytesIO(inner.read(name)))
                for name in inner.namelist() if name.endswith('.csv')
            }
    tables = {}
    for name, frame in frames.items():
        item = {
            'rows': len(frame), 'columns': list(frame.columns),
            'missing': frame.isna().sum().to_dict(),
        }
        for key in ['chck_de', 'obs_date', 'tm', 'base_date', 'target_date']:
            if key in frame:
                item[key] = [frame[key].min(), frame[key].max()]
        for key in ['station_id', 'dam_name', 'stnNm', 'split', 'lag_days']:
            if key in frame:
                item[key] = frame[key].value_counts().to_dict()
        tables[name] = item
    algae = frames['algae_measure_sample.csv']
    pairs = frames['eval_pairs_sample.csv']
    target = 'iem_bgalage_cell_co'
    joined = pairs.copy()
    for date_key, value_key in [('base_date', 'base_cells'), ('target_date', 'target_cells')]:
        right = algae[['station_id', 'chck_de', target]].rename(
            columns={'chck_de': date_key, target: value_key})
        joined = joined.merge(right, on=['station_id', date_key], how='left', validate='many_to_one')
    genera = ['iem_bgalage_microsts', 'iem_bgalage_anba', 'iem_bgalage_osrtria', 'iem_bgalage_apzo']
    complete = algae[[target] + genera].notna().all(axis=1)
    return {
        'source': str(source), 'tables': tables,
        'diagnostics': {
            'missing_base_cells_in_pairs': int(joined.base_cells.isna().sum()),
            'missing_target_cells_in_pairs': int(joined.target_cells.isna().sum()),
            'duplicate_station_target_pairs_beyond_first': int(pairs.duplicated(['station_id', 'target_date']).sum()),
            'duplicate_station_measurement_rows': int(algae.duplicated(['station_id', 'chck_de']).sum()),
            'genus_total_mismatches': int((algae.loc[complete, genera].sum(axis=1) != algae.loc[complete, target]).sum()),
            'observed_target_summary': algae[target].describe().to_dict(),
            'missing_target_measurement_rows': algae.loc[algae[target].isna(), ['station_id', 'chck_de']].to_dict('records'),
        },
        'note': 'Descriptive audit only. No fitting, tuning, or model comparison on sample test labels.',
    }


if __name__ == '__main__':
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r'C:\Users\dylan\Downloads\sample_participants.zip')
    output = Path(__file__).with_name('sample_audit.json')
    output.write_text(json.dumps(audit(source), ensure_ascii=False, indent=2), encoding='utf-8')
    print(output)
