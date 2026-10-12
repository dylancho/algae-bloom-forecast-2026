import json
import unittest
from pathlib import Path

import pandas as pd

from algae.features import load_bundle
from algae.features_v2 import build_extra

ROOT = Path(__file__).resolve().parents[1]


class CausalityV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = load_bundle(ROOT/'02_dataset_bundle_algae2026.zip')

    def _mutate(self, cutoff, inclusive):
        changed = {k: v.copy() for k, v in self.tables.items()}
        for name, date in [('algae_measure', 'chck_de'), ('asos_daily', 'tm'), ('dam_daily', 'obs_date')]:
            frame = changed[name]
            cols = [c for c in frame.select_dtypes('number') if c not in ['dam_code', 'stnId']]
            mask = frame[date] >= cutoff if inclusive else frame[date] > cutoff
            frame.loc[mask, cols] = 99999999
        return changed

    def test_future_mutations_do_not_change_v2_features(self):
        cutoff = '2024-08-01'
        pairs = self.tables['eval_pairs']
        pairs = pairs[(pairs.base_date <= cutoff) & (pairs.base_date >= '2024-07-01')].groupby('station_id').tail(1)
        before = build_extra(self.tables, pairs)
        after = build_extra(self._mutate(cutoff, inclusive=False), pairs)
        pd.testing.assert_frame_equal(before, after)

    def test_daily_same_day_observations_are_not_available_in_v2(self):
        pairs = pd.DataFrame([{'station_id': 'GJGR', 'base_date': '2024-07-29'}])
        before = build_extra(self.tables, pairs)
        changed = {k: v.copy() for k, v in self.tables.items()}
        for name, date in [('asos_daily', 'tm'), ('dam_daily', 'obs_date')]:
            frame = changed[name]
            cols = [c for c in frame.select_dtypes('number') if c not in ['dam_code', 'stnId']]
            frame.loc[frame[date] >= '2024-07-29', cols] = 99999999
        after = build_extra(changed, pairs)
        pd.testing.assert_frame_equal(before, after)

    def test_target_schedule_is_not_a_v2_input(self):
        pairs = self.tables['eval_pairs'].groupby('station_id').tail(1).copy()
        before = build_extra(self.tables, pairs)
        pairs['target_date'] = '2099-01-01'
        pairs['lag_days'] = 10000
        pairs['split'] = 'anything'
        after = build_extra(self.tables, pairs)
        pd.testing.assert_frame_equal(before, after)

    def test_v2_dataset_preserves_v1_columns_exactly(self):
        manifest = json.loads((ROOT/'artifacts/features_v2/manifest.json').read_text(encoding='utf-8'))
        v1 = pd.read_pickle(ROOT/'artifacts/features/dataset.pkl')
        v2 = pd.read_pickle(ROOT/'artifacts/features_v2/dataset.pkl')
        pd.testing.assert_frame_equal(v1[manifest['v1_features']], v2[manifest['v1_features']])
        self.assertFalse(set(manifest['new_features']) & set(manifest['v1_features']))

    def test_v2_training_labels_precede_forecast_year(self):
        from algae.experiment import folds
        from algae.exp2 import get_data
        data, names = get_data('full2')
        self.assertTrue(data.loc[data.split.isin(['test', 'excluded']), 'target'].isna().all())
        for name in ['target_date', 'lag_days', 'split', 'target']:
            self.assertNotIn(name, names)
        for year in [2021, 2022, 2023, 2024, 2025]:
            train, valid = folds(data, year)
            self.assertLess(train.target_date.max(), pd.Timestamp(f'{year}-01-01'))
            self.assertFalse(train.index.intersection(valid.index).size)


if __name__ == '__main__':
    unittest.main()
