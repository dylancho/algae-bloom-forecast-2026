import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from algae.features import load_bundle, build_features, TARGET

ROOT = Path(__file__).resolve().parents[1]


class CausalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = load_bundle(ROOT/'02_dataset_bundle_algae2026.zip')

    def test_future_mutations_do_not_change_past_features(self):
        cutoff = '2024-08-01'
        pairs = self.tables['eval_pairs']
        pairs = pairs[(pairs.base_date <= cutoff) & (pairs.base_date >= '2024-07-01')].groupby('station_id').tail(1)
        before, _ = build_features(self.tables, pairs)
        changed = {k: v.copy() for k, v in self.tables.items()}
        for name, date in [('algae_measure','chck_de'), ('asos_daily','tm'), ('dam_daily','obs_date')]:
            frame = changed[name]
            cols = [c for c in frame.select_dtypes('number') if c not in ['dam_code','stnId']]
            frame.loc[frame[date] > cutoff, cols] = 99999999
        after, _ = build_features(changed, pairs)
        pd.testing.assert_frame_equal(before, after)

    def test_target_schedule_is_not_an_input(self):
        pairs = self.tables['eval_pairs'].groupby('station_id').tail(1).copy()
        before, _ = build_features(self.tables, pairs)
        pairs['target_date'] = '2099-01-01'
        pairs['lag_days'] = 10000
        pairs['split'] = 'anything'
        after, _ = build_features(self.tables, pairs)
        pd.testing.assert_frame_equal(before, after)

    def test_missing_base_uses_previous_valid_measurement(self):
        pairs = pd.DataFrame([{'station_id':'CS','base_date':'2026-07-20'}])
        x, audit = build_features(self.tables, pairs)
        self.assertEqual(x.base_cells_missing.iloc[0], 1)
        self.assertGreater(x.cells_age_lag0.iloc[0], 0)
        source = self.tables['algae_measure']
        source = source[(source.station_id=='CS') & (source.chck_de<'2026-07-20') & source[TARGET].notna()]
        expected = source.sort_values('chck_de')[TARGET].iloc[-1]
        self.assertAlmostEqual(x.cells_log_lag0.iloc[0], np.log10(expected+1))
        self.assertLess(audit.weather_latest_source.iloc[0], audit.base_date.iloc[0])

    def test_daily_same_day_observations_are_not_available(self):
        pairs = pd.DataFrame([{'station_id':'GJGR','base_date':'2024-07-29'}])
        before, _ = build_features(self.tables, pairs)
        changed = {k: v.copy() for k, v in self.tables.items()}
        for name, date in [('asos_daily','tm'),('dam_daily','obs_date')]:
            frame = changed[name]
            cols = [c for c in frame.select_dtypes('number') if c not in ['dam_code','stnId']]
            frame.loc[frame[date] >= '2024-07-29', cols] = 99999999
        after, _ = build_features(changed, pairs)
        pd.testing.assert_frame_equal(before, after)

    def test_every_cv_training_label_precedes_forecast_year(self):
        from algae.experiment import get_data,folds
        data,_=get_data()
        self.assertTrue(data.loc[data.split.isin(['test','excluded']),'target'].isna().all())
        for year in [2021,2022,2023,2024,2025]:
            train,valid=folds(data,year)
            self.assertLess(train.target_date.max(),pd.Timestamp(f'{year}-01-01'))
            self.assertFalse(train.index.intersection(valid.index).size)


if __name__ == '__main__':
    unittest.main()
