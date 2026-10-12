import unittest

import numpy as np
import pandas as pd

from algae.experiment import ROOT, folds
from algae.features import TARGET, load_bundle
from algae.features_v6 import build_dynamic, build_static, cell_filter, thermal_filter
from algae.wamis import SLOTS, load as load_gauge


class StrategyCausalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = load_bundle(ROOT/'02_dataset_bundle_algae2026.zip')
        cls.flows = {s: load_gauge(c[0], 'flw') for s, c in SLOTS.items()}

    def test_events_ignore_future_and_same_day_daily_observations(self):
        pairs = pd.DataFrame([{'station_id': 'CS', 'base_date': '2024-07-29'}])
        before = build_static(self.tables, pairs, self.flows)
        altered = {k: v.copy() for k, v in self.tables.items()}
        a = altered['algae_measure']
        for c in a.select_dtypes('number'):
            a.loc[a.chck_de > '2024-07-29', c] = 999999
        wx = altered['asos_daily']
        for c in wx.select_dtypes('number'):
            if c != 'stnId':
                wx.loc[wx.tm >= '2024-07-29', c] = 999999
        flow = {s: q.mask(q.index >= pd.Timestamp('2024-07-29'), 999999) for s, q in self.flows.items()}
        after = build_static(altered, pairs, flow)
        pd.testing.assert_frame_equal(before, after)

    def test_target_schedule_does_not_change_features(self):
        pairs = pd.DataFrame([{'station_id': 'MGMR', 'base_date': '2024-07-29'}])
        before = build_static(self.tables, pairs, self.flows)
        pairs['target_date'] = '2099-01-01'
        pairs['lag_days'] = 999
        after = build_static(self.tables, pairs, self.flows)
        pd.testing.assert_frame_equal(before, after)

    def test_thermal_fit_and_filter_ignore_later_observations(self):
        dates = pd.date_range('2022-01-01', '2023-07-31')
        air = 15+8*np.sin(np.arange(len(dates))/60)
        wx = pd.DataFrame({'avgTa': air, 'sumSsHr': 6.}, index=dates)
        obs_dates = dates[::7]
        own = pd.DataFrame({'iem_wtrtp': 3+air[::7]}, index=obs_dates)
        before, audit = thermal_filter(own, wx, '2023-06-30')
        changed = own.copy()
        changed.loc[changed.index > '2023-06-30', 'iem_wtrtp'] = 9999
        changed_wx = wx.copy()
        changed_wx.loc[changed_wx.index >= '2023-06-30', 'avgTa'] = 9999
        after, _ = thermal_filter(changed, changed_wx, '2023-06-30')
        pd.testing.assert_frame_equal(before, after)
        self.assertTrue((audit.fit_n >= 25).all())
        # Every 2023 regression uses the 2022 observations only.
        self.assertEqual(before.loc['2023-06-30', 'fit_n'],
                         ((own.index.year == 2022) & (own.index > dates[0])).sum())
        day = own.loc['2023-06-01':'2023-06-30'].index[-1]
        alter_now = own.copy()
        alter_now.loc[day, 'iem_wtrtp'] += 5
        _, shifted = thermal_filter(alter_now, wx, '2023-06-30')
        p0 = audit.loc[audit.date.eq(day), 'pred'].iloc[0]
        p1 = shifted.loc[shifted.date.eq(day), 'pred'].iloc[0]
        self.assertEqual(p0, p1)

    def test_cell_filter_is_prefix_invariant(self):
        dates = pd.date_range('2020-01-01', periods=20, freq='7D')
        a = pd.DataFrame({TARGET: np.arange(20)**3}, index=dates)
        before = cell_filter(a, dates[10])
        a.loc[dates[11]:, TARGET] = 1e9
        after = cell_filter(a, dates[10])
        pd.testing.assert_frame_equal(before, after)
        self.assertTrue(np.isfinite(before.to_numpy()).all())

    def test_weights_only_change_training_objective(self):
        from algae.exp6 import training_weights
        dates = pd.date_range('2021-01-01', periods=40, freq='7D')
        train = pd.DataFrame({'station_id': ['CS']*40, 'base_date': dates})
        delta = np.r_[np.zeros(32), np.ones(4), -np.ones(4)]
        for method in ['event_weight', 'lds_weight']:
            w = training_weights(train, delta, method)
            self.assertEqual(len(w), len(train))
            self.assertTrue(np.isfinite(w).all())
            self.assertAlmostEqual(w.mean(), 1.)
            self.assertGreater(w[-1], w[0])

    def test_build_preserves_labels_and_masked_real_parent(self):
        from algae.experiment import get_data
        base, names = get_data()
        v6 = pd.read_pickle(ROOT/'artifacts/features_v6/dataset.pkl')
        pd.testing.assert_frame_equal(base, v6[base.columns])
        self.assertTrue(v6.loc[v6.split.isin(['test', 'excluded']), 'target'].isna().all())
        masked = pd.read_pickle(ROOT/'artifacts/features_v6/history_masked.pkl')
        for year in range(2018, 2025):
            train, _ = folds(v6, year, since=2017)
            aug = masked[masked.parent_index.isin(train.index)]
            parents = base.loc[aug.parent_index]
            np.testing.assert_allclose(aug.target, parents.target)
            np.testing.assert_allclose(aug.cells_log_lag0, parents.cells_log_lag0)
            self.assertTrue((aug.removed_date < aug.base_date).all())
            self.assertTrue((aug.target_date < pd.Timestamp(f'{year}-01-01')).all())

    def test_history_mixup_respects_both_parents_and_recomputed_growth(self):
        from algae.experiment import get_data
        base, _ = get_data()
        mix = pd.read_pickle(ROOT/'artifacts/features_v6/history_mixup.pkl')
        parents = base.loc[mix.parent_index]
        donors = base.loc[mix.donor_index]
        self.assertTrue(np.array_equal(parents.station_id, donors.station_id))
        self.assertTrue(np.array_equal(parents.lag_days, donors.lag_days))
        f = mix.fraction.to_numpy()
        expected = (1-f)*parents.cells_log_lag0.to_numpy()+f*donors.cells_log_lag0.to_numpy()
        np.testing.assert_allclose(mix.cells_log_lag0, expected, rtol=1e-9, atol=1e-9)
        growth = (mix.cells_log_lag0-mix.cells_log_lag1)/(mix.cells_age_lag1-mix.cells_age_lag0)
        np.testing.assert_allclose(mix.growth_per_day, growth, equal_nan=True)
        for year in range(2018, 2025):
            train, _ = folds(base, year, since=2017)
            aug = mix[mix.parent_index.isin(train.index) & mix.donor_index.isin(train.index)]
            self.assertTrue((aug.info_end < pd.Timestamp(f'{year}-01-01')).all())


if __name__ == '__main__':
    unittest.main()
