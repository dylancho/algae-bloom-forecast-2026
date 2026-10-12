import unittest
from pathlib import Path

import pandas as pd

from algae import alert
from algae.features import load_bundle

ROOT = Path(__file__).resolve().parents[1]


class AlertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = load_bundle(ROOT/'02_dataset_bundle_algae2026.zip')
        p = cls.tables['eval_pairs']
        cls.pairs = p[(p.base_date >= '2024-08-01') & (p.base_date <= '2024-08-20')].copy()
        cls.cut = '2024-08-20'

    def test_future_mutation_does_not_change_features(self):
        before = alert.build(self.tables, self.pairs)
        changed = {k: v.copy() for k, v in self.tables.items()}
        for name, date in [('algae_measure', 'chck_de'), ('dam_daily', 'obs_date')]:
            frame = changed[name]
            cols = [c for c in frame.select_dtypes('number') if c not in ['dam_code']]
            frame.loc[frame[date] > self.cut, cols] = 99999999
        after = alert.build(changed, self.pairs)
        pd.testing.assert_frame_equal(before, after)

    def test_dam_base_date_not_read(self):
        last = self.pairs.base_date.max()
        pairs = self.pairs[self.pairs.base_date.eq(last)]
        before = alert.build(self.tables, pairs)
        changed = {k: v.copy() for k, v in self.tables.items()}
        d = changed['dam_daily']
        d.loc[d.obs_date.ge(last), ['tdqty', 'iqty']] = 99999999
        after = alert.build(changed, pairs)
        pd.testing.assert_frame_equal(before, after)

    def test_state_needs_two_consecutive(self):
        self.assertEqual(alert._state_path([0, 20000, 0, 20000]), [0, 0, 0, 0])
        self.assertEqual(alert._state_path([0, 20000, 30000, 500, 400]), [0, 0, 2, 2, 0])
        self.assertEqual(alert._state_path([20000, 30000, 5000, 3000]), [0, 2, 2, 1])


if __name__ == '__main__':
    unittest.main()
