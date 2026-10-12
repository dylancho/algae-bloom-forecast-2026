import unittest
from pathlib import Path

import pandas as pd

from algae import sameday
from algae.features import build_features, load_bundle

ROOT = Path(__file__).resolve().parents[1]
DAILY = [('asos_daily', 'tm'), ('dam_daily', 'obs_date')]


def mutate(tables, when):
    changed = {k: v.copy() for k, v in tables.items()}
    for name, date in DAILY:
        frame = changed[name]
        cols = [c for c in frame.select_dtypes('number') if c not in ['dam_code', 'stnId']]
        frame.loc[when(frame[date]), cols] = 99999999
    return changed


class SameDayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = load_bundle(ROOT/'02_dataset_bundle_algae2026.zip')
        p = cls.tables['eval_pairs']
        cls.pairs = p[p.base_date.eq('2024-08-05')].copy()
        assert len(cls.pairs)
        cls.base = '2024-08-05'

    def test_after_base_date_is_never_read(self):
        before = sameday.build(self.tables, self.pairs)
        after = sameday.build(mutate(self.tables, lambda d: d > self.base), self.pairs)
        pd.testing.assert_frame_equal(before, after)

    def test_base_date_is_read(self):
        before = sameday.build(self.tables, self.pairs)
        after = sameday.build(mutate(self.tables, lambda d: d == self.base), self.pairs)
        self.assertFalse(before[sameday.EXTRA].equals(after[sameday.EXTRA]))
        self.assertFalse(before.sdf_wx_avgTa_7d.equals(after.sdf_wx_avgTa_7d))

    def test_shift_equals_v1_one_day_later(self):
        """Same-day windows on base d equal v1 windows computed for base d+1 (daily columns)."""
        full = sameday.full_sameday(self.tables, self.pairs)
        later = self.pairs.copy()
        later['base_date'] = (pd.to_datetime(later.base_date)+pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')
        v1, _ = build_features(self.tables, later)
        cols = [c for c in v1 if c.startswith(('wx_', 'dam'))]
        pd.testing.assert_frame_equal(full[cols].reset_index(drop=True), v1[cols].reset_index(drop=True))


if __name__ == '__main__':
    unittest.main()
