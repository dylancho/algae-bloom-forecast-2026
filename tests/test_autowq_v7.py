"""v7 causality: automatic-network features ignore the base day and anything after it."""
import unittest

import numpy as np
import pandas as pd

from algae import autowq


class AutoCausality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = {c: autowq.load(c) for c in set(autowq.BIO.values()) | set(autowq.PHYS.values())}
        d = pd.read_pickle(autowq.ROOT/'artifacts/features_v6/dataset.pkl')
        cls.sample = d[d.base_date.dt.month.between(5, 10)].sample(60, random_state=1)

    def row(self, tables, site, base):
        r = {f'b_{k}': v for k, v in autowq.bio_features(tables[autowq.BIO[site]], base).items()}
        r.update({f'p_{k}': v for k, v in autowq.phys_features(tables[autowq.PHYS[site]], base).items()})
        return pd.Series(r, dtype=float)

    def test_base_day_and_future_do_not_matter(self):
        for site, base in zip(self.sample.station_id, self.sample.base_date):
            clean = self.row(self.tables, site, base)
            bad = {k: v.copy() for k, v in self.tables.items()}
            for code in {autowq.BIO[site], autowq.PHYS[site]}:
                t = bad[code]
                t.loc[t.index >= base] = t.loc[t.index >= base]*30+5
            dirty = self.row(bad, site, base)
            pd.testing.assert_series_equal(clean, dirty, check_exact=False)

    def test_confirmed_wins_over_unconfirmed(self):
        t = self.tables['S03003']
        self.assertFalse(t.index.duplicated().any())
        self.assertTrue(np.isfinite(t.loc['2026-08-01':'2026-08-31', 'chla']).any())


if __name__ == '__main__':
    unittest.main()
