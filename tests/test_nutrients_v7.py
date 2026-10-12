"""v7 causality: nutrient features must not change when post-cutoff samples change."""
import unittest

import numpy as np
import pandas as pd

from algae import nutrients


class NutrientCausality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = nutrients.load()
        d = pd.read_pickle(nutrients.ROOT/'artifacts/features_v6/dataset.pkl')
        cls.sample = d[d.base_date.dt.month.between(5, 10)].sample(60, random_state=0)

    def features(self, tables, lag):
        out = []
        for site, base in zip(self.sample.station_id, self.sample.base_date):
            row = {}
            for k, name in enumerate(nutrients.SLOTS[site]):
                for f, v in nutrients.slot_features(tables[name], base, lag).items():
                    row[f'nu{k}_{f}'] = v
            out.append(row)
        return pd.DataFrame(out)

    def test_future_samples_do_not_matter(self):
        for lag in [1, 14]:
            clean = self.features(self.tables, lag)
            for i, (site, base) in enumerate(zip(self.sample.station_id, self.sample.base_date)):
                cut = base-pd.Timedelta(days=lag)
                bad = {k: v.copy() for k, v in self.tables.items()}
                for name in nutrients.SLOTS[site]:
                    t = bad[name]
                    t.loc[t.index > cut] = t.loc[t.index > cut]*50+7
                row = self.features_one(bad, site, base, lag)
                expected = clean.iloc[i].dropna().to_dict()
                self.assertEqual(set(row), set(expected))
                for k, v in expected.items():
                    self.assertTrue(np.isclose(row[k], v), (site, base, k))

    def features_one(self, tables, site, base, lag):
        row = {}
        for k, name in enumerate(nutrients.SLOTS[site]):
            for f, v in nutrients.slot_features(tables[name], base, lag).items():
                if pd.notna(v):
                    row[f'nu{k}_{f}'] = v
        return row

    def test_age_respects_lag(self):
        f = self.features(self.tables, 14)
        self.assertTrue((f.filter(like='age_days') >= 14).all().all())


if __name__ == '__main__':
    unittest.main()
