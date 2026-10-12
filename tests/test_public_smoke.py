"""Synthetic smoke tests: no local/private data required."""
import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_tracks import haversine_km
from analyze_path_intensity_fixed_windows import bh_adjust

class PublicSmokeTests(unittest.TestCase):
    def test_zero_distance(self):
        self.assertAlmostEqual(float(haversine_km(30, 104, 30, 104)), 0)

    def test_fdr_adjustment(self):
        adjusted = bh_adjust(pd.Series([.01, .04, .03]))
        self.assertAlmostEqual(adjusted.iloc[0], .03)
        self.assertAlmostEqual(adjusted.iloc[1], .04)
        self.assertAlmostEqual(adjusted.iloc[2], .04)

if __name__ == '__main__':
    unittest.main()
