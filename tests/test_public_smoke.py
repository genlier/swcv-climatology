"""Synthetic smoke tests: no local/private data required."""
import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_tracks import haversine_km
from analyze_catalogue_count_sensitivity import annual_counts

class PublicSmokeTests(unittest.TestCase):
    def test_zero_distance(self):
        self.assertAlmostEqual(float(haversine_km(30, 104, 30, 104)), 0)

    def test_annual_and_complete_winter_counts(self):
        events = pd.DataFrame({'year':[1979,1979,2023,2023],
            'season_year':[1979,1980,2023,2024],
            'season':['other','DJF','other','DJF'], 'track_cluster':[1,1,1,1]})
        self.assertEqual(annual_counts(events, 'annual_all')['count'].sum(), 4)
        self.assertEqual(annual_counts(events, 'djf_cluster1')['count'].sum(), 1)

if __name__ == '__main__':
    unittest.main()
