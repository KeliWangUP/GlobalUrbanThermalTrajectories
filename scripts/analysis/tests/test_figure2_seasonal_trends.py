import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_figure2_seasonal_trends import (
    paired_year_trends,
    select_group,
    summarize_full_sample,
    summarize_group,
    summarize_paired_group,
)


GROUP = ("Dfb", "Europe", "night")


def example_tables():
    common = {"climate": "Dfb", "region": "Europe", "hemisphere": "north", "slot": "terra_night"}
    trends = pd.DataFrame([
        {**common, "city_id": city, "season": season, "beta_urban": urban, "beta_rural": rural, "n_years": 17}
        for city, season, urban, rural in (
            (1, "JJA", 2.0, 1.0), (1, "DJF", 0.1, 0.5),
            (2, "JJA", 4.0, 2.0), (2, "DJF", 1.0, 1.5),
            (3, "JJA", 4.0, 3.0),
        )
    ])
    contrasts = pd.DataFrame([
        {**common, "city_id": 1, "warm_season": "JJA", "cold_season": "DJF",
         "beta_urban_warm": 2.0, "beta_rural_warm": 1.0, "beta_urban_cold": 0.1, "beta_rural_cold": 0.5,
         "n_years_warm": 17, "n_years_cold": 17, "gamma_suhi": 123.0},
        {**common, "city_id": 2, "warm_season": "JJA", "cold_season": "DJF",
         "beta_urban_warm": 4.0, "beta_rural_warm": 2.0, "beta_urban_cold": 1.0, "beta_rural_cold": 1.5,
         "n_years_warm": 17, "n_years_cold": 17, "gamma_suhi": 456.0},
    ])
    return trends, contrasts


class Figure2Tests(unittest.TestCase):
    def test_paired_city_selection_and_derived_sign(self):
        trends, contrasts = example_tables()
        selected = select_group(trends, contrasts, "terra", GROUP, 17)
        self.assertEqual(selected.city_id.tolist(), [1, 2])
        self.assertEqual(selected.gamma_suhi.tolist(), [123.0, 456.0])
        np.testing.assert_allclose(selected.d_cold, [-0.4, -0.5])
        np.testing.assert_allclose(selected.gamma_suhi_derived, [1.4, 2.5])
        summary = summarize_group(selected, np.random.default_rng(42)).set_index("metric")
        self.assertEqual(summary.loc["d_cold", "n"], 2)
        self.assertAlmostEqual(summary.loc["d_cold", "mean"], -0.45)
        self.assertAlmostEqual(summary.loc["gamma_suhi", "mean"], 1.95)
        self.assertLess(summary.loc["d_cold", "ci_upper"], 0)
        self.assertAlmostEqual(summary.loc["gamma_suhi", "mean"], summary.loc["d_warm", "mean"] - summary.loc["d_cold", "mean"])

    def test_source_disagreement_fails(self):
        trends, contrasts = example_tables()
        contrasts.loc[0, "beta_rural_cold"] = 9.0
        with self.assertRaisesRegex(ValueError, "disagree"):
            select_group(trends, contrasts, "terra", GROUP, 17)

    def test_bootstrap_resamples_all_metrics_together(self):
        frame = pd.DataFrame({
            "beta_urban_warm": [1., 5., 8.], "beta_rural_warm": [0., 0., 0.],
            "beta_urban_cold": [0., 0., 0.], "beta_rural_cold": [0., 0., 0.],
            "d_warm": [1., 5., 8.], "d_cold": [0., 0., 0.],
            "gamma_suhi_derived": [1., 5., 8.],
        })
        summary = summarize_group(frame, np.random.default_rng(23)).set_index("metric")
        for column in ("mean", "ci_lower", "ci_upper"):
            self.assertEqual(summary.loc["beta_urban_warm", column], summary.loc["gamma_suhi", column])

    def test_refits_on_shared_years_and_local_seasons(self):
        rows = []
        candidates = []
        for city, hemisphere in ((1, "north"), (2, "south")):
            for time in ("day", "night"):
                candidates.append({
                    "city_id": city, "slot": f"terra_{time}", "hemisphere": hemisphere,
                    "climate": "Dfb", "region": "Europe",
                    "warm_season": "JJA" if hemisphere == "north" else "DJF",
                    "cold_season": "DJF" if hemisphere == "north" else "JJA",
                    "beta_urban_warm": 1., "beta_rural_warm": 1.,
                    "beta_urban_cold": 1., "beta_rural_cold": 1.,
                })
            for year in range(2003, 2008):
                for season in ("JJA", "DJF"):
                    t = year - 2003
                    warm = season == ("JJA" if hemisphere == "north" else "DJF")
                    row = {"ID_UC_G0": city, "season_year": year, "season": season}
                    for sensor in ("terra", "aqua"):
                        for time in ("day", "night"):
                            prefix = f"{sensor}_{time}"
                            urban = t * t if warm else t
                            rural = 0 if warm else 2 * t
                            row[f"urban_{prefix}_mean"] = urban
                            row[f"rural_{prefix}_mean"] = rural
                            row[f"suhi_{prefix}_mean"] = urban - rural
                            row[f"qc_pass_season_{prefix}"] = not (
                                city == 1 and sensor == "terra" and time == "night"
                                and season == "DJF" and year in (2003, 2004)
                            )
                    rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seasons.parquet"
            pd.DataFrame(rows).to_parquet(path, index=False)
            full, common = paired_year_trends(path, pd.DataFrame(candidates), 2003, 2007, 3)
        day_full = full.set_index(["city_id", "slot"]).loc[(1, "terra_day")]
        day_common = common.set_index(["city_id", "slot"]).loc[(1, "terra_day")]
        self.assertEqual(day_full["n_common_years"], 5)
        self.assertAlmostEqual(day_full["beta_urban_warm"], 40)
        self.assertEqual(day_common["n_common_years"], 3)
        self.assertAlmostEqual(day_common["beta_urban_warm"], 60)
        self.assertAlmostEqual(day_common["d_cold"], -10)
        south = common.set_index(["city_id", "slot"]).loc[(2, "terra_day")]
        self.assertAlmostEqual(south["beta_urban_warm"], 40)
        self.assertAlmostEqual(south["beta_urban_cold"], 10)
        day = common.loc[common.slot.eq("terra_day")]
        night = common.loc[common.slot.eq("terra_night")]
        summaries, diff = summarize_paired_group(day, night, np.random.default_rng(42))
        self.assertEqual(diff["n"], 2)
        self.assertAlmostEqual(diff["mean_gamma_day_minus_night"], 0)
        self.assertAlmostEqual(diff["ci_lower"], 0)
        self.assertAlmostEqual(diff["ci_upper"], 0)
        for stat in summaries:
            self.assertAlmostEqual(stat.set_index("metric").loc["gamma_suhi", "mean"], 60)
        all_summary = summarize_full_sample(full, np.random.default_rng(12))
        southern = all_summary.loc[(all_summary.hemisphere == "south") & (all_summary.time == "day")].iloc[0]
        self.assertEqual(southern["n"], 1)
        self.assertEqual(southern["n_reversal"], 1)
        self.assertEqual(southern["share_reversal"], 1)


if __name__ == "__main__":
    unittest.main()
