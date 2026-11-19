import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.era5_modis_agreement_figures import (
    SLOTS,
    plot_betas,
    regional_differences,
    summaries,
    validate_and_select,
    validate_regions,
    verify_common_year_fits,
)


def _row(city, region, warm, cold):
    row = {
        "ID_UC_G0": city, "slot": "terra_day", "hemisphere": "north",
        "koppen_geiger_class": "Cfa", "summer_season": "JJA", "winter_season": "DJF",
        "n_common_years": 15, "common_years": list(range(2003, 2018)), "GC_DEV_USR": region,
    }
    for series in ("era5", "urban", "rural"):
        row[f"beta_{series}_summer"] = warm
        row[f"beta_{series}_winter"] = cold
        row[f"gamma_{series}"] = warm - cold
    row["gamma_suhi"] = 0.0
    return row


class AgreementFigureTests(unittest.TestCase):
    def test_regions_come_from_contrast_and_must_be_valid(self):
        frame = pd.DataFrame([_row(1, " Europe ", 2.0, 1.0)])
        self.assertEqual(validate_regions(frame).loc[0, "GC_DEV_USR"], "Europe")
        for value in (None, " ", 123):
            invalid = frame.copy()
            invalid.loc[0, "GC_DEV_USR"] = value
            with self.assertRaises(ValueError):
                validate_regions(invalid)
        other_slot = _row(1, "East Asia", 2.0, 1.0)
        other_slot["slot"] = "aqua_day"
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_regions(pd.concat([frame, pd.DataFrame([other_slot])], ignore_index=True))
        with self.assertRaisesRegex(ValueError, "GC_DEV_USR"):
            validate_and_select(frame.drop(columns="GC_DEV_USR"), 1e-8)

    def test_select_rejects_duplicate_and_bad_gamma(self):
        frame = pd.DataFrame([_row(1, "Europe", 2.0, 1.0)])
        self.assertEqual(len(validate_and_select(frame, 1e-8)), 1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_and_select(pd.concat([frame, frame]), 1e-8)
        frame.loc[0, "gamma_suhi"] = 1.0
        with self.assertRaisesRegex(ValueError, "decomposition"):
            validate_and_select(frame, 1e-8)

    def test_common_years_and_all_six_slopes_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seasons.parquet"
            rows = []
            for year in range(2003, 2020):
                for season in ("JJA", "DJF"):
                    elapsed = year - 2003
                    rate = 0.2 if season == "JJA" else 0.1
                    obs = {
                        "ID_UC_G0": 1, "season_year": year, "season": season,
                        "era5_temperature_2m_c": rate * elapsed,
                        "era5_qc_pass": not (year == 2018 and season == "JJA"),
                    }
                    for slot in SLOTS:
                        obs[f"qc_pass_season_{slot}"] = not (year == 2019 and season == "DJF")
                        obs[f"urban_{slot}_mean"] = rate * elapsed
                        obs[f"rural_{slot}_mean"] = rate * elapsed
                        obs[f"suhi_{slot}_mean"] = 0.0
                    rows.append(obs)
            duckdb.from_df(pd.DataFrame(rows)).write_parquet(str(path))
            frame = pd.DataFrame([_row(1, "Europe", 2.0, 1.0)])
            verify_common_year_fits(frame, path, 1e-8)
            frame.at[0, "common_years"] = list(range(2004, 2019))
            with self.assertRaisesRegex(ValueError, "common years"):
                verify_common_year_fits(frame, path, 1e-8)
            frame.at[0, "common_years"] = list(range(2003, 2018))
            frame.loc[0, "beta_urban_winter"] = 1.5
            with self.assertRaisesRegex(ValueError, "trend slopes"):
                verify_common_year_fits(frame, path, 1e-8)

    def test_summary_uses_paired_cities_and_median_is_not_additive(self):
        frame = pd.DataFrame([
            _row(1, "Europe", 0.0, 0.0),
            _row(2, "Europe", 1.0, 10.0),
            _row(3, "Europe", 10.0, 2.0),
            _row(4, "East Asia", 0.0, 0.0),
        ])
        frame = validate_and_select(frame, 1e-8)
        summary = summaries(frame, "Cfa")
        differences = regional_differences(summary, "Europe", "East Asia").set_index("metric")
        self.assertEqual(differences.loc["gamma_era5", "median_difference_reference_minus_comparison"], 0.0)
        warm = differences.loc["beta_era5_summer", "median_difference_reference_minus_comparison"]
        cold = differences.loc["beta_era5_winter", "median_difference_reference_minus_comparison"]
        self.assertEqual(warm - cold, -1.0)
        self.assertAlmostEqual(
            differences.loc["gamma_era5", "mean_difference_reference_minus_comparison"],
            differences.loc["beta_era5_summer", "mean_difference_reference_minus_comparison"]
            - differences.loc["beta_era5_winter", "mean_difference_reference_minus_comparison"],
        )
        figure = plot_betas(summary, "Cfa", "terra_day")
        self.assertEqual(figure.axes[0].get_xlim(), figure.axes[1].get_xlim())
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
