import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.config import load_analysis_config
from src.eda_seasonal_contrast import analyze_seasonal_contrasts
from src.eda_visualizations.seasonal_contrasts import (
    create_seasonal_contrast_outputs
)


CLIMATE_GROUPS = ("Tropical", "Arid", "Continental")


def _synthetic_monthly_panel() -> pd.DataFrame:
    cities = (
        (1, 25.0, "Tropical"),
        (2, -25.0, "Arid"),
        (3, 0.0, "Continental")
    )
    records = []
    for city_id, latitude, climate_group in cities:
        for year in range(1999, 2021):
            for month in range(1, 13):
                north = latitude > 10
                south = latitude < -10
                summer = (
                    (north and month in (6, 7, 8))
                    or (south and month in (12, 1, 2))
                )
                winter = (
                    (north and month in (12, 1, 2))
                    or (south and month in (6, 7, 8))
                )
                season_year = year + (month == 12 and (summer or winter))
                elapsed = season_year - 2000
                record = {
                    "ID_UC_G0": city_id,
                    "year": year,
                    "month": month,
                    "centroid_lat": latitude,
                    "koppen_geiger_macro_group": climate_group,
                    "qc_pass_A_T": True,
                    "qc_pass_A_A": True
                }
                for sensor, offset in (("terra", 0.0), ("aqua", 2.0)):
                    if summer:
                        urban_day = 20 + offset + 0.3 * elapsed
                        rural_day = 15 + offset + 0.2 * elapsed
                        urban_night = 18 + offset + 0.2 * elapsed
                        rural_night = 14 + offset + 0.1 * elapsed
                    else:
                        urban_day = 10 + offset + 0.1 * elapsed
                        rural_day = 8 + offset + 0.1 * elapsed
                        urban_night = 9 + offset + 0.1 * elapsed
                        rural_night = 7 + offset + 0.05 * elapsed
                    for side, day, night in (
                        ("urban", urban_day, urban_night),
                        ("rural", rural_day, rural_night),
                        ("suhi", urban_day - rural_day, urban_night - rural_night)
                    ):
                        record[f"{side}_{sensor}_day_mean"] = day
                        record[f"{side}_{sensor}_night_mean"] = night
                records.append(record)
    return pd.DataFrame(records)


class SeasonalContrastEdaTests(unittest.TestCase):
    def test_config_defines_local_seasons_and_equatorial_exclusion(self) -> None:
        config = load_analysis_config()
        self.assertEqual(config.seasonal_contrast.northern_summer_months, (6, 7, 8))
        self.assertEqual(config.seasonal_contrast.northern_winter_months, (12, 1, 2))
        self.assertEqual(
            config.seasonal_contrast.minimum_valid_months_per_season,
            3
        )
        self.assertEqual(config.seasonal_contrast.equatorial_policy, "exclude")

    def test_southern_season_years_and_contrast_trends_decompose(self) -> None:
        results = analyze_seasonal_contrasts(
            panel=_synthetic_monthly_panel(),
            start_year=2000,
            end_year=2020,
            equatorial_band_degrees=10,
            northern_summer_months=(6, 7, 8),
            northern_winter_months=(12, 1, 2),
            minimum_valid_months_per_season=3,
            minimum_trend_years=15,
            climate_groups=CLIMATE_GROUPS,
            bootstrap_replicates=100,
            confidence_level=0.95,
            random_seed=42
        )

        labels = results.city_month_labels
        north_december = labels.loc[
            labels["ID_UC_G0"].eq(1)
            & labels["year"].eq(2000)
            & labels["month"].eq(12)
        ].iloc[0]
        south_december = labels.loc[
            labels["ID_UC_G0"].eq(2)
            & labels["year"].eq(2000)
            & labels["month"].eq(12)
        ].iloc[0]
        equatorial_summer = labels.loc[
            labels["ID_UC_G0"].eq(3)
            & labels["year"].eq(2000)
            & labels["month"].eq(7)
        ].iloc[0]
        self.assertEqual((north_december["is_summer"], north_december["is_winter"]), (0, 1))
        self.assertEqual(north_december["season_year"], 2001)
        self.assertEqual((south_december["is_summer"], south_december["is_winter"]), (1, 0))
        self.assertEqual(south_december["season_year"], 2001)
        self.assertEqual(
            (equatorial_summer["is_summer"], equatorial_summer["is_winter"]),
            (0, 0)
        )
        self.assertEqual(equatorial_summer["season"], "equatorial")

        equatorial_rows = results.city_year_seasons.loc[
            results.city_year_seasons["ID_UC_G0"].eq(3)
        ]
        self.assertTrue(equatorial_rows.empty)
        contrast = results.city_year_contrasts.loc[
            results.city_year_contrasts["ID_UC_G0"].eq(1)
            & results.city_year_contrasts["season_year"].eq(2005)
        ].iloc[0]
        self.assertAlmostEqual(contrast["contrast_terra_urban_D"], 11.0)
        self.assertAlmostEqual(contrast["contrast_terra_rural_D"], 7.5)
        self.assertAlmostEqual(contrast["contrast_terra_suhi_D"], 3.5)

        trends = results.city_trends.loc[
            results.city_trends["sensor"].eq("terra")
        ]
        self.assertEqual(len(trends), 2)
        self.assertTrue(
            (trends["gamma_urban_D"].sub(2.0).abs() < 1e-10).all()
        )
        self.assertTrue(
            (trends["gamma_rural_D"].sub(1.0).abs() < 1e-10).all()
        )
        self.assertTrue(
            (trends["gamma_suhi_D"].sub(1.0).abs() < 1e-10).all()
        )
        self.assertTrue(
            (
                trends["gamma_suhi_D"]
                - trends["gamma_urban_D"]
                + trends["gamma_rural_D"]
            ).abs().lt(1e-10).all()
        )
        summary = results.climate_summary.loc[
            results.climate_summary["climate_group"].eq("All cities")
            & results.climate_summary["sensor"].eq("terra")
            & results.climate_summary["dataset"].eq("suhi")
            & results.climate_summary["metric"].eq("D")
        ].iloc[0]
        self.assertEqual(summary["positive_gamma_pct"], 100)
        self.assertEqual(summary["median_trend_direction"], "increasing")

    def test_outputs_write_city_tables_and_figures(self) -> None:
        results = analyze_seasonal_contrasts(
            panel=_synthetic_monthly_panel(),
            start_year=2000,
            end_year=2020,
            equatorial_band_degrees=10,
            northern_summer_months=(6, 7, 8),
            northern_winter_months=(12, 1, 2),
            minimum_valid_months_per_season=3,
            minimum_trend_years=15,
            climate_groups=CLIMATE_GROUPS,
            bootstrap_replicates=100,
            confidence_level=0.95,
            random_seed=42
        )
        config = load_analysis_config()
        with tempfile.TemporaryDirectory() as temporary_dir:
            outputs = create_seasonal_contrast_outputs(
                results=results,
                output_dir=Path(temporary_dir),
                start_year=2000,
                end_year=2020,
                climate_groups=CLIMATE_GROUPS,
                confidence_level=0.95,
                parquet_compression=config.parquet_compression,
                dpi=30
            )
            self.assertEqual(len(outputs), 7)
            self.assertTrue(all(path.is_file() for path in outputs))
            label_table = pd.read_parquet(outputs[0])
            self.assertTrue({"is_summer", "is_winter"} <= set(label_table))
            self.assertEqual(str(label_table["is_summer"].dtype), "int8")


if __name__ == "__main__":
    unittest.main()
