import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.diurnal_eda import (
    aggregate_sensor_city_months,
    create_diurnal_sensor_figures
)


class DiurnalEdaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.panel = pd.DataFrame(
            [
                {
                    "ID_UC_G0": 1,
                    "year": 2000,
                    "month": 1,
                    "suhi_terra_day_mean": 2.0,
                    "suhi_terra_night_mean": 1.0,
                    "qc_pass_A_T": True,
                    "suhi_aqua_day_mean": 20.0,
                    "suhi_aqua_night_mean": 10.0,
                    "qc_pass_A_A": True
                },
                {
                    "ID_UC_G0": 1,
                    "year": 2001,
                    "month": 1,
                    "suhi_terra_day_mean": 4.0,
                    "suhi_terra_night_mean": 3.0,
                    "qc_pass_A_T": True,
                    "suhi_aqua_day_mean": 30.0,
                    "suhi_aqua_night_mean": 15.0,
                    "qc_pass_A_A": False
                },
                {
                    "ID_UC_G0": 1,
                    "year": 2002,
                    "month": 1,
                    "suhi_terra_day_mean": 100.0,
                    "suhi_terra_night_mean": 100.0,
                    "qc_pass_A_T": False,
                    "suhi_aqua_day_mean": 24.0,
                    "suhi_aqua_night_mean": 14.0,
                    "qc_pass_A_A": True
                },
                {
                    "ID_UC_G0": 2,
                    "year": 2000,
                    "month": 1,
                    "suhi_terra_day_mean": -2.0,
                    "suhi_terra_night_mean": -4.0,
                    "qc_pass_A_T": True,
                    "suhi_aqua_day_mean": 5.0,
                    "suhi_aqua_night_mean": 4.0,
                    "qc_pass_A_A": False
                }
            ]
        )

    def test_aggregates_paired_qc_values_by_city_and_month(self) -> None:
        terra, terra_summary = aggregate_sensor_city_months(
            self.panel,
            "terra",
            2000,
            2020,
            "median"
        )
        aqua, aqua_summary = aggregate_sensor_city_months(
            self.panel,
            "aqua",
            2000,
            2020,
            "median"
        )

        city_one_terra = terra.loc[terra["ID_UC_G0"] == 1].iloc[0]
        self.assertEqual(city_one_terra["day_suhi"], 3.0)
        self.assertEqual(city_one_terra["night_suhi"], 2.0)
        self.assertEqual(city_one_terra["n_valid_years"], 2)
        self.assertEqual(len(terra), 2)
        self.assertEqual(
            terra_summary.loc[1, "qc_pass_city_year_months"],
            3
        )
        self.assertEqual(terra_summary.loc[1, "plotted_cities"], 2)
        self.assertEqual(terra_summary.loc[1, "total_cities"], 2)

        city_one_aqua = aqua.iloc[0]
        self.assertEqual(city_one_aqua["day_suhi"], 22.0)
        self.assertEqual(city_one_aqua["night_suhi"], 12.0)
        self.assertEqual(city_one_aqua["n_valid_years"], 2)
        self.assertEqual(len(aqua), 1)
        self.assertEqual(
            aqua_summary.loc[1, "qc_pass_city_year_months"],
            2
        )
        self.assertEqual(aqua_summary.loc[1, "plotted_cities"], 1)

    def test_creates_two_sensor_figures(self) -> None:
        terra, _ = aggregate_sensor_city_months(
            self.panel,
            "terra",
            2000,
            2020,
            "median"
        )
        aqua, _ = aggregate_sensor_city_months(
            self.panel,
            "aqua",
            2000,
            2020,
            "median"
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            outputs = create_diurnal_sensor_figures(
                {"terra": terra, "aqua": aqua},
                Path(temporary_dir),
                2000,
                2020,
                "median",
                80
            )
            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(path.is_file() for path in outputs))
            self.assertTrue(all(path.stat().st_size > 0 for path in outputs))


if __name__ == "__main__":
    unittest.main()
