from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.diurnal_eda import (
    _symmetric_axis_limits,
    _plot_month_by_year,
    add_local_season_coordinates,
    aggregate_sensor_city_months,
    aggregate_sensor_local_city_months,
    create_diurnal_sensor_figures,
    create_local_season_figures,
    create_year_panel_figures,
    prepare_sensor_year_month_rows
)
from src.eda_logging import configure_eda_logging


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

    def test_axis_limits_are_symmetric_around_zero(self) -> None:
        self.assertEqual(_symmetric_axis_limits([-5.0, 3.0]), (-5.25, 5.25))

    def test_year_month_rows_keep_each_year_and_qc_counts(self) -> None:
        terra, summary = prepare_sensor_year_month_rows(
            self.panel,
            "terra",
            2000,
            2001
        )

        self.assertEqual(len(terra), 3)
        self.assertEqual(
            summary.loc[(2000, 1), "qc_pass_city_months"],
            2
        )
        self.assertEqual(
            summary.loc[(2001, 1), "qc_pass_city_months"],
            1
        )
        self.assertEqual(
            summary.loc[(2001, 1), "total_city_months"],
            1
        )

    def test_shifts_southern_months_and_aligned_year(self) -> None:
        panel = pd.DataFrame({
            "ID_UC_G0": [1, 2, 2, 2],
            "year": [2000, 2000, 2000, 2000],
            "month": [1, 1, 6, 7],
            "centroid_lat": [10.0, -10.0, -10.0, -10.0]
        })

        aligned = add_local_season_coordinates(panel)

        self.assertEqual(aligned["local_month"].tolist(), [1, 7, 12, 1])
        self.assertEqual(aligned["local_year"].tolist(), [2000, 2000, 2000, 2001])
        self.assertEqual(
            aligned["hemisphere"].tolist(),
            [
                "Northern Hemisphere",
                "Southern Hemisphere",
                "Southern Hemisphere",
                "Southern Hemisphere"
            ]
        )

    def test_local_month_aggregation_uses_aligned_months(self) -> None:
        panel = self.panel.copy()
        panel["centroid_lat"] = panel["ID_UC_G0"].map({1: 10.0, 2: -10.0})

        local_months, _ = aggregate_sensor_local_city_months(
            panel,
            "terra",
            2000,
            2020,
            "median"
        )

        north = local_months.loc[local_months["ID_UC_G0"] == 1].iloc[0]
        south = local_months.loc[local_months["ID_UC_G0"] == 2].iloc[0]
        self.assertEqual(north["month"], 1)
        self.assertEqual(south["month"], 7)
        self.assertEqual(north["hemisphere"], "Northern Hemisphere")
        self.assertEqual(south["hemisphere"], "Southern Hemisphere")

    def test_creates_local_season_figures_by_hemisphere(self) -> None:
        panel = self.panel.copy()
        panel["centroid_lat"] = panel["ID_UC_G0"].map({1: 10.0, 2: -10.0})
        with tempfile.TemporaryDirectory() as temporary_dir:
            outputs = create_local_season_figures(
                panel,
                Path(temporary_dir),
                2000,
                2020,
                "median",
                40
            )
            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(
                path.parent.name == "local_season"
                for path in outputs
            ))
            self.assertTrue(all(
                path.is_file() and path.stat().st_size > 0
                for path in outputs
            ))

    def test_local_aligned_year_panel_uses_rolled_over_year(self) -> None:
        southern_july = self.panel.iloc[[0]].copy()
        southern_july["ID_UC_G0"] = 3
        southern_july["year"] = 2000
        southern_july["month"] = 7
        southern_july["centroid_lat"] = -10.0
        panel = self.panel.copy()
        panel["centroid_lat"] = panel["ID_UC_G0"].map({1: 10.0, 2: -10.0})
        panel = pd.concat([panel, southern_july], ignore_index=True)
        aligned = add_local_season_coordinates(panel)
        aligned["year"] = aligned["local_year"]
        aligned["month"] = aligned["local_month"]
        rows, summary = prepare_sensor_year_month_rows(
            aligned,
            "terra",
            2001,
            2001
        )

        southern = rows.loc[rows["ID_UC_G0"] == 3].iloc[0]
        self.assertEqual(southern["year"], 2001)
        self.assertEqual(southern["month"], 1)
        self.assertEqual(southern["hemisphere"], "Southern Hemisphere")

        with tempfile.TemporaryDirectory() as temporary_dir:
            output = _plot_month_by_year(
                sensor="terra",
                month=1,
                sensor_rows=rows,
                summary=summary,
                output_dir=Path(temporary_dir),
                start_year=2001,
                end_year=2001,
                lower=-105,
                upper=105,
                dpi=40,
                local_season=True
            )
            self.assertTrue(output.is_file())
            self.assertEqual(output.parent.parts[-2:], ("by_year", "terra"))
            self.assertIn("by_aligned_year_2001-2001", output.name)

    def test_creates_a_year_facet_month_figure(self) -> None:
        terra, summary = prepare_sensor_year_month_rows(
            self.panel,
            "terra",
            2000,
            2020
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = _plot_month_by_year(
                sensor="terra",
                month=1,
                sensor_rows=terra,
                summary=summary,
                output_dir=Path(temporary_dir),
                start_year=2000,
                end_year=2020,
                lower=-105,
                upper=105,
                dpi=40
            )
            self.assertTrue(output.is_file())
            self.assertEqual(output.parent.name, "terra")
            self.assertGreater(output.stat().st_size, 0)

    def test_creates_twelve_month_figures_for_each_sensor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            with redirect_stdout(StringIO()):
                outputs = create_year_panel_figures(
                    self.panel,
                    Path(temporary_dir),
                    2000,
                    2000,
                    30
                )

        self.assertEqual(len(outputs), 24)
        self.assertEqual(
            sum(path.parent.name == "terra" for path in outputs),
            12
        )
        self.assertEqual(
            sum(path.parent.name == "aqua" for path in outputs),
            12
        )

    def test_eda_logs_are_written_to_timestamped_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            logger, log_file = configure_eda_logging(Path(temporary_dir))
            try:
                logger.info("synthetic EDA log record")
                for handler in logger.handlers:
                    handler.flush()
                self.assertTrue(log_file.is_file())
                self.assertIn("synthetic EDA log record", log_file.read_text(
                    encoding="utf-8"
                ))
                self.assertRegex(log_file.name, r"^03_eda_\d{8}_\d{6}_\d{6}\.log$")
            finally:
                for handler in tuple(logger.handlers):
                    logger.removeHandler(handler)
                    handler.close()


if __name__ == "__main__":
    unittest.main()
