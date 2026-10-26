import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.diurnal_eda import (
    LATITUDE_GROUP_ORDER,
    add_local_season_coordinates,
    aggregate_sensor_city_months,
    aggregate_sensor_local_city_months,
    prepare_sensor_year_month_rows
)
from src.eda_logging import configure_eda_logging
from src.eda_visualizations.calendar_month import (
    create_calendar_month_figures
)
from src.eda_visualizations.common import (
    create_year_grid,
    symmetric_axis_limits
)
from src.eda_visualizations.local_season import (
    create_local_season_month_figures,
    create_local_season_year_figures
)
from src.eda_visualizations.year_panels import (
    create_calendar_year_figures
)


class DiurnalEdaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.panel = pd.DataFrame(
            [
                {
                    "ID_UC_G0": 1,
                    "year": 2000,
                    "month": 1,
                    "centroid_lat": 20.0,
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
                    "centroid_lat": 20.0,
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
                    "centroid_lat": 20.0,
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
                    "centroid_lat": -20.0,
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
        terra, summary = aggregate_sensor_city_months(
            self.panel,
            "terra",
            2000,
            2020,
            "median"
        )
        city_one = terra.loc[terra["ID_UC_G0"] == 1].iloc[0]
        self.assertEqual(city_one["day_suhi"], 3.0)
        self.assertEqual(city_one["night_suhi"], 2.0)
        self.assertEqual(city_one["n_valid_years"], 2)
        self.assertEqual(summary.loc[1, "qc_pass_city_year_months"], 3)
        self.assertEqual(summary.loc[1, "plotted_cities"], 2)

    def test_shifts_southern_months_and_aligned_year(self) -> None:
        panel = pd.DataFrame({
            "ID_UC_G0": [1, 2, 2, 2],
            "year": [2000, 2000, 2000, 2000],
            "month": [1, 1, 6, 7],
            "centroid_lat": [20.0, -20.0, -20.0, -20.0]
        })

        aligned = add_local_season_coordinates(panel, 10)

        self.assertEqual(aligned["local_month"].tolist(), [1, 7, 12, 1])
        self.assertEqual(aligned["local_year"].tolist(), [2000, 2000, 2000, 2001])
        self.assertEqual(
            aligned["latitude_group"].tolist(),
            [
                "Northern Hemisphere",
                "Southern Hemisphere",
                "Southern Hemisphere",
                "Southern Hemisphere"
            ]
        )

    def test_local_month_aggregation_keeps_hemisphere_and_month(self) -> None:
        local, _ = aggregate_sensor_local_city_months(
            self.panel,
            "terra",
            2000,
            2020,
            "median",
            10
        )
        north = local.loc[local["ID_UC_G0"] == 1].iloc[0]
        south = local.loc[local["ID_UC_G0"] == 2].iloc[0]
        self.assertEqual(north["month"], 1)
        self.assertEqual(south["month"], 7)
        self.assertEqual(north["latitude_group"], "Northern Hemisphere")
        self.assertEqual(south["latitude_group"], "Southern Hemisphere")

    def test_latitudes_within_ten_degrees_are_equatorial(self) -> None:
        panel = pd.DataFrame({
            "ID_UC_G0": [1, 2, 3, 4, 5],
            "year": [2000] * 5,
            "month": [1] * 5,
            "centroid_lat": [10.0, -10.0, 10.1, -10.1, 0.0]
        })

        aligned = add_local_season_coordinates(panel, 10)

        self.assertEqual(
            aligned["latitude_group"].tolist(),
            [
                "Equatorial Belt",
                "Equatorial Belt",
                "Northern Hemisphere",
                "Southern Hemisphere",
                "Equatorial Belt"
            ]
        )
        self.assertEqual(aligned["local_month"].tolist(), [1, 1, 1, 7, 1])

    def test_axis_limits_are_symmetric_around_zero(self) -> None:
        frame = pd.DataFrame({
            "day_suhi": [-5.0],
            "night_suhi": [3.0]
        })
        self.assertEqual(symmetric_axis_limits([frame]), (-5.25, 5.25))

    def test_creates_calendar_month_figures_for_both_sensors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            outputs = create_calendar_month_figures(
                self.panel,
                Path(temporary_dir),
                2000,
                2020,
                "median",
                40
            )
            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(path.is_file() for path in outputs))

    def test_creates_year_facet_grid(self) -> None:
        rows, summary = prepare_sensor_year_month_rows(
            self.panel,
            "terra",
            2000,
            2020
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            figure = create_year_grid(
                rows=rows,
                summary=summary,
                sensor="terra",
                month=1,
                start_year=2000,
                end_year=2020,
                lower=-105,
                upper=105,
                title="Test",
                x_label="Daytime SUHI"
            )
            output = Path(temporary_dir) / "panels.png"
            figure.savefig(output, dpi=40)
            plt.close(figure)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_creates_calendar_year_month_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            outputs = create_calendar_year_figures(
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

    def test_local_figures_are_separated_by_latitude_group_and_sensor(self) -> None:
        panel = self.panel.copy()
        equator_row = panel.iloc[[0]].copy()
        equator_row["ID_UC_G0"] = 3
        equator_row["centroid_lat"] = 0.0
        panel = pd.concat([panel, equator_row], ignore_index=True)
        with tempfile.TemporaryDirectory() as temporary_dir:
            outputs = create_local_season_month_figures(
                panel,
                Path(temporary_dir),
                2000,
                2020,
                "median",
                40,
                10
            )
            self.assertEqual(len(outputs), 6)
            self.assertEqual(
                {path.parent.name for path in outputs},
                {
                    group.lower().replace(" ", "_")
                    for group in LATITUDE_GROUP_ORDER
                }
            )
            self.assertEqual(
                sum(path.name.startswith("terra_") for path in outputs),
                3
            )
            self.assertTrue(all(path.is_file() for path in outputs))

    def test_local_year_panels_are_separated_by_three_groups(self) -> None:
        panel = self.panel.copy()
        equator_row = panel.iloc[[0]].copy()
        equator_row["ID_UC_G0"] = 3
        equator_row["centroid_lat"] = 0.0
        southern_july = panel.loc[
            (panel["ID_UC_G0"] == 2)
            & (panel["year"] == 2000)
        ].copy()
        southern_july["month"] = 7
        panel = pd.concat(
            [panel, equator_row, southern_july],
            ignore_index=True
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            outputs = create_local_season_year_figures(
                panel,
                Path(temporary_dir),
                2000,
                2001,
                30,
                10
            )
            self.assertEqual(len(outputs), 72)
            self.assertEqual(
                {path.parent.name for path in outputs},
                {
                    group.lower().replace(" ", "_")
                    for group in LATITUDE_GROUP_ORDER
                }
            )
            self.assertTrue(all(path.is_file() for path in outputs))

    def test_eda_logs_are_written_to_timestamped_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            logger, log_file = configure_eda_logging(Path(temporary_dir))
            try:
                logger.info("synthetic EDA log record")
                for handler in logger.handlers:
                    handler.flush()
                self.assertTrue(log_file.is_file())
                self.assertIn(
                    "synthetic EDA log record",
                    log_file.read_text(encoding="utf-8")
                )
                self.assertRegex(
                    log_file.name,
                    r"^03_eda_\d{8}_\d{6}_\d{6}\.log$"
                )
            finally:
                for handler in tuple(logger.handlers):
                    logger.removeHandler(handler)
                    handler.close()


if __name__ == "__main__":
    unittest.main()
