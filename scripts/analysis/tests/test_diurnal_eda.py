import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba

from src.diurnal_eda import (
    LATITUDE_GROUP_ORDER,
    add_local_season_coordinates,
    aggregate_sensor_city_months,
    aggregate_sensor_local_city_months,
    prepare_sensor_year_month_rows
)
from src.eda_logging import configure_eda_logging
from src.eda_city_slopes import calculate_city_month_slopes
from src.eda_climate_month_trends import summarize_climate_month_slopes
from src.eda_trends import annual_sensor_trend_summary
from src.eda_visualizations.calendar_month import (
    create_calendar_month_figures
)
from src.eda_visualizations.city_slopes import (
    CLIMATE_GROUP_COLORS,
    UNCLASSIFIED_COLOR,
    _create_city_slope_grid,
    create_city_slope_figures
)
from src.eda_visualizations.climate_month_trends import (
    create_climate_month_trend_heatmaps
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
from src.eda_visualizations.yearly_trends import (
    create_yearly_trend_figures
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

    def test_annual_summary_uses_stored_city_asymmetry(self) -> None:
        panel = pd.DataFrame({
            "ID_UC_G0": [1, 2, 3],
            "year": [2000] * 3,
            "month": [1] * 3,
            "centroid_lat": [20.0] * 3,
            "suhi_terra_day_mean": [0.0, 10.0, 100.0],
            "suhi_terra_night_mean": [100.0, 0.0, 10.0],
            "A_T": [-100.0, 10.0, 90.0],
            "M_T": [50.0, 5.0, 55.0],
            "qc_pass_A_T": [True] * 3
        })

        summary = annual_sensor_trend_summary(
            panel,
            "terra",
            2000,
            2000,
            10,
            100,
            0.95,
            42
        )
        row = summary.loc[
            summary["latitude_group"].eq("Northern Hemisphere")
            & summary["month"].eq(1)
        ].iloc[0]

        self.assertEqual(row["n_cities"], 3)
        self.assertEqual(row["median_day"], 10.0)
        self.assertEqual(row["median_night"], 10.0)
        self.assertEqual(row["median_asymmetry"], 10.0)
        self.assertEqual(row["median_mean"], 50.0)
        self.assertLessEqual(row["ci_lower_asymmetry"], 10.0)
        self.assertGreaterEqual(row["ci_upper_asymmetry"], 10.0)

    def test_yearly_trend_figures_create_sensor_and_latitude_groups(self) -> None:
        panel = pd.DataFrame({
            "ID_UC_G0": [1, 2, 3],
            "year": [2000] * 3,
            "month": [1] * 3,
            "centroid_lat": [20.0, -20.0, 0.0],
            "suhi_terra_day_mean": [2.0, 3.0, 4.0],
            "suhi_terra_night_mean": [1.0, 2.0, 2.0],
            "A_T": [1.0, 1.0, 2.0],
            "M_T": [1.5, 2.5, 3.0],
            "qc_pass_A_T": [True] * 3,
            "suhi_aqua_day_mean": [3.0, 4.0, 5.0],
            "suhi_aqua_night_mean": [1.0, 2.0, 3.0],
            "A_A": [2.0, 2.0, 2.0],
            "M_A": [2.0, 3.0, 4.0],
            "qc_pass_A_A": [True] * 3
        })

        with tempfile.TemporaryDirectory() as temporary_dir:
            outputs = create_yearly_trend_figures(
                panel,
                Path(temporary_dir),
                2000,
                2000,
                10,
                100,
                0.95,
                42,
                30
            )
            self.assertEqual(len(outputs), 6)
            self.assertTrue(all(path.is_file() for path in outputs))
            self.assertEqual(
                {path.parent.name for path in outputs},
                {
                    "northern_hemisphere",
                    "southern_hemisphere",
                    "equatorial_belt"
                }
            )

    def test_city_month_slopes_use_qc_years_and_are_scaled_per_decade(
        self
    ) -> None:
        records = []
        for city_id, latitude, valid_year_count in (
            (1, 20.0, 18),
            (2, -20.0, 18),
            (3, 0.0, 14)
        ):
            for year in range(2000, 2000 + valid_year_count):
                records.append({
                    "ID_UC_G0": city_id,
                    "year": year,
                    "month": 1,
                    "centroid_lat": latitude,
                    "koppen_geiger_macro_group": {
                        1: "Tropical",
                        2: "Arid",
                        3: "Temperate"
                    }[city_id],
                    "suhi_terra_day_mean": (
                        city_id + 0.2 * (year - 2000)
                    ),
                    "suhi_terra_night_mean": (
                        city_id - 0.1 * (year - 2000)
                    ),
                    "qc_pass_A_T": not (city_id == 1 and year == 2001)
                })

        slopes, coverage = calculate_city_month_slopes(
            pd.DataFrame(records),
            "terra",
            2000,
            2017,
            10,
            15,
            ("Tropical", "Arid", "Temperate")
        )

        self.assertEqual(len(slopes), 2)
        self.assertEqual(
            set(slopes["latitude_group"]),
            {"Northern Hemisphere", "Southern Hemisphere"}
        )
        self.assertEqual(
            set(slopes["koppen_geiger_macro_group"]),
            {"Tropical", "Arid"}
        )
        self.assertEqual(
            slopes.set_index("ID_UC_G0")[
                "koppen_geiger_macro_group"
            ].to_dict(),
            {1: "Tropical", 2: "Arid"}
        )
        self.assertTrue((slopes["n_valid_years"] >= 15).all())
        self.assertTrue(
            (slopes["beta_day_per_decade"].sub(2.0).abs() < 1e-10).all()
        )
        self.assertTrue(
            (slopes["beta_night_per_decade"].add(1.0).abs() < 1e-10).all()
        )
        equator_january = coverage.loc[
            coverage["latitude_group"].eq("Equatorial Belt")
            & coverage["month"].eq(1)
        ].iloc[0]
        self.assertEqual(equator_january["n_cities_with_qc_data"], 1)
        self.assertEqual(equator_january["n_cities_plotted"], 0)
        self.assertEqual(equator_january["n_cities_below_minimum"], 1)

    def test_city_slope_figures_create_separate_month_grids(self) -> None:
        records = []
        for city_id, latitude in ((1, 20.0), (2, -20.0), (3, 0.0)):
            for year in range(2000, 2015):
                record = {
                    "ID_UC_G0": city_id,
                    "year": year,
                    "month": 1,
                    "centroid_lat": latitude,
                    "koppen_geiger_macro_group": {
                        1: "Tropical",
                        2: "Arid",
                        3: "Temperate"
                    }[city_id],
                    "qc_pass_A_T": True,
                    "qc_pass_A_A": True
                }
                for sensor, offset in (("terra", 0.0), ("aqua", 1.0)):
                    record[f"suhi_{sensor}_day_mean"] = (
                        offset + 0.2 * (year - 2000)
                    )
                    record[f"suhi_{sensor}_night_mean"] = (
                        offset - 0.1 * (year - 2000)
                    )
                records.append(record)

        with tempfile.TemporaryDirectory() as temporary_dir:
            outputs = create_city_slope_figures(
                panel=pd.DataFrame(records),
                output_dir=Path(temporary_dir),
                start_year=2000,
                end_year=2014,
                equatorial_band_degrees=10,
                minimum_years=15,
                climate_groups=("Tropical", "Arid", "Temperate"),
                dpi=30
            )

            self.assertEqual(len(outputs), 6)
            self.assertTrue(all(path.is_file() for path in outputs))
            self.assertEqual(
                {path.parent.name for path in outputs},
                {
                    "northern_hemisphere",
                    "southern_hemisphere",
                    "equatorial_belt"
                }
            )

    def test_city_slope_scatter_colors_points_by_climate_group(self) -> None:
        slopes = pd.DataFrame({
            "ID_UC_G0": [1, 2, 3],
            "latitude_group": ["Northern Hemisphere"] * 3,
            "koppen_geiger_macro_group": ["Tropical", "Arid", None],
            "month": [1, 1, 1],
            "beta_day_per_decade": [-0.2, 0.1, 0.3],
            "beta_night_per_decade": [-0.1, 0.2, 0.4]
        })
        coverage = pd.DataFrame({
            "latitude_group": ["Northern Hemisphere"] * 12,
            "month": range(1, 13),
            "n_cities_plotted": [3] + [0] * 11,
            "n_cities_with_qc_data": [3] + [0] * 11,
            "n_cities_below_minimum": [0] * 12
        })
        figure = _create_city_slope_grid(
            slopes=slopes,
            coverage=coverage,
            sensor="terra",
            latitude_group="Northern Hemisphere",
            climate_groups=("Tropical", "Arid"),
            start_year=2000,
            end_year=2014,
            minimum_years=15,
            lower=-1,
            upper=1
        )
        try:
            collections = figure.axes[0].collections
            self.assertEqual(len(collections), 3)
            expected_colors = [
                to_rgba(CLIMATE_GROUP_COLORS[0], alpha=0.6),
                to_rgba(CLIMATE_GROUP_COLORS[1], alpha=0.6),
                to_rgba(UNCLASSIFIED_COLOR, alpha=0.6)
            ]
            actual_colors = [
                tuple(collection.get_facecolors()[0])
                for collection in collections
            ]
            for actual, expected in zip(actual_colors, expected_colors):
                self.assertEqual(actual, expected)
            legend_labels = {
                text.get_text() for text in figure.legends[0].get_texts()
            }
            self.assertTrue(
                {"Tropical", "Arid", "Unclassified"} <= legend_labels
            )
        finally:
            plt.close(figure)

    def test_climate_month_summary_aligns_south_and_summarizes_beta_a(
        self
    ) -> None:
        records = []
        for city_id, latitude, source_month, beta_day, beta_night in (
            (1, 20.0, 1, 1.0, 2.0),
            (2, -20.0, 7, 3.0, 4.0)
        ):
            for year in range(2000, 2015):
                elapsed = year - 2000
                day = city_id + beta_day * elapsed / 10
                night = city_id + beta_night * elapsed / 10
                records.append({
                    "ID_UC_G0": city_id,
                    "year": year,
                    "month": source_month,
                    "centroid_lat": latitude,
                    "koppen_geiger_macro_group": "Tropical",
                    "suhi_terra_day_mean": day,
                    "suhi_terra_night_mean": night,
                    "A_T": day - night,
                    "qc_pass_A_T": True
                })

        summary = summarize_climate_month_slopes(
            pd.DataFrame(records),
            "terra",
            2000,
            2014,
            10,
            15,
            ("Tropical", "Arid"),
            100,
            0.95,
            42
        )
        tropical_january = summary.loc[
            summary["koppen_geiger_macro_group"].eq("Tropical")
            & summary["month"].eq(1)
        ].iloc[0]

        self.assertEqual(tropical_january["n_cities"], 2)
        self.assertAlmostEqual(tropical_january["median_beta_day"], 2.0)
        self.assertAlmostEqual(tropical_january["median_beta_night"], 3.0)
        self.assertAlmostEqual(tropical_january["median_beta_asymmetry"], -1.0)
        self.assertLessEqual(
            tropical_january["ci_lower_beta_asymmetry"],
            -1.0
        )
        self.assertGreaterEqual(
            tropical_january["ci_upper_beta_asymmetry"],
            -1.0
        )
        arid_january = summary.loc[
            summary["koppen_geiger_macro_group"].eq("Arid")
            & summary["month"].eq(1)
        ].iloc[0]
        self.assertEqual(arid_january["n_cities"], 0)

    def test_climate_month_heatmaps_write_six_figures_and_statistics(
        self
    ) -> None:
        records = []
        climate_groups = ("Tropical", "Arid")
        for city_id, latitude, climate_group, source_month in (
            (1, 20.0, "Tropical", 1),
            (2, -20.0, "Arid", 7)
        ):
            for year in range(2000, 2015):
                elapsed = year - 2000
                record = {
                    "ID_UC_G0": city_id,
                    "year": year,
                    "month": source_month,
                    "centroid_lat": latitude,
                    "koppen_geiger_macro_group": climate_group,
                    "qc_pass_A_T": True,
                    "qc_pass_A_A": True
                }
                for sensor, offset in (("terra", 0.0), ("aqua", 1.0)):
                    day = offset + 0.1 * elapsed
                    night = offset - 0.05 * elapsed
                    record[f"suhi_{sensor}_day_mean"] = day
                    record[f"suhi_{sensor}_night_mean"] = night
                    record["A_T" if sensor == "terra" else "A_A"] = (
                        day - night
                    )
                records.append(record)

        with tempfile.TemporaryDirectory() as temporary_dir:
            outputs = create_climate_month_trend_heatmaps(
                panel=pd.DataFrame(records),
                output_dir=Path(temporary_dir),
                start_year=2000,
                end_year=2014,
                equatorial_band_degrees=10,
                minimum_years=15,
                climate_groups=climate_groups,
                bootstrap_replicates=100,
                confidence_level=0.95,
                random_seed=42,
                dpi=30
            )
            statistics_path = (
                Path(temporary_dir)
                / "climate_month_trend_heatmaps"
                / "climate_month_D_N_A_trend_statistics_2000-2014.csv"
            )
            self.assertEqual(len(outputs), 6)
            self.assertTrue(all(path.is_file() for path in outputs))
            self.assertTrue(statistics_path.is_file())
            statistics = pd.read_csv(statistics_path)
            self.assertEqual(set(statistics["sensor"]), {"terra", "aqua"})
            self.assertEqual(set(statistics["confidence_level"]), {0.95})
            self.assertEqual(set(statistics["bootstrap_replicates"]), {100})
            self.assertEqual(
                set(statistics["statistic"]),
                {
                    "median_beta_day",
                    "ci_lower_beta_day",
                    "ci_upper_beta_day",
                    "median_beta_night",
                    "ci_lower_beta_night",
                    "ci_upper_beta_night",
                    "median_beta_asymmetry",
                    "ci_lower_beta_asymmetry",
                    "ci_upper_beta_asymmetry"
                }
            )

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
