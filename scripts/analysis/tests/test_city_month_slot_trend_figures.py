import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src.city_month_slot_trend_figures import (
    _filter_trends_by_hemisphere,
    _validate_trends,
    build_city_month_slot_trend_figures,
    summarize_trends,
)


def _trends():
    rows = []
    for city_id, climate, base in ((1, "Af", 1.0), (2, "Cfa", -1.0)):
        for month in range(1, 13):
            for slot_index, slot in enumerate(
                ("terra_day", "terra_night", "aqua_day", "aqua_night")
            ):
                beta_urban = base + month * 0.1 + slot_index * 0.05
                beta_rural = 0.2 + month * 0.02
                rows.append(
                    {
                        "ID_UC_G0": city_id,
                        "climate": climate,
                        "centroid_lat": 20.0 if city_id == 1 else -20.0,
                        "hemisphere": "north" if city_id == 1 else "south",
                        "month": month,
                        "slot": slot,
                        "beta_suhi": beta_urban - beta_rural,
                        "beta_urban": beta_urban,
                        "beta_rural": beta_rural,
                        "se_suhi": 0.02,
                        "se_urban": 0.03,
                        "se_rural": 0.01,
                        "n_valid_years": 17,
                    }
                )
    return pd.DataFrame(rows)


def _config(root: Path, source: Path, output: Path) -> Path:
    config = {
        "paths": {
            "trend_file": str(source),
            "output_dir": str(output),
        },
        "climate_groups": ["A", "C"],
        "hemisphere": "all",
        "dpi": 72,
        "decomposition_tolerance": 1e-8,
    }
    path = root / "figures_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


class CityMonthSlotTrendFigureTests(unittest.TestCase):
    def test_hemisphere_filter_selects_requested_cities_and_excludes_equator(self):
        frame = _trends()
        frame.loc[frame["ID_UC_G0"].eq(2), "hemisphere"] = "south"
        north, north_info = _filter_trends_by_hemisphere(frame, "north")
        self.assertEqual(set(north["ID_UC_G0"]), {1})
        self.assertEqual(north_info["selected_cities"], 1)
        self.assertEqual(north_info["excluded_cities"], 1)

        frame.loc[frame["ID_UC_G0"].eq(2), "hemisphere"] = "equator"
        north, north_info = _filter_trends_by_hemisphere(frame, "north")
        self.assertEqual(set(north["ID_UC_G0"]), {1})
        self.assertEqual(north_info["excluded_equator_rows"], 48)

    def test_summary_has_median_iqr_and_city_count(self):
        summary = summarize_trends(_trends(), ["A", "C"])
        january = summary.loc[
            summary["climate_group"].eq("A")
            & summary["month"].eq(1)
            & summary["slot"].eq("terra_day")
        ].iloc[0]
        self.assertEqual(january["n_cities"], 1)
        self.assertAlmostEqual(
            january["median_beta_suhi"],
            january["q25_beta_suhi"],
        )
        self.assertAlmostEqual(
            january["median_beta_suhi"],
            january["q75_beta_suhi"],
        )
        self.assertEqual(len(summary), 2 * 12 * 4)

    def test_builds_both_figures_summary_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "city_month_slot_trends.parquet"
            connection = duckdb.connect(":memory:")
            trends = _trends()
            trends.loc[trends["ID_UC_G0"].eq(2), "climate"] = None
            connection.register("trends", trends)
            connection.execute(
                f"COPY trends TO '{source.as_posix()}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            connection.close()
            output = root / "figures"
            config = _config(root, source, output)

            self.assertEqual(
                build_city_month_slot_trend_figures(config),
                output,
            )
            self.assertTrue(
                (output / "figure_1_climate_monthly_suhi_trends.png").is_file()
            )
            self.assertTrue(
                (
                    output
                    / "figure_2_urban_rural_trend_decomposition.png"
                ).is_file()
            )
            summary = pd.read_csv(
                output / "climate_month_slot_trend_summary.csv"
            )
            self.assertEqual(len(summary), 96)
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertLessEqual(
                manifest["max_absolute_city_level_decomposition_error"],
                1e-8,
            )
            self.assertEqual(
                manifest["climate_grouping_exclusions"]["unclassified_rows"],
                48,
            )
            self.assertEqual(
                manifest["climate_grouping_exclusions"]["unclassified_cities"],
                1,
            )
            self.assertEqual(manifest["hemisphere_filter"]["selection"], "all")
            self.assertEqual(
                manifest["hemisphere_filter"]["source_cities"],
                2,
            )

    def test_builds_figures_for_selected_hemisphere(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "city_month_slot_trends.parquet"
            connection = duckdb.connect(":memory:")
            connection.register("trends", _trends())
            connection.execute(
                f"COPY trends TO '{source.as_posix()}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            connection.close()

            output = root / "north_figures"
            config = _config(root, source, output)
            config_data = json.loads(config.read_text(encoding="utf-8"))
            config_data["hemisphere"] = "north"
            config.write_text(json.dumps(config_data), encoding="utf-8")

            self.assertEqual(
                build_city_month_slot_trend_figures(config),
                output,
            )
            summary = pd.read_csv(
                output / "climate_month_slot_trend_summary.csv"
            )
            self.assertEqual(
                summary.loc[
                    summary["climate_group"].eq("A"),
                    "n_cities",
                ].max(),
                1,
            )
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["hemisphere_filter"]["selection"], "north")
            self.assertEqual(
                manifest["hemisphere_filter"]["selected_cities"],
                1,
            )

    def test_rejects_non_additive_city_level_coefficients(self):
        frame = _trends()
        frame.loc[0, "beta_suhi"] += 0.01
        config = {
            "decomposition_tolerance": 1e-8,
        }
        with self.assertRaisesRegex(ValueError, "decomposition failed"):
            _validate_trends(frame, config)


if __name__ == "__main__":
    unittest.main()
