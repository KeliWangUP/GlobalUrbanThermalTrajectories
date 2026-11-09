import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src.city_season_slot_trend_figures import (
    _filter_trends_by_hemisphere,
    build_city_season_slot_trend_figures,
    summarize_trends,
)


def _trends():
    rows = []
    for city_id, climate, latitude, base in (
        (1, "Af", 20.0, 1.0),
        (2, "Cfa", -20.0, -1.0),
    ):
        for season_index, season in enumerate(("DJF", "MAM", "JJA", "SON")):
            for slot_index, slot in enumerate(
                ("terra_day", "terra_night", "aqua_day", "aqua_night")
            ):
                beta_urban = base + season_index * 0.2 + slot_index * 0.05
                beta_rural = 0.1 + season_index * 0.03
                rows.append(
                    {
                        "ID_UC_G0": city_id,
                        "climate": climate,
                        "centroid_lat": latitude,
                        "hemisphere": "north" if latitude > 0 else "south",
                        "season": season,
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


class CitySeasonSlotTrendFigureTests(unittest.TestCase):
    def test_filters_hemisphere_and_summarizes_all_four_seasons(self):
        frame = _trends()
        north, counts = _filter_trends_by_hemisphere(frame, "north")
        self.assertEqual(set(north["ID_UC_G0"]), {1})
        self.assertEqual(counts["selected_cities"], 1)
        summary = summarize_trends(north, ["A", "C"])
        self.assertEqual(len(summary), 2 * 4 * 4)
        self.assertEqual(
            summary.loc[summary["climate_group"].eq("C"), "n_cities"].sum(),
            0,
        )
        self.assertEqual(
            summary.loc[summary["climate_group"].eq("A"), "n_cities"].sum(),
            16,
        )

    def test_generates_seasonal_figures_and_records_hemisphere(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "city_season_slot_trends.parquet"
            connection = duckdb.connect(":memory:")
            connection.register("trends", _trends())
            connection.execute(
                f"COPY trends TO '{source.as_posix()}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            connection.close()

            output_dir = root / "figures"
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "paths": {
                            "trend_file": str(source),
                            "output_dir": str(output_dir),
                        },
                        "climate_groups": ["A", "C"],
                        "hemisphere": "south",
                        "dpi": 72,
                        "decomposition_tolerance": 1e-8,
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                build_city_season_slot_trend_figures(config_path),
                output_dir,
            )
            self.assertTrue(
                (output_dir / "figure_1_climate_seasonal_suhi_trends.png").is_file()
            )
            self.assertTrue(
                (
                    output_dir
                    / "figure_2_urban_rural_seasonal_decomposition.png"
                ).is_file()
            )
            summary = pd.read_csv(
                output_dir / "climate_season_slot_trend_summary.csv"
            )
            self.assertEqual(
                summary.loc[
                    summary["climate_group"].eq("C"),
                    "n_cities",
                ].sum(),
                16,
            )
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["hemisphere_filter"]["selection"],
                "south",
            )
            self.assertEqual(
                manifest["hemisphere_filter"]["selected_cities"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
