import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src.city_season_subclimate_trend_figures import (
    _select_subclimates,
    build_city_season_subclimate_trend_figures,
    summarize_subclimate_trends,
)


def _trends():
    rows = []
    for city_id, climate, latitude, base in (
        (1, "Cfa", 30.0, 1.0),
        (2, "Cfb", -30.0, -1.0),
        (3, "Dfa", 40.0, 0.5),
    ):
        for season_index, season in enumerate(("DJF", "MAM", "JJA", "SON")):
            for slot_index, slot in enumerate(
                ("terra_day", "terra_night", "aqua_day", "aqua_night")
            ):
                beta_urban = base + season_index * 0.2 + slot_index * 0.03
                beta_rural = 0.1 + season_index * 0.02
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


class CitySeasonSubclimateTrendFigureTests(unittest.TestCase):
    def test_selects_subclimates_within_major_group(self):
        frame = _trends()
        selected, subclasses, counts = _select_subclimates(
            frame, "C", []
        )
        self.assertEqual(subclasses, ["Cfa", "Cfb"])
        self.assertEqual(set(selected["climate"]), {"Cfa", "Cfb"})
        self.assertEqual(counts["selected_cities"], 2)

        selected, subclasses, _ = _select_subclimates(
            frame, "C", ["Cfb"]
        )
        self.assertEqual(subclasses, ["Cfb"])
        self.assertEqual(set(selected["climate"]), {"Cfb"})

    def test_summarizes_each_subclimate_season_and_slot(self):
        selected, subclasses, _ = _select_subclimates(_trends(), "C", [])
        summary = summarize_subclimate_trends(selected, subclasses)
        self.assertEqual(len(summary), 2 * 4 * 4)
        self.assertEqual(set(summary["climate_group"]), {"Cfa", "Cfb"})
        self.assertTrue(summary["n_cities"].eq(1).all())

    def test_generates_subclimate_figures_with_hemisphere_filter(self):
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
                        "major_climate_group": "C",
                        "subclimate_classes": [],
                        "hemisphere": "north",
                        "dpi": 72,
                        "decomposition_tolerance": 1e-8,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                build_city_season_subclimate_trend_figures(config_path),
                output_dir,
            )
            self.assertTrue(
                (output_dir / "figure_1_subclimate_seasonal_suhi_trends.png").is_file()
            )
            self.assertTrue(
                (
                    output_dir
                    / "figure_2_subclimate_urban_rural_decomposition.png"
                ).is_file()
            )
            summary = pd.read_csv(
                output_dir / "subclimate_season_slot_trend_summary.csv"
            )
            self.assertEqual(set(summary["climate_group"]), {"Cfa"})
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["major_climate_group"], "C")
            self.assertEqual(manifest["subclimate_classes"], ["Cfa"])
            self.assertEqual(manifest["hemisphere_filter"]["selection"], "north")


if __name__ == "__main__":
    unittest.main()
