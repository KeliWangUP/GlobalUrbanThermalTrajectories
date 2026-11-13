import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src.city_seasonal_gamma_subclimate_figures import (
    _publish_staged_outputs,
    _validate_gamma_trends,
    build_city_seasonal_gamma_subclimate_figures,
    summarize_subclimate_gammas,
)


SLOTS = ("terra_day", "terra_night", "aqua_day", "aqua_night")


def _gamma_trends():
    rows = []
    for city_id, climate, hemisphere, gamma_urban, gamma_rural in (
        (1, "Cfa", "north", 2.0, 1.0),
        (2, "Cfb", "south", 3.0, 1.0),
        (3, "Dfa", "north", 1.0, 0.5),
    ):
        for slot in SLOTS:
            rows.append(
                {
                    "ID_UC_G0": city_id,
                    "climate": climate,
                    "GC_DEV_USR": (
                        "Region 1" if hemisphere == "north" else "Region 2"
                    ),
                    "centroid_lat": 30.0 if hemisphere == "north" else -30.0,
                    "hemisphere": hemisphere,
                    "slot": slot,
                    "summer_season": "JJA" if hemisphere == "north" else "DJF",
                    "winter_season": "DJF" if hemisphere == "north" else "JJA",
                    "n_valid_years_summer": 17,
                    "n_valid_years_winter": 17,
                    "gamma_urban": gamma_urban,
                    "gamma_rural": gamma_rural,
                    "gamma_suhi": gamma_urban - gamma_rural,
                    "built_surface_2000_m2": city_id * 1_000_000.0,
                    "built_surface_slope_m2_per_year": (
                        (city_id - 2) * 2_500.0
                    ),
                    "built_volume_total_2000_m3": city_id * 10_000_000.0,
                    "built_volume_total_slope_m3_per_year": (
                        (city_id - 2) * 5_000.0
                    ),
                }
            )
    return pd.DataFrame(rows)


class CitySeasonalGammaSubclimateFigureTests(unittest.TestCase):
    def test_publishing_replaces_generated_files_and_preserves_others(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "figures"
            output_dir.mkdir()
            (output_dir / "figure.png").write_text("old", encoding="utf-8")
            (output_dir / "unrelated.txt").write_text("keep", encoding="utf-8")
            staging = root / "staging"
            staging.mkdir()
            (staging / "figure.png").write_text("new", encoding="utf-8")
            (staging / "manifest.json").write_text("{}", encoding="utf-8")

            _publish_staged_outputs(staging, output_dir)

            self.assertEqual(
                (output_dir / "figure.png").read_text(encoding="utf-8"), "new"
            )
            self.assertEqual(
                (output_dir / "unrelated.txt").read_text(encoding="utf-8"), "keep"
            )
            if os.name == "posix":
                self.assertEqual(
                    stat.S_IMODE(output_dir.stat().st_mode), 0o2775
                )
                self.assertEqual(
                    stat.S_IMODE((output_dir / "figure.png").stat().st_mode),
                    0o664,
                )

    def test_summarizes_subclass_slot_gamma_distributions(self):
        frame = _gamma_trends()
        summary = summarize_subclimate_gammas(frame, ["Cfa", "Cfb"])
        self.assertEqual(len(summary), 2 * 4)
        self.assertEqual(set(summary["climate"]), {"Cfa", "Cfb"})
        self.assertTrue(summary["n_cities"].eq(1).all())
        cfa = summary.loc[
            summary["climate"].eq("Cfa")
            & summary["slot"].eq("terra_day")
        ].iloc[0]
        self.assertEqual(cfa["median_gamma_suhi"], 1.0)
        self.assertEqual(cfa["median_gamma_urban"], 2.0)
        self.assertEqual(cfa["median_gamma_rural"], 1.0)

    def test_validates_ghsl_growth_features(self):
        frame = _gamma_trends()
        self.assertEqual(_validate_gamma_trends(frame, 1e-8), 0.0)

        frame.loc[0, "built_surface_slope_m2_per_year"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite GHSL"):
            _validate_gamma_trends(frame, 1e-8)

    def test_creates_subclimate_gamma_figures_with_hemisphere_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "city_seasonal_trend_contrasts.parquet"
            connection = duckdb.connect(":memory:")
            connection.register("gamma_trends", _gamma_trends())
            connection.execute(
                f"COPY gamma_trends TO '{source.as_posix()}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            connection.close()

            output_dir = root / "figures"
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "paths": {
                            "gamma_file": str(source),
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
                build_city_seasonal_gamma_subclimate_figures(config_path),
                output_dir,
            )
            self.assertTrue(
                (output_dir / "figure_1_subclimate_gamma_suhi.png").is_file()
            )
            self.assertTrue(
                (
                    output_dir
                    / "figure_2_subclimate_gamma_decomposition.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    output_dir
                    / "figure_3_city_seasonal_synchrony_by_subclimate.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    output_dir
                    / "figure_4_city_seasonal_synchrony_by_region.png"
                ).is_file()
            )
            self.assertTrue(
                (output_dir / "figure_5_gamma_vs_built_surface_growth.png").is_file()
            )
            self.assertTrue(
                (output_dir / "figure_6_gamma_vs_built_volume_growth.png").is_file()
            )
            summary = pd.read_csv(
                output_dir / "subclimate_seasonal_gamma_summary.csv"
            )
            self.assertEqual(set(summary["climate"]), {"Cfa"})
            self.assertTrue(summary["n_cities"].eq(1).all())
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["major_climate_group"], "C")
            self.assertEqual(manifest["subclimate_classes"], ["Cfa"])
            self.assertEqual(manifest["hemisphere_filter"]["selection"], "north")
            self.assertEqual(
                manifest["synchrony_scatter"]["x"], "gamma_rural"
            )
            self.assertEqual(
                manifest["synchrony_scatter"]["y"], "gamma_urban"
            )
            self.assertEqual(
                manifest["building_growth_scatter"]["y_metrics"],
                ["gamma_urban", "gamma_rural", "gamma_suhi"],
            )
            self.assertEqual(
                manifest["building_growth_scatter"]["x_metrics"][
                    "built_surface"
                ]["source_column"],
                "built_surface_slope_m2_per_year",
            )


if __name__ == "__main__":
    unittest.main()
