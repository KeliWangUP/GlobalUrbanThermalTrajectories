import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src.city_seasonal_gamma_subclimate_figures import (
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
                }
            )
    return pd.DataFrame(rows)


class CitySeasonalGammaSubclimateFigureTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
