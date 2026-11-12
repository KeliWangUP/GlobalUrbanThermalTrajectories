import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src.city_annual_slot_trends import build_city_annual_slot_trends
from src.city_annual_subclimate_trend_figures import (
    build_city_annual_subclimate_trend_figures,
    summarize_annual_subclimates,
)


SLOTS = ("terra_day", "terra_night", "aqua_day", "aqua_night")


def _annual_source(path: Path) -> None:
    rows = []
    for city_id, climate, latitude, urban_slope in (
        (1, "Cfa", 30.0, 0.2),
        (2, "Cfb", -30.0, 0.3),
        (3, "Dfa", 40.0, 0.1),
    ):
        for year in range(2003, 2020):
            row = {
                "ID_UC_G0": city_id,
                "year": year,
                "centroid_lat": latitude,
                "koppen_geiger_class": climate,
            }
            urban = 20 + urban_slope * (year - 2003)
            rural = 10 + 0.1 * (year - 2003)
            for slot in SLOTS:
                row[f"qc_pass_year_{slot}"] = not (
                    city_id == 1 and slot == "terra_day" and year == 2007
                )
                row[f"urban_{slot}_mean"] = urban
                row[f"rural_{slot}_mean"] = rural
                row[f"suhi_{slot}_mean"] = urban - rural
            rows.append(row)

    connection = duckdb.connect(":memory:")
    connection.register("annual_rows", pd.DataFrame(rows))
    connection.execute(
        f"COPY annual_rows TO '{path.as_posix()}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.close()


class CityAnnualSubclimateTrendFigureTests(unittest.TestCase):
    def test_summarizes_subclass_slot_metrics(self):
        frame = pd.DataFrame(
            [
                {
                    "ID_UC_G0": 1,
                    "climate": "Cfa",
                    "slot": "terra_day",
                    "beta_urban": 2.0,
                    "beta_rural": 1.0,
                    "beta_suhi": 1.0,
                },
                {
                    "ID_UC_G0": 2,
                    "climate": "Cfa",
                    "slot": "terra_day",
                    "beta_urban": 4.0,
                    "beta_rural": 1.0,
                    "beta_suhi": 3.0,
                },
            ]
        )
        summary = summarize_annual_subclimates(frame, ["Cfa"])
        terra_day = summary.loc[summary["slot"].eq("terra_day")].iloc[0]
        self.assertEqual(terra_day["n_cities"], 2)
        self.assertEqual(terra_day["median_beta_suhi"], 2.0)
        self.assertEqual(terra_day["q25_beta_suhi"], 1.5)
        self.assertEqual(terra_day["q75_beta_suhi"], 2.5)

    def test_builds_annual_trends_and_figures_with_hemisphere_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annual_source = root / "suhi_annual.parquet"
            _annual_source(annual_source)

            trend_dir = root / "gold"
            trend_config = root / "trend_config.json"
            trend_config.write_text(
                json.dumps(
                    {
                        "paths": {
                            "annual_suhi_file": str(annual_source),
                            "output_dir": str(trend_dir),
                        },
                        "years": {
                            "start": 2003,
                            "end": 2019,
                            "minimum_valid_years": 15,
                        },
                        "beta_scale_years": 10,
                    }
                ),
                encoding="utf-8",
            )
            trend_path = build_city_annual_slot_trends(trend_config)

            output_dir = root / "figures"
            figure_config = root / "figure_config.json"
            figure_config.write_text(
                json.dumps(
                    {
                        "paths": {
                            "trend_file": str(trend_path),
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
                build_city_annual_subclimate_trend_figures(figure_config),
                output_dir,
            )
            connection = duckdb.connect(":memory:")
            row = connection.execute(
                """
                SELECT beta_suhi, beta_urban, beta_rural, n_valid_years
                FROM read_parquet(?)
                WHERE ID_UC_G0 = 1 AND slot = 'terra_day'
                """,
                [trend_path.as_posix()],
            ).fetchone()
            connection.close()
            self.assertAlmostEqual(row[0], 1.0)
            self.assertAlmostEqual(row[1], 2.0)
            self.assertAlmostEqual(row[2], 1.0)
            self.assertEqual(row[3], 16)

            self.assertTrue(
                (output_dir / "figure_1_annual_subclimate_suhi_trends.png").is_file()
            )
            self.assertTrue(
                (
                    output_dir
                    / "figure_2_annual_subclimate_urban_rural_decomposition.png"
                ).is_file()
            )
            summary = pd.read_csv(
                output_dir / "annual_subclimate_trend_summary.csv"
            )
            self.assertEqual(set(summary["climate"]), {"Cfa"})
            self.assertTrue(summary["n_cities"].eq(1).all())
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["major_climate_group"], "C")
            self.assertEqual(manifest["subclimate_classes"], ["Cfa"])
            self.assertEqual(manifest["hemisphere_filter"]["selection"], "north")


if __name__ == "__main__":
    unittest.main()
