import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src.paired_month_city_trends import (
    build_paired_month_city_trends,
    calculate_paired_month_deltas,
)


def _trend_rows():
    rows = []
    values = {
        1: {"climate": "Dfc", 2: 1.0, 7: 3.0},
        2: {"climate": "Dfb", 2: 2.0, 7: 1.0},
        3: {"climate": "Dfa", 2: 4.0, 7: 4.0},
        4: {"climate": "Cfa", 4: -1.0, 7: 2.0},
        5: {"climate": "Cfb", 4: 2.0, 7: 1.0},
        6: {"climate": "Csa", 4: 0.0, 7: 2.0},
        7: {"climate": "Af", 2: 9.0, 7: 100.0, 4: 9.0},
    }
    for city_id, city_data in values.items():
        for month in (2, 4, 7):
            if month not in city_data:
                continue
            for slot in ("terra_day", "aqua_day"):
                rows.append(
                    {
                        "ID_UC_G0": city_id,
                        "climate": city_data["climate"],
                        "month": month,
                        "slot": slot,
                        "beta_suhi": city_data[month],
                    }
                )
    return pd.DataFrame(rows)


class PairedMonthCityTrendTests(unittest.TestCase):
    def test_pairs_same_climate_cities_and_calculates_delta_statistics(self):
        paired, summary = calculate_paired_month_deltas(_trend_rows())
        self.assertEqual(len(summary), 4)
        d_terra = summary.loc[
            summary["climate_group"].eq("D")
            & summary["slot"].eq("terra_day")
        ].iloc[0]
        self.assertEqual(d_terra["n_paired_cities"], 3)
        self.assertAlmostEqual(d_terra["median_delta_beta_suhi"], 0.0)
        self.assertEqual(d_terra["n_positive"], 1)
        self.assertEqual(d_terra["n_zero"], 1)
        self.assertEqual(d_terra["n_negative"], 1)
        self.assertAlmostEqual(d_terra["positive_fraction"], 1 / 3)

        c_aqua = summary.loc[
            summary["climate_group"].eq("C")
            & summary["slot"].eq("aqua_day")
        ].iloc[0]
        self.assertEqual(c_aqua["n_paired_cities"], 3)
        self.assertAlmostEqual(c_aqua["median_delta_beta_suhi"], 2.0)
        self.assertEqual(c_aqua["n_positive"], 2)
        self.assertEqual(c_aqua["positive_fraction"], 2 / 3)
        self.assertNotIn(7, set(paired["ID_UC_G0"]))

    def test_writes_summary_paired_rows_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "city_month_slot_trends.parquet"
            connection = duckdb.connect(":memory:")
            connection.register("trends", _trend_rows())
            connection.execute(
                f"COPY trends TO '{source.as_posix()}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            connection.close()
            output_dir = root / "paired_result"
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "paths": {
                            "trend_file": str(source),
                            "output_dir": str(output_dir),
                        }
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                build_paired_month_city_trends(config_path),
                output_dir,
            )
            summary = pd.read_csv(
                output_dir / "paired_month_delta_summary.csv"
            )
            paired = pd.read_csv(
                output_dir / "paired_month_delta_city_level.csv"
            )
            self.assertEqual(len(summary), 4)
            self.assertEqual(len(paired), 12)
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["delta_definition"],
                "later month beta_suhi minus earlier month beta_suhi",
            )


if __name__ == "__main__":
    unittest.main()
