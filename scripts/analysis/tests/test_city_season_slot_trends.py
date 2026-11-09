import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src.city_season_slot_trends import build_city_season_slot_trends


SLOTS = ("terra_day", "terra_night", "aqua_day", "aqua_night")
SEASONS = ("DJF", "MAM", "JJA", "SON")


def _write_seasonal_source(path: Path) -> None:
    rows = []
    for city_id, climate, latitude in (
        (1, "Af", 20.0),
        (2, "Cfa", -20.0),
    ):
        for year in range(2003, 2020):
            for season in SEASONS:
                row = {
                    "ID_UC_G0": city_id,
                    "season_year": year,
                    "season": season,
                    "centroid_lat": latitude,
                    "koppen_geiger_class": climate,
                }
                for slot_index, slot in enumerate(SLOTS):
                    passed = year != 2007
                    if slot == "aqua_day" and year < 2006:
                        passed = False
                    urban = 20 + 0.2 * (year - 2003) + slot_index * 0.01
                    rural = 10 + 0.1 * (year - 2003)
                    if not passed:
                        urban += 1000
                    row[f"qc_pass_season_{slot}"] = passed
                    row[f"urban_{slot}_mean"] = urban
                    row[f"rural_{slot}_mean"] = rural
                    row[f"suhi_{slot}_mean"] = urban - rural
                rows.append(row)

    connection = duckdb.connect(":memory:")
    connection.register("seasonal_rows", pd.DataFrame(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"COPY seasonal_rows TO '{path.as_posix()}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.close()


class CitySeasonSlotTrendTests(unittest.TestCase):
    def test_builds_qc_filtered_seasonal_trends(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "suhi_seasonal.parquet"
            _write_seasonal_source(source)
            output_dir = root / "gold"
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "paths": {
                            "seasonal_suhi_file": str(source),
                            "output_dir": str(output_dir),
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

            output_path = build_city_season_slot_trends(config_path)
            connection = duckdb.connect(":memory:")
            row = connection.execute(
                """
                SELECT climate, hemisphere, season, slot,
                       beta_suhi, beta_urban, beta_rural,
                       se_suhi, n_valid_years
                FROM read_parquet(?)
                WHERE ID_UC_G0 = 1 AND season = 'DJF' AND slot = 'terra_day'
                """,
                [output_path.as_posix()],
            ).fetchone()
            missing_qc_fit_count = connection.execute(
                """
                SELECT count(*)
                FROM read_parquet(?)
                WHERE slot = 'aqua_day'
                """,
                [output_path.as_posix()],
            ).fetchone()[0]
            connection.close()

            self.assertEqual(row[:4], ("Af", "north", "DJF", "terra_day"))
            self.assertAlmostEqual(row[4], 1.0)
            self.assertAlmostEqual(row[5], 2.0)
            self.assertAlmostEqual(row[6], 1.0)
            self.assertAlmostEqual(row[7], 0.0, places=10)
            self.assertEqual(row[8], 16)
            self.assertEqual(missing_qc_fit_count, 0)
            self.assertTrue((output_dir / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
