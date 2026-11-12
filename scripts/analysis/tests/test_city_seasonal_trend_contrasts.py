import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src.city_seasonal_trend_contrasts import (
    build_city_seasonal_trend_contrasts,
)


SLOTS = ("terra_day", "terra_night", "aqua_day", "aqua_night")
SEASONS = ("DJF", "MAM", "JJA", "SON")


def _seasonal_trends():
    rows = []
    city_data = (
        (1, "Cfa", 30.0, "north"),
        (2, "Cfb", -30.0, "south"),
        (3, "Af", 0.0, "equator"),
    )
    season_values = {
        "north": {
            "DJF": (1.0, 0.5),
            "JJA": (3.0, 1.5),
            "MAM": (2.0, 1.0),
            "SON": (2.0, 1.0),
        },
        "south": {
            "DJF": (4.0, 1.5),
            "JJA": (1.0, 0.5),
            "MAM": (2.0, 1.0),
            "SON": (2.0, 1.0),
        },
        "equator": {
            season: (2.0, 1.0) for season in SEASONS
        },
    }
    for city_id, climate, latitude, hemisphere in city_data:
        for season in SEASONS:
            beta_urban, beta_rural = season_values[hemisphere][season]
            for slot in SLOTS:
                if (
                    city_id == 1
                    and slot == "aqua_night"
                    and season == "DJF"
                ):
                    continue
                rows.append(
                    {
                        "ID_UC_G0": city_id,
                        "climate": climate,
                        "centroid_lon": float(city_id),
                        "centroid_lat": latitude,
                        "city_area_km2": 100.0 + city_id,
                        "country": "Country A" if city_id == 1 else "Country B",
                        "GC_DEV_USR": "Region A" if city_id == 1 else "Region B",
                        "hemisphere": hemisphere,
                        "season": season,
                        "slot": slot,
                        "beta_urban": beta_urban,
                        "beta_rural": beta_rural,
                        "beta_suhi": beta_urban - beta_rural,
                        "n_valid_years": 17,
                    }
                )
    return pd.DataFrame(rows)


class CitySeasonalTrendContrastTests(unittest.TestCase):
    def test_builds_hemisphere_aware_city_slot_contrasts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "city_season_slot_trends.parquet"
            connection = duckdb.connect(":memory:")
            connection.register("trends", _seasonal_trends())
            connection.execute(
                f"COPY trends TO '{source.as_posix()}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            connection.close()

            output_dir = root / "contrasts"
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "paths": {
                            "seasonal_trend_file": str(source),
                            "output_dir": str(output_dir),
                        },
                        "decomposition_tolerance": 1e-8,
                    }
                ),
                encoding="utf-8",
            )
            output_path = build_city_seasonal_trend_contrasts(config_path)

            connection = duckdb.connect(":memory:")
            rows = connection.execute(
                """
                SELECT ID_UC_G0, hemisphere, slot, summer_season, winter_season,
                       gamma_suhi, gamma_urban, gamma_rural
                FROM read_parquet(?)
                ORDER BY ID_UC_G0, slot
                """,
                [output_path.as_posix()],
            ).fetchall()
            connection.close()

            self.assertEqual(len(rows), 7)
            north = [row for row in rows if row[0] == 1]
            south = [row for row in rows if row[0] == 2]
            self.assertEqual(len(north), 3)
            self.assertEqual(len(south), 4)
            self.assertTrue(
                all(row[3:5] == ("JJA", "DJF") for row in north)
            )
            self.assertTrue(
                all(row[3:5] == ("DJF", "JJA") for row in south)
            )
            self.assertTrue(
                all(
                    abs(row[5] - (row[6] - row[7])) < 1e-12
                    for row in rows
                )
            )
            self.assertTrue(all(row[5:8] == (1.0, 2.0, 1.0) for row in north))
            self.assertTrue(all(row[5:8] == (2.0, 3.0, 1.0) for row in south))

            connection = duckdb.connect(":memory:")
            metadata_rows = connection.execute(
                """
                SELECT centroid_lon, city_area_km2, country, GC_DEV_USR
                FROM read_parquet(?)
                WHERE ID_UC_G0 = 1 AND slot = 'terra_day'
                """,
                [output_path.as_posix()],
            ).fetchone()
            connection.close()
            self.assertEqual(
                metadata_rows,
                (1.0, 101.0, "Country A", "Region A"),
            )

            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["equator_policy"], "exclude")
            self.assertEqual(manifest["counts"]["equatorial_cities"], 1)
            self.assertEqual(
                manifest["counts"]["excluded_incomplete_city_slots"], 1
            )
            self.assertEqual(manifest["output_rows"], 7)


if __name__ == "__main__":
    unittest.main()
