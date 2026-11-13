import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from diagnose_city_pixel_zone_counts import build_city_pixel_zone_counts


class CityPixelZoneCountTests(unittest.TestCase):
    def test_writes_total_and_water_eligible_counts_per_city(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "city_pixel_classification.parquet"
            connection = duckdb.connect(":memory:")
            connection.execute(
                f"""
                COPY (
                    SELECT * FROM (VALUES
                        (1, 'p1', 'stable', TRUE),
                        (1, 'p2', 'development', TRUE),
                        (1, 'p3', 'development', FALSE),
                        (2, 'p4', 'stable', FALSE),
                        (2, 'p5', 'stable', TRUE)
                    ) AS pixels(ID_UC_G0, pixel_id, urban_zone, water_eligible)
                ) TO '{source.as_posix()}' (FORMAT PARQUET)
                """
            )
            connection.close()

            gold_manifest = root / "gold_manifest.json"
            gold_manifest.write_text(
                json.dumps(
                    {
                        "city_count": 2,
                        "urban_classification": {
                            "baseline_year": 2000,
                            "stable_isa_threshold": 0.7,
                            "stable_label": "stable",
                            "development_label": "development",
                        },
                    }
                ),
                encoding="utf-8",
            )
            metadata = root / "city_metadata.parquet"
            connection = duckdb.connect(":memory:")
            connection.execute(
                f"""
                COPY (
                    SELECT * FROM (VALUES
                        (1, -70.0, 40.0, 120.5, 'Country A', 'Region A',
                         14, 'Cfa', 'Temperate'),
                        (2, 20.0, -30.0, 80.0, 'Country B', 'Region B',
                         4, 'BWh', 'Arid')
                    ) AS cities(
                        ID_UC_G0, centroid_lon, centroid_lat, city_area_km2,
                        country, GC_DEV_USR, koppen_geiger_code,
                        koppen_geiger_class, koppen_geiger_macro_group
                    )
                ) TO '{metadata.as_posix()}' (FORMAT PARQUET)
                """
            )
            connection.close()
            output_csv = root / "counts.csv"
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "paths": {
                            "pixel_classification": str(source),
                            "gold_manifest": str(gold_manifest),
                            "city_metadata": str(metadata),
                            "output_csv": str(output_csv),
                        },
                        "columns": {
                            "city_id": "ID_UC_G0",
                            "longitude": "centroid_lon",
                            "latitude": "centroid_lat",
                            "city_area": "city_area_km2",
                            "country": "country",
                            "ghsl_region": "GC_DEV_USR",
                            "koppen_code": "koppen_geiger_code",
                            "koppen_class": "koppen_geiger_class",
                            "koppen_macro_group": "koppen_geiger_macro_group",
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(build_city_pixel_zone_counts(config), output_csv)
            with output_csv.open(encoding="utf-8", newline="") as file:
                rows = {
                    int(row["ID_UC_G0"]): row
                    for row in csv.DictReader(file)
                }

            self.assertEqual(rows[1]["stable_pixel_count"], "1")
            self.assertEqual(rows[1]["development_pixel_count"], "2")
            self.assertEqual(rows[1]["water_eligible_pixel_count"], "2")
            self.assertEqual(rows[1]["stable_water_eligible_pixel_count"], "1")
            self.assertEqual(
                rows[1]["development_water_eligible_pixel_count"], "1"
            )
            self.assertEqual(rows[2]["stable_pixel_count"], "2")
            self.assertEqual(rows[2]["development_pixel_count"], "0")
            self.assertEqual(rows[2]["stable_water_excluded_pixel_count"], "1")
            self.assertEqual(rows[2]["stable_isa_threshold"], "0.7")
            self.assertEqual(rows[1]["country"], "Country A")
            self.assertEqual(rows[1]["GC_DEV_USR"], "Region A")
            self.assertEqual(rows[1]["koppen_geiger_class"], "Cfa")
            self.assertEqual(rows[1]["climate_major"], "C")
            self.assertEqual(rows[1]["hemisphere"], "north")
            self.assertEqual(rows[1]["centroid_lon"], "-70.0")
            self.assertEqual(rows[1]["city_area_km2"], "120.5")


if __name__ == "__main__":
    unittest.main()
