import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from build_city_month_gold import build_gold, literal
from diagnose_city_pixel_zone_counts import build_city_pixel_zone_counts


CONFIG_TEMPLATE = MODULE_DIR / "city_month_gold_config.json"


def _copy_query(connection, query, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"COPY ({query}) TO {literal(path.as_posix())} "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )


class CityMonthGoldTests(unittest.TestCase):
    def test_builds_paired_qc_and_city_trend_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixel_root = root / "pixel_lst"
            connection = duckdb.connect(":memory:")
            for city_id in (10, 20):
                slot_fields = []
                for slot in ("terra_day", "terra_night", "aqua_day", "aqua_night"):
                    slot_fields.extend(
                        (
                            f"(20.0 + (year - 2000) * 0.1 + month * 0.01 "
                            f"+ pixel_x * 0.01) AS {slot}_mean",
                            f"5.0 AS {slot}_clear_days_mean",
                            f"1.0 AS {slot}_n8day_mean",
                        )
                    )
                query = f"""
                    SELECT {city_id}::INTEGER AS ID_UC_G0,
                           ('MODIS_{city_id}_' || pixel_x::VARCHAR)
                               AS pixel_id,
                           year::SMALLINT AS year,
                           month::TINYINT AS month,
                           {", ".join(slot_fields)}
                    FROM range(2000, 2004) AS years(year)
                    CROSS JOIN range(1, 13) AS months(month)
                    CROSS JOIN range(0, 4) AS pixels(pixel_x)
                """
                _copy_query(
                    connection,
                    query,
                    pixel_root / f"city_id={city_id}" / "data.parquet",
                )

            development_query = """
                SELECT 10::INTEGER AS ID_UC_G0, 'MODIS_10_0'::VARCHAR AS pixel_id,
                       0.0::FLOAT AS water_fraction,
                       0.8::FLOAT AS isa_urban_fraction_2000
                UNION ALL
                SELECT 10::INTEGER, 'MODIS_10_1'::VARCHAR, 0.1::FLOAT, 0.9::FLOAT
                UNION ALL
                SELECT 10::INTEGER, 'MODIS_10_2'::VARCHAR, 0.0::FLOAT, 0.4::FLOAT
                UNION ALL
                SELECT 10::INTEGER, 'MODIS_10_3'::VARCHAR, 0.1::FLOAT, 0.6::FLOAT
                UNION ALL
                SELECT 20::INTEGER, 'MODIS_20_0'::VARCHAR, 0.0::FLOAT, 0.8::FLOAT
                UNION ALL
                SELECT 20::INTEGER, 'MODIS_20_1'::VARCHAR, 0.1::FLOAT, 0.9::FLOAT
                UNION ALL
                SELECT 20::INTEGER, 'MODIS_20_2'::VARCHAR, 0.0::FLOAT, 0.4::FLOAT
                UNION ALL
                SELECT 20::INTEGER, 'MODIS_20_3'::VARCHAR, 0.1::FLOAT, 0.6::FLOAT
            """
            development = root / "pixel_development.parquet"
            _copy_query(connection, development_query, development)

            for year in range(2000, 2004):
                rural_fields = []
                for slot in ("terra_day", "terra_night", "aqua_day", "aqua_night"):
                    rural_fields.extend(
                        (
                            f"(10.0 + ({year} - 2000) * 0.05 + month * 0.02) "
                            f"AS {slot}_mean",
                            f"4.0 AS {slot}_clear_days_mean",
                            f"1.0 AS {slot}_n8day_mean",
                            f"5::INTEGER AS {slot}_count",
                        )
                    )
                rural_query = f"""
                    SELECT city_id::INTEGER AS ID_UC_G0,
                           {year}::SMALLINT AS year,
                           month::TINYINT AS month,
                           {", ".join(rural_fields)}
                    FROM (VALUES (10), (20)) AS cities(city_id)
                    CROSS JOIN range(1, 13) AS months(month)
                """
                _copy_query(
                    connection,
                    rural_query,
                    root / "rural" / f"rural_{year}.parquet",
                )

            metadata = root / "city_metadata.parquet"
            _copy_query(
                connection,
                """
                SELECT * FROM (VALUES
                    (10::INTEGER, -70.0::DOUBLE, 30.0::DOUBLE, 120.0::DOUBLE,
                     'Country A', 1::INTEGER, 'Af', 'Tropical', 'Region A'),
                    (20::INTEGER, 20.0::DOUBLE, -30.0::DOUBLE, 80.0::DOUBLE,
                     'Country B', 4::INTEGER, 'BWh', 'Arid', 'Region B')
                ) AS metadata(
                    ID_UC_G0, centroid_lon, centroid_lat, city_area_km2,
                    country, koppen_geiger_code, koppen_geiger_class,
                    koppen_geiger_macro_group, GC_DEV_USR
                )
                """,
                metadata,
            )
            connection.close()

            city_ids = root / "priority_city_ids.json"
            city_ids.write_text(json.dumps([10, 20]), encoding="utf-8")
            silver_manifest = root / "silver_manifest.json"
            silver_manifest.write_text('{"format_version": 1}\n', encoding="utf-8")
            output_dir = root / "gold"

            config = json.loads(CONFIG_TEMPLATE.read_text(encoding="utf-8"))
            config["paths"] = {
                "pixel_lst": str(pixel_root / "city_id=*" / "data.parquet"),
                "pixel_development": str(development),
                "rural_lst": str(root / "rural" / "rural_*.parquet"),
                "city_metadata": str(metadata),
                "priority_city_ids": str(city_ids),
                "silver_manifest": str(silver_manifest),
                "output_dir": str(output_dir),
            }
            config["study"].update(
                {"start_year": 2000, "end_year": 2003, "expected_city_count": 2}
            )
            config["processing"].update(
                {"duckdb_threads": 1, "duckdb_memory": "512MB"}
            )
            config["windows"] = [[2000, 2003]]
            config["qc"].update(
                {
                    "min_urban_pixels": 2,
                    "min_rural_pixels": 5,
                    "min_valid_year_fraction": 1.0,
                    "min_each_half_fraction": 1.0,
                    "max_missing_span_years": 0,
                }
            )
            config_path = root / "gold_config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            self.assertEqual(build_gold(config_path), output_dir)
            result = duckdb.connect(":memory:")
            paired_rows = result.execute(
                "SELECT COUNT(*) FROM read_parquet(?)",
                [str(output_dir / "city_month_paired_qc.parquet")],
            ).fetchone()[0]
            self.assertEqual(paired_rows, 2 * 4 * 12 * 4)
            zone_path = output_dir / "city_month_zone_paired_qc.parquet"
            zone_rows, valid_zone_rows = result.execute(
                """
                SELECT COUNT(*), COUNT(*) FILTER (WHERE paired_valid)
                FROM read_parquet(?)
                """,
                [str(zone_path)],
            ).fetchone()
            self.assertEqual(zone_rows, 2 * 3 * 4 * 12 * 4)
            self.assertEqual(valid_zone_rows, zone_rows)
            zone_labels = {
                row[0]
                for row in result.execute(
                    "SELECT DISTINCT urban_zone FROM read_parquet(?)",
                    [str(zone_path)],
                ).fetchall()
            }
            self.assertEqual(zone_labels, {"stable", "development", "overall"})
            ghsl_regions = {
                row[0]
                for row in result.execute(
                    "SELECT DISTINCT GC_DEV_USR FROM read_parquet(?)",
                    [str(zone_path)],
                ).fetchall()
            }
            self.assertEqual(ghsl_regions, {"Region A", "Region B"})
            classified_counts = result.execute(
                """
                SELECT urban_zone, COUNT(*)
                FROM read_parquet(?)
                WHERE water_eligible
                GROUP BY urban_zone
                """,
                [str(output_dir / "city_pixel_classification.parquet")],
            ).fetchall()
            self.assertEqual(
                dict(classified_counts),
                {"stable": 4, "development": 4},
            )

            wide_path = output_dir / "city_month_trend_wide_2000_2003.parquet"
            city_count, slope, climate = result.execute(
                """
                SELECT COUNT(*),
                       MAX(suhi_td_m01_trend_decade)
                           FILTER (WHERE ID_UC_G0 = 10),
                       MAX(koppen_geiger_macro_group)
                           FILTER (WHERE ID_UC_G0 = 10)
                FROM read_parquet(?)
                """,
                [str(wide_path)],
            ).fetchone()
            self.assertEqual(city_count, 2)
            self.assertAlmostEqual(slope, 0.5)
            self.assertEqual(climate, "Tropical")
            self.assertTrue((output_dir / "manifest.json").is_file())
            gold_manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                gold_manifest["urban_classification"]["stable_isa_threshold"],
                0.7,
            )
            count_csv = root / "city_pixel_zone_counts.csv"
            count_config = root / "count_config.json"
            count_config.write_text(
                json.dumps(
                    {
                        "paths": {
                            "pixel_classification": str(
                                output_dir / "city_pixel_classification.parquet"
                            ),
                            "gold_manifest": str(output_dir / "manifest.json"),
                            "city_metadata": str(metadata),
                            "output_csv": str(count_csv),
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
            build_city_pixel_zone_counts(count_config)
            count_row = result.execute(
                """
                SELECT stable_pixel_count, development_pixel_count,
                       stable_water_eligible_pixel_count,
                       development_water_eligible_pixel_count
                FROM read_csv_auto(?)
                WHERE ID_UC_G0 = 10
                """,
                [str(count_csv)],
            ).fetchone()
            self.assertEqual(count_row, (2, 2, 2, 2))
            result.close()

    def test_refuses_to_overwrite_an_existing_gold_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads(CONFIG_TEMPLATE.read_text(encoding="utf-8"))
            output_dir = root / "gold"
            output_dir.mkdir()
            config["paths"]["output_dir"] = str(output_dir)
            config_path = root / "gold_config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(FileExistsError):
                build_gold(config_path)


if __name__ == "__main__":
    unittest.main()
