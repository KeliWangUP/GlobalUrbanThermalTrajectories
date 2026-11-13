import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src.city_season_zone_trends import build_city_season_zone_trends
from src.city_season_zone_gamma_figures import (
    build_city_season_zone_gamma_figures,
)


SLOTS = ("terra_day", "terra_night", "aqua_day", "aqua_night")
ZONES = ("stable", "development", "overall")


def _write_source(path: Path) -> None:
    rows = []
    urban_slopes = {
        "DJF": 0.05,
        "MAM": 0.10,
        "JJA": 0.20,
        "SON": 0.15,
    }
    rural_slopes = {
        "DJF": 0.10,
        "MAM": 0.08,
        "JJA": 0.05,
        "SON": 0.07,
    }
    for city_id, climate, latitude in ((1, "Cfa", 30.0), (2, "Cfb", -30.0)):
        for year in range(2000, 2005):
            for month in range(1, 13):
                season = (
                    "DJF"
                    if month in (12, 1, 2)
                    else "MAM"
                    if month in (3, 4, 5)
                    else "JJA"
                    if month in (6, 7, 8)
                    else "SON"
                )
                for urban_zone_index, urban_zone in enumerate(ZONES):
                    for slot_index, slot in enumerate(SLOTS):
                        urban = (
                            20.0
                            + urban_slopes[season] * (year - 2000)
                            + month * 0.01
                            + urban_zone_index * 0.1
                            + slot_index * 0.01
                        )
                        rural = (
                            10.0
                            + rural_slopes[season] * (year - 2000)
                            + month * 0.005
                            + slot_index * 0.005
                        )
                        rows.append(
                            (
                                city_id,
                                latitude,
                                climate,
                                "C",
                                "Region A" if city_id == 1 else "Region B",
                                urban_zone,
                                year,
                                month,
                                slot,
                                True,
                                urban,
                                rural,
                                urban - rural,
                            )
                        )

    connection = duckdb.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE source_data AS
        SELECT * FROM (VALUES
            """ + ", ".join(
                "(" + ", ".join(
                    (
                        str(row[0]),
                        str(row[1]),
                        "'" + row[2] + "'",
                        "'" + row[3] + "'",
                        "'" + row[4] + "'",
                        "'" + row[5] + "'",
                        str(row[6]),
                        str(row[7]),
                        "'" + row[8] + "'",
                        "TRUE",
                        repr(row[10]),
                        repr(row[11]),
                        repr(row[12]),
                    )
                ) + ")"
                for row in rows
            )
        + """) AS source(
            ID_UC_G0, centroid_lat, koppen_geiger_class, climate_major,
            GC_DEV_USR, urban_zone, year, month, slot, paired_valid, urban_lst,
            rural_lst, suhi
        )
        """
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"COPY source_data TO '{path.as_posix()}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.close()


class CitySeasonZoneTrendTests(unittest.TestCase):
    def test_builds_seasonal_zone_trends_and_hemisphere_gamma(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "city_month_zone_paired_qc.parquet"
            _write_source(source)
            gold_manifest = root / "gold_manifest.json"
            gold_manifest.write_text(
                json.dumps(
                    {
                        "urban_classification": {
                            "baseline_year": 2000,
                            "stable_isa_threshold": 0.7,
                            "stable_label": "stable",
                            "development_label": "development",
                        }
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "seasonal_zone_trends"
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "paths": {
                            "monthly_zone_gold": str(source),
                            "gold_manifest": str(gold_manifest),
                            "output_dir": str(output_dir),
                        },
                        "urban_classification": {
                            "baseline_year": 2000,
                            "stable_isa_threshold": 0.7,
                            "stable_label": "stable",
                            "development_label": "development",
                        },
                        "urban_groups": list(ZONES),
                        "windows": [[2001, 2004]],
                        "qc": {
                            "min_valid_months_per_season": 2,
                            "min_valid_year_fraction": 1.0,
                            "min_each_half_fraction": 1.0,
                            "max_missing_span_years": 0,
                        },
                        "beta_scale_years": 10,
                        "decomposition_tolerance": 1e-8,
                        "processing": {
                            "duckdb_threads": 1,
                            "duckdb_memory": "512MB",
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                build_city_season_zone_trends(config_path), output_dir
            )
            connection = duckdb.connect(":memory:")
            trends_path = output_dir / "city_season_zone_trends_2001_2004.parquet"
            gamma_path = output_dir / "city_season_zone_gamma_2001_2004.parquet"
            trend_count = connection.execute(
                "SELECT COUNT(*) FROM read_parquet(?)", [trends_path.as_posix()]
            ).fetchone()[0]
            gamma_count = connection.execute(
                "SELECT COUNT(*) FROM read_parquet(?)", [gamma_path.as_posix()]
            ).fetchone()[0]
            region_names = {
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT GC_DEV_USR FROM read_parquet(?)",
                    [gamma_path.as_posix()],
                ).fetchall()
            }
            north = connection.execute(
                """
                SELECT summer_season, winter_season, gamma_urban,
                       gamma_rural, gamma_suhi
                FROM read_parquet(?)
                WHERE ID_UC_G0 = 1 AND urban_zone = 'stable'
                  AND slot = 'terra_day'
                """,
                [gamma_path.as_posix()],
            ).fetchone()
            south = connection.execute(
                """
                SELECT summer_season, winter_season, gamma_urban,
                       gamma_rural, gamma_suhi
                FROM read_parquet(?)
                WHERE ID_UC_G0 = 2 AND urban_zone = 'stable'
                  AND slot = 'terra_day'
                """,
                [gamma_path.as_posix()],
            ).fetchone()
            connection.close()

            self.assertEqual(trend_count, 2 * len(ZONES) * 4 * len(SLOTS))
            self.assertEqual(gamma_count, 2 * len(ZONES) * len(SLOTS))
            self.assertEqual(region_names, {"Region A", "Region B"})
            self.assertEqual(north[:2], ("JJA", "DJF"))
            self.assertAlmostEqual(north[2], 1.5)
            self.assertAlmostEqual(north[3], -0.5)
            self.assertAlmostEqual(north[4], 2.0)
            self.assertEqual(south[:2], ("DJF", "JJA"))
            self.assertAlmostEqual(south[2], -1.5)
            self.assertAlmostEqual(south[3], 0.5)
            self.assertAlmostEqual(south[4], -2.0)
            self.assertTrue((output_dir / "manifest.json").is_file())

            figure_dir = root / "seasonal_zone_figures"
            figure_config_path = root / "figure_config.json"
            figure_config_path.write_text(
                json.dumps(
                    {
                        "paths": {
                            "gamma_file": str(gamma_path),
                            "output_dir": str(figure_dir),
                        },
                        "major_climate_group": "C",
                        "subclimate_classes": ["Cfa"],
                        "hemisphere": "all",
                        "synchrony_color_by": "ghsl_region",
                        "dpi": 72,
                        "decomposition_tolerance": 1e-8,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                build_city_season_zone_gamma_figures(figure_config_path),
                figure_dir,
            )
            self.assertTrue(
                (figure_dir / "figure_1_zone_gamma_suhi_summary.png").is_file()
            )
            self.assertTrue(
                (figure_dir / "figure_2_zone_city_synchrony.png").is_file()
            )
            figure_manifest = json.loads(
                (figure_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                figure_manifest["synchrony_color_by"], "GC_DEV_USR"
            )
            connection = duckdb.connect(":memory:")
            summary_rows = connection.execute(
                "SELECT COUNT(*) FROM read_csv_auto(?)",
                [(figure_dir / "city_season_zone_gamma_summary.csv").as_posix()],
            ).fetchone()[0]
            connection.close()
            self.assertEqual(summary_rows, len(ZONES) * len(SLOTS))


if __name__ == "__main__":
    unittest.main()
