import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src.city_season_era5_trends import build_city_season_era5_trends


SLOTS = ("terra_day", "terra_night", "aqua_day", "aqua_night")


def _fixture(root: Path) -> Path:
    source = root / "seasonal.parquet"
    con = duckdb.connect()
    fields = [
        "ID_UC_G0 BIGINT", "season_year INTEGER", "season VARCHAR",
        "centroid_lat DOUBLE", "koppen_geiger_class VARCHAR",
        "era5_temperature_2m_c DOUBLE", "era5_qc_pass BOOLEAN",
    ]
    for slot in SLOTS:
        fields.extend([
            f"qc_pass_season_{slot} BOOLEAN", f"urban_{slot}_mean DOUBLE",
            f"rural_{slot}_mean DOUBLE", f"suhi_{slot}_mean DOUBLE",
        ])
    con.execute(f"CREATE TABLE seasonal ({', '.join(fields)})")
    rows = []
    for city, latitude in ((1, 30), (2, -30), (3, 0), (4, 30), (5, 30)):
        for year in range(2003, 2007):
            elapsed = year - 2003
            for season in ("DJF", "JJA"):
                urban = float(elapsed ** 2 if season == "JJA" else 2 * elapsed)
                rural = float(elapsed)
                era5 = urban
                era5_qc = not (city == 4 and year == 2004)
                row = [city, year, season, latitude, "Dfb", era5, era5_qc]
                for slot in SLOTS:
                    qc = not (city == 1 and year == 2004 and season == "DJF")
                    if city == 5 and (
                        (year == 2003 and season == "DJF")
                        or (year == 2006 and season == "JJA")
                    ):
                        qc = False
                    row.extend([qc, urban, rural, urban - rural])
                rows.append(row)
    con.executemany(f"INSERT INTO seasonal VALUES ({','.join('?' for _ in fields)})", rows)
    con.execute(f"COPY seasonal TO '{source.as_posix()}' (FORMAT PARQUET)")
    con.close()

    modis_config = {
        "paths": {"seasonal_suhi_file": str(source), "output_dir": str(root / "modis")},
        "years": {"start": 2003, "end": 2006, "minimum_valid_years": 3},
        "beta_scale_years": 10,
    }
    (root / "modis.json").write_text(json.dumps(modis_config), encoding="utf-8")
    config = {
        "paths": {
            "seasonal_suhi_file": str(source),
            "modis_trend_config": str(root / "modis.json"),
            "output_dir": str(root / "output"),
        },
        "decomposition_tolerance": 1e-8,
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


class CitySeasonEra5TrendTests(unittest.TestCase):
    def test_paired_seasonal_and_common_year_contrasts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _fixture(root)
            manifest = build_city_season_era5_trends(config)
            con = duckdb.connect()
            seasonal = con.execute(
                """
                SELECT n_valid_years, valid_years, beta_era5, beta_urban, beta_rural, beta_suhi
                FROM read_parquet(?) WHERE ID_UC_G0=1 AND season='JJA' AND slot='terra_day'
                """,
                [str(root / "output" / "city_season_era5_trends.parquet")],
            ).fetchone()
            north = con.execute(
                """
                SELECT summer_season, winter_season, n_common_years, common_years,
                       beta_era5_summer, beta_era5_winter, gamma_era5,
                       gamma_suhi, gamma_urban, gamma_rural
                FROM read_parquet(?) WHERE ID_UC_G0=1 AND slot='terra_day'
                """,
                [str(root / "output" / "city_season_era5_contrasts.parquet")],
            ).fetchone()
            south = con.execute(
                """
                SELECT summer_season, winter_season, gamma_era5
                FROM read_parquet(?) WHERE ID_UC_G0=2 AND slot='terra_day'
                """,
                [str(root / "output" / "city_season_era5_contrasts.parquet")],
            ).fetchone()
            equator = con.execute(
                "SELECT COUNT(*) FROM read_parquet(?) WHERE ID_UC_G0=3",
                [str(root / "output" / "city_season_era5_contrasts.parquet")],
            ).fetchone()[0]
            era5_filtered = con.execute(
                """
                SELECT n_valid_years, valid_years FROM read_parquet(?)
                WHERE ID_UC_G0=4 AND season='JJA' AND slot='terra_day'
                """,
                [str(root / "output" / "city_season_era5_trends.parquet")],
            ).fetchone()
            insufficient_overlap = con.execute(
                """
                SELECT COUNT(*) FROM read_parquet(?)
                WHERE ID_UC_G0=5 AND slot='terra_day'
                """,
                [str(root / "output" / "city_season_era5_contrasts.parquet")],
            ).fetchone()[0]
            con.close()
            self.assertEqual(seasonal[:2], (4, [2003, 2004, 2005, 2006]))
            self.assertAlmostEqual(seasonal[2], 30.0)
            self.assertAlmostEqual(seasonal[5], seasonal[3] - seasonal[4])
            self.assertEqual(north[:4], ("JJA", "DJF", 3, [2003, 2005, 2006]))
            self.assertAlmostEqual(north[4], 200 / 7)
            self.assertAlmostEqual(north[5], 20)
            self.assertAlmostEqual(north[6], 60 / 7)
            self.assertAlmostEqual(north[7], north[8] - north[9])
            self.assertEqual(south[:2], ("DJF", "JJA"))
            self.assertAlmostEqual(south[2], -10)
            self.assertEqual(equator, 0)
            self.assertEqual(era5_filtered, (3, [2003, 2005, 2006]))
            self.assertEqual(insufficient_overlap, 0)
            self.assertEqual(manifest["years"]["minimum_valid_years"], 3)
            self.assertTrue((root / "output" / "manifest.json").is_file())
            with self.assertRaises(FileExistsError):
                build_city_season_era5_trends(config)

    def test_duplicate_season_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _fixture(root)
            source = root / "seasonal.parquet"
            con = duckdb.connect()
            con.execute(
                f"CREATE TABLE duplicate AS SELECT * FROM read_parquet('{source.as_posix()}')"
            )
            con.execute("INSERT INTO duplicate SELECT * FROM duplicate LIMIT 1")
            source.unlink()
            con.execute(f"COPY duplicate TO '{source.as_posix()}' (FORMAT PARQUET)")
            con.close()
            with self.assertRaisesRegex(ValueError, "duplicate keys"):
                build_city_season_era5_trends(config)
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
