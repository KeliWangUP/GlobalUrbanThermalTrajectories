import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src.city_era5_alignment import align_city_era5


def _write_fixture(root: Path) -> Path:
    raw = root / "raw"
    raw.mkdir()
    rows = [
        (1, "2019-12", 3.0),
        (1, "2020-01", 6.0),
        (1, "2020-02", 9.0),
        (2, "2019-12", None),
        (2, "2020-01", None),
        (2, "2020-02", None),
    ]
    rows.extend((1, f"2020-{month:02d}", float(month)) for month in range(3, 10))
    rows.extend((4, f"2020-{month:02d}", float(month)) for month in (1, 3, 4, 5, 6, 7, 8, 9))
    with (raw / "era5.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["city_id", "month", "temperature_2m_c"])
        writer.writerows(rows)

    con = duckdb.connect(":memory:")
    try:
        for kind, query in {
            "monthly": """
                SELECT * FROM (VALUES
                    (1, 2019, 12, 42.0), (1, 2020, 1, 43.0),
                    (2, 2020, 1, 44.0), (3, 2020, 1, 45.0),
                    (1, 2018, 1, 46.0)
                ) t(ID_UC_G0, year, month, original_value)
            """,
            "seasonal": """
                SELECT * FROM (VALUES
                    (1, 2020, 'DJF', 42.0), (2, 2020, 'DJF', 43.0),
                    (3, 2020, 'DJF', 44.0), (1, 2019, 'DJF', 45.0),
                    (4, 2020, 'DJF', 46.0)
                ) t(ID_UC_G0, season_year, season, original_value)
            """,
            "annual": """
                SELECT * FROM (VALUES
                    (1, 2020, 42.0), (2, 2020, 43.0),
                    (3, 2020, 44.0), (1, 2019, 45.0), (4, 2020, 46.0)
                ) t(ID_UC_G0, year, original_value)
            """,
        }.items():
            con.execute(
                f"COPY ({query}) TO '{(root / (kind + '.parquet')).as_posix()}' "
                "(FORMAT PARQUET)"
            )
    finally:
        con.close()
    config = {
        "paths": {
            "era5_raw_dir": "raw",
            "monthly_suhi_file": "monthly.parquet",
            "seasonal_suhi_file": "seasonal.parquet",
            "annual_suhi_file": "annual.parquet",
            "output_dir": "out",
        },
        "require_complete_city_months": False,
        "minimum_valid_months_per_season": 2,
        "minimum_valid_months_per_year": 9,
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


class CityEra5AlignmentTests(unittest.TestCase):
    def test_alignment_preserves_rows_and_handles_partial_periods(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _write_fixture(root)
            report = align_city_era5(config)
            con = duckdb.connect(":memory:")
            try:
                monthly = con.execute(
                    """
                    SELECT ID_UC_G0, year, month, original_value,
                           era5_temperature_2m_c, era5_record_present
                    FROM read_parquet(?) ORDER BY ID_UC_G0, year, month
                    """,
                    [str(root / "out" / "suhi_monthly_era5.parquet")],
                ).fetchall()
                seasonal = con.execute(
                    """
                    SELECT ID_UC_G0, season_year, era5_temperature_2m_c,
                           era5_n_available_months, era5_n_valid_months, era5_qc_pass
                    FROM read_parquet(?) ORDER BY ID_UC_G0, season_year
                    """,
                    [str(root / "out" / "suhi_seasonal_era5.parquet")],
                ).fetchall()
                annual = con.execute(
                    """
                    SELECT ID_UC_G0, year, era5_temperature_2m_c,
                           era5_n_available_months, era5_n_valid_months, era5_qc_pass
                    FROM read_parquet(?) ORDER BY ID_UC_G0, year
                    """,
                    [str(root / "out" / "suhi_annual_era5.parquet")],
                ).fetchall()
            finally:
                con.close()
            self.assertEqual(monthly[0], (1, 2018, 1, 46.0, None, False))
            self.assertIn((2, 2020, 1, 44.0, None, True), monthly)
            self.assertIn((1, 2020, 6.0, 3, 3, True), seasonal)
            self.assertIn((2, 2020, None, 3, 0, False), seasonal)
            self.assertIn((3, 2020, None, 0, 0, False), seasonal)
            self.assertIn((1, 2019, None, 0, 0, False), seasonal)
            self.assertIn((4, 2020, None, 1, 1, False), seasonal)
            self.assertIn((1, 2019, None, 1, 1, False), annual)
            self.assertIn((1, 2020, 57 / 9, 9, 9, True), annual)
            self.assertIn((4, 2020, None, 8, 8, False), annual)
            self.assertEqual(report["outputs"]["monthly"]["rows"], 5)
            self.assertTrue((root / "out" / "era5_alignment_report.json").is_file())

    def test_duplicate_era5_city_month_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _write_fixture(root)
            with (root / "raw" / "era5.csv").open("a", encoding="utf-8") as file:
                file.write("1,2020-01,6\n")
            with self.assertRaisesRegex(ValueError, "duplicate city-month"):
                align_city_era5(config)
            self.assertFalse((root / "out" / "suhi_monthly_era5.parquet").exists())

    def test_invalid_temperature_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _write_fixture(root)
            with (root / "raw" / "era5.csv").open("a", encoding="utf-8") as file:
                file.write("1,2020-11,not-a-temperature\n")
            with self.assertRaisesRegex(ValueError, "invalid"):
                align_city_era5(config)

    def test_complete_mode_rejects_missing_city_month_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_fixture(root)
            config = json.loads(path.read_text(encoding="utf-8"))
            config["require_complete_city_months"] = True
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lacks .* SUHI city-month records"):
                align_city_era5(path)
            self.assertFalse((root / "out" / "suhi_monthly_era5.parquet").exists())


if __name__ == "__main__":
    unittest.main()
