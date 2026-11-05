import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src.city_month_slot_trends import build_city_month_slot_trends


def _write_source(path: Path) -> None:
    connection = duckdb.connect(":memory:")
    slot_columns = []
    slot_values = []
    for slot in ("terra_day", "terra_night", "aqua_day", "aqua_night"):
        slot_columns.extend(
            [
                f"qc_pass_{slot} BOOLEAN",
                f"urban_{slot}_mean DOUBLE",
                f"rural_{slot}_mean DOUBLE",
                f"suhi_{slot}_mean DOUBLE",
            ]
        )
        slot_values.extend(
            [
                f"TRUE AS qc_pass_{slot}",
                f"urban_value AS urban_{slot}_mean",
                f"rural_value AS rural_{slot}_mean",
                f"(urban_value - rural_value) AS suhi_{slot}_mean",
            ]
        )

    years = list(range(2003, 2020))
    residuals = np.array([(-1.0) ** i for i in range(len(years))])
    residuals -= residuals.mean()
    elapsed = np.array(years) - 2003
    residuals -= (
        np.dot(residuals, elapsed) / np.dot(elapsed, elapsed)
    ) * elapsed
    rows = []
    for year, residual in zip(years, residuals):
        for city_id, climate, latitude in (
            (1, "Af", 20.0),
            (2, "Cfa", -20.0),
        ):
            slot_qc = {
                "terra_day": year != 2007,
                "terra_night": True,
                "aqua_day": year in years[:14],
                "aqua_night": True,
            }
            fields = [
                str(city_id),
                str(year),
                "1",
                str(latitude),
                "'" + climate + "'",
            ]
            for slot in ("terra_day", "terra_night", "aqua_day", "aqua_night"):
                passed = slot_qc[slot]
                offset = 0.01 * (
                    ("terra_day", "terra_night", "aqua_day", "aqua_night").index(slot)
                )
                urban = 20 + 0.2 * (year - 2003) + residual + offset
                rural = 10 + 0.1 * (year - 2003) + 0.5 * residual
                if not passed:
                    urban += 1000
                fields.extend(
                    [
                        str(passed).upper(),
                        repr(urban),
                        repr(rural),
                        repr(urban - rural),
                    ]
                )
            rows.append("(" + ", ".join(fields) + ")")

    columns = [
        "ID_UC_G0 INTEGER",
        "year INTEGER",
        "month INTEGER",
        "centroid_lat DOUBLE",
        "koppen_geiger_class VARCHAR",
        *slot_columns,
    ]
    connection.execute(
        f"""
        CREATE TABLE source_data AS
        SELECT * FROM (VALUES {", ".join(rows)})
        AS source({", ".join(column.split()[0] for column in columns)})
        """
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"COPY source_data TO '{path.as_posix()}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.close()


class CityMonthSlotTrendTests(unittest.TestCase):
    def test_builds_paired_qc_trends_and_standard_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "suhi_monthly.parquet"
            _write_source(source)
            output_dir = root / "result"
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "paths": {
                            "monthly_suhi_file": str(source),
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

            output_path = build_city_month_slot_trends(config_path)
            connection = duckdb.connect(":memory:")
            rows = connection.execute(
                """
                SELECT ID_UC_G0, climate, centroid_lat, hemisphere, month, slot,
                       beta_suhi, beta_urban, beta_rural,
                       se_suhi, se_urban, se_rural, n_valid_years
                FROM read_parquet(?)
                ORDER BY ID_UC_G0, slot
                """,
                [output_path.as_posix()],
            ).fetchall()
            connection.close()

            self.assertEqual(len(rows), 6)
            by_key = {(row[0], row[5]): row for row in rows}
            day = by_key[(1, "terra_day")]
            self.assertEqual(day[1:5], ("Af", 20.0, "north", 1))
            self.assertEqual(day[12], 16)

            years = np.array([year for year in range(2003, 2020) if year != 2007])
            residuals = np.array(
                [(-1.0) ** i for i in range(17)], dtype=float
            )
            residuals -= residuals.mean()
            elapsed_all = np.arange(17, dtype=float)
            residuals -= (
                np.dot(residuals, elapsed_all)
                / np.dot(elapsed_all, elapsed_all)
            ) * elapsed_all
            kept_residuals = residuals[np.array(years) - 2003]
            x = years.astype(float)
            x_centered = x - x.mean()
            urban = 20 + 0.2 * (x - 2003) + kept_residuals
            rural = 10 + 0.1 * (x - 2003) + 0.5 * kept_residuals
            suhi = urban - rural

            def slope_se(values):
                centered = values - values.mean()
                slope_per_year = np.dot(x_centered, centered) / np.dot(
                    x_centered, x_centered
                )
                residual = centered - slope_per_year * x_centered
                se_per_year = np.sqrt(
                    np.dot(residual, residual)
                    / (len(x) - 2)
                    / np.dot(x_centered, x_centered)
                )
                return slope_per_year * 10, se_per_year * 10

            expected_suhi = slope_se(suhi)
            expected_urban = slope_se(urban)
            expected_rural = slope_se(rural)
            self.assertAlmostEqual(day[6], expected_suhi[0])
            self.assertAlmostEqual(day[7], expected_urban[0])
            self.assertAlmostEqual(day[8], expected_rural[0])
            for actual, expected in zip(
                day[6:12],
                (
                    expected_suhi[0],
                    expected_urban[0],
                    expected_rural[0],
                    expected_suhi[1],
                    expected_urban[1],
                    expected_rural[1],
                ),
            ):
                self.assertAlmostEqual(actual, expected)

            self.assertNotIn((1, "aqua_day"), by_key)
            self.assertTrue((output_dir / "manifest.json").is_file())

    def test_refuses_to_overwrite_an_existing_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "suhi_monthly.parquet"
            _write_source(source)
            output_dir = root / "result"
            output_dir.mkdir()
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "paths": {
                            "monthly_suhi_file": str(source),
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
            with self.assertRaises(FileExistsError):
                build_city_month_slot_trends(config_path)


if __name__ == "__main__":
    unittest.main()
