import csv
import hashlib
import io
import json
import runpy
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src.city_lst_csv import COUNT_COLUMNS, EXPECTED_COLUMNS, convert_csv
from src.pixel_silver import (
    PIXEL_COLUMNS,
    build_pixel_silver,
    load_pixel_silver_config
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_inputs(
    root: Path,
    invalid_value: bool = False,
    duplicate_key: bool = False,
    invalid_rural: bool = False,
    extra_archive_entry: bool = False
) -> Path:
    ids = root / "city_ids.json"
    ids.write_text(json.dumps([6, 9]), encoding="utf-8")
    development = root / "development.parquet"
    pq.write_table(pa.table({
        "pixel_id": ["MODIS_1_2", "MODIS_3_4", "MODIS_5_6"],
        "ID_UC_G0": pa.array([6, 9, 12], type=pa.int32()),
        "modis_x": pa.array([1, 3, 5], type=pa.int32()),
        "modis_y": pa.array([2, 4, 6], type=pa.int32()),
        "lon": pa.array([1.0, 3.0, 5.0], type=pa.float32()),
        "lat": pa.array([2.0, 4.0, 6.0], type=pa.float32()),
        "other_field": pa.array([None, 0.25, 1.0], type=pa.float32())
    }), development, compression="zstd")

    archive_path = root / "lst.tar"
    with tarfile.open(archive_path, "w") as archive:
        for city_id, pixel_id in ((6, "MODIS_1_2"), (9, "MODIS_3_4")):
            arrays = {}
            for name in PIXEL_COLUMNS:
                if name == "pixel_id":
                    arrays[name] = pa.array([pixel_id] * 12, type=pa.large_string())
                elif name == "ID_UC_G0":
                    arrays[name] = pa.array([city_id] * 12, type=pa.int64())
                elif name == "year":
                    arrays[name] = pa.array([2020] * 12, type=pa.int16())
                elif name == "month":
                    months = list(range(1, 13))
                    if duplicate_key and city_id == 6:
                        months[-1] = 1
                    arrays[name] = pa.array(months, type=pa.int8())
                elif name.endswith("_mean") and not (
                    "clear_days" in name or "n8day" in name
                ):
                    arrays[name] = pa.array([None, *([10.125] * 11)], type=pa.float32())
                else:
                    value = "1" if name.endswith("_count") or "n8day" in name else "0"
                    if name.endswith("_median"):
                        value = "10.125"
                    if "clear_days" in name:
                        value = "4.5"
                    values = [None, *([value] * 11)]
                    if invalid_value and city_id == 6 and name == "aqua_day_median":
                        values[2] = "not-a-number"
                    arrays[name] = pa.array(values, type=pa.large_string())
            temporary = root / f"{city_id}.parquet"
            pq.write_table(pa.table(arrays), temporary, compression="zstd")
            archive.add(
                temporary,
                arcname=f"pixel_lst_parquet_v2/city_id={city_id}/data.parquet"
            )
        if extra_archive_entry:
            info = tarfile.TarInfo("../unexpected.parquet")
            info.size = 3
            archive.addfile(info, io.BytesIO(b"bad"))

    rural_dir = root / "rural"
    rural_dir.mkdir()
    with (rural_dir / "rural_2020.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        for city_id in (6, 9):
            for month in range(1, 13):
                row = {name: "1.5" for name in EXPECTED_COLUMNS}
                row.update({"ID_UC_G0": city_id, "year": 2020, "month": month})
                for name in COUNT_COLUMNS:
                    row[name] = "2"
                if month == 1:
                    row["aqua_day_mean"] = ""
                if invalid_rural and city_id == 6 and month == 2:
                    row["terra_day_mean"] = "nan"
                writer.writerow(row)

    config = {
        "paths": {
            "pixel_lst_tar": str(archive_path),
            "pixel_development_parquet": str(development),
            "rural_csv_dir": str(rural_dir),
            "priority_city_ids_json": str(ids),
            "output_dir": str(root / "silver")
        },
        "study": {"start_year": 2020, "end_year": 2020, "expected_city_count": 2},
        "processing": {
            "compression": "zstd", "batch_size": 5,
            "duckdb_threads": 1, "duckdb_memory_limit": "512MB"
        }
    }
    path = root / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


class PixelSilverTests(unittest.TestCase):
    def test_builds_city_partitioned_silver_without_changing_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _make_inputs(root)
            original_archive = _digest(root / "lst.tar")
            original_dev = _digest(root / "development.parquet")
            config = load_pixel_silver_config(path)
            output = build_pixel_silver(config)

            self.assertEqual(output, root / "silver")
            self.assertEqual(_digest(root / "lst.tar"), original_archive)
            self.assertEqual(_digest(root / "development.parquet"), original_dev)
            self.assertEqual(
                _digest(output / "pixel_development" / "data.parquet"),
                original_dev
            )
            city_file = output / "pixel_lst" / "city_id=6" / "data.parquet"
            city = pq.ParquetFile(city_file).read()
            self.assertEqual(city.num_rows, 12)
            self.assertEqual(city.schema.field("pixel_id").type, pa.string())
            self.assertEqual(city.schema.field("ID_UC_G0").type, pa.int32())
            self.assertEqual(city.schema.field("terra_day_median").type, pa.float64())
            self.assertEqual(city.schema.field("terra_day_count").type, pa.int32())
            self.assertIsNone(city.column("terra_day_mean")[0].as_py())
            self.assertEqual(city.column("terra_day_mean")[1].as_py(), 10.125)
            self.assertEqual(city.column("terra_day_median")[1].as_py(), 10.125)
            self.assertEqual(city.column("terra_day_count")[1].as_py(), 1)

            rural = pq.ParquetFile(output / "rural_lst" / "rural_2020.parquet").read()
            self.assertEqual(rural.num_rows, 24)
            self.assertEqual(rural.schema.field("aqua_day_mean").type, pa.float64())
            self.assertEqual(rural.column("aqua_day_mean").null_count, 2)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                (manifest["city_count"], manifest["pixel_lst_rows"],
                 manifest["priority_pixel_count"], manifest["development_rows"],
                 manifest["rural_rows"]),
                (2, 24, 2, 3, 24)
            )
            self.assertEqual(manifest["sources"][str(root / "lst.tar")]["sha256"], original_archive)
            self.assertFalse(list(root.glob(".silver_staging_*")))
            with duckdb.connect(":memory:") as connection:
                glob = str(output / "pixel_lst" / "city_id=*" / "data.parquet")
                rows = connection.execute(
                    "SELECT count(*) FROM read_parquet(?, hive_partitioning=true) "
                    "WHERE city_id=6 AND ID_UC_G0=6 AND year=2020",
                    [glob]
                ).fetchone()[0]
                self.assertEqual(rows, 12)
            with self.assertRaises(FileExistsError):
                build_pixel_silver(config)

    def test_invalid_numeric_value_refuses_to_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _make_inputs(root, invalid_value=True)
            original = _digest(root / "lst.tar")
            with self.assertRaisesRegex(ValueError, "aqua_day_median"):
                build_pixel_silver(load_pixel_silver_config(path))
            self.assertEqual(_digest(root / "lst.tar"), original)
            self.assertFalse((root / "silver").exists())
            self.assertEqual(len(list(root.glob(".silver_staging_*"))), 1)

    def test_duplicate_keys_stop_before_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _make_inputs(root, duplicate_key=True)
            with self.assertRaisesRegex(ValueError, "Invalid LST keys"):
                build_pixel_silver(load_pixel_silver_config(path))
            self.assertFalse((root / "silver").exists())
            self.assertFalse(list(root.glob(".silver_staging_*")))

    def test_rural_nonfinite_value_refuses_to_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _make_inputs(root, invalid_rural=True)
            with self.assertRaisesRegex(ValueError, "Non-finite value"):
                build_pixel_silver(load_pixel_silver_config(path))
            self.assertFalse((root / "silver").exists())

    def test_rejects_unexpected_archive_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _make_inputs(root, extra_archive_entry=True)
            with self.assertRaisesRegex(ValueError, "Unexpected archive path"):
                build_pixel_silver(load_pixel_silver_config(path))
            self.assertFalse((root / "silver").exists())

    def test_rejects_output_overlapping_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _make_inputs(root)
            config = json.loads(path.read_text(encoding="utf-8"))
            config["paths"]["output_dir"] = str(root / "rural" / "silver")
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlaps an input"):
                build_pixel_silver(load_pixel_silver_config(path))

    def test_nonfinite_development_value_stops_before_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _make_inputs(root)
            source = root / "development.parquet"
            table = pq.ParquetFile(source).read()
            index = table.schema.get_field_index("other_field")
            table = table.set_column(index, "other_field", pa.array(
                [None, float("nan"), 1.0], type=pa.float32()
            ))
            pq.write_table(table, source, compression="zstd")
            with self.assertRaisesRegex(ValueError, "Non-finite development value"):
                build_pixel_silver(load_pixel_silver_config(path))
            self.assertFalse((root / "silver").exists())
            self.assertFalse(list(root.glob(".silver_staging_*")))

    def test_existing_city_csv_entry_point_uses_shared_converter(self):
        namespace = runpy.run_path(str(MODULE_DIR / "01_convert_exports.py"))
        self.assertIs(namespace["convert_csv"], convert_csv)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_inputs(root)
            output = root / "legacy_rural_2020.parquet"
            convert_csv(root / "rural" / "rural_2020.csv", output, 2020, "zstd")
            self.assertEqual(pq.ParquetFile(output).metadata.num_rows, 24)


if __name__ == "__main__":
    unittest.main()
