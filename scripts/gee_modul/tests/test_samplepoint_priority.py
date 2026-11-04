import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src import samplepoint_priority as priority


class SamplepointPriorityTests(unittest.TestCase):
    def write_csv(self, path, fieldnames, rows):
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_reads_unique_ids_and_stable_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cities.csv"
            config = {"cities": {"priority_city_csv": str(path)}}
            self.write_csv(path, ["city"], [{"city": "42"}, {"city": "7"}, {"city": "42"}])
            ids, digest, resolved_path = priority.load_priority_city_ids(config, "city")
            self.assertEqual(ids, [7, 42])
            self.assertEqual(resolved_path, path.resolve())
            self.write_csv(path, ["city"], [{"city": "7"}, {"city": "42"}])
            self.assertEqual(priority.load_priority_city_ids(config, "city")[1], digest)

    def test_rejects_missing_invalid_and_empty_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cities.csv"
            config = {"cities": {"priority_city_csv": str(path)}}
            for fields, rows in (
                (["wrong"], [{"wrong": "42"}]),
                (["city"], [{"city": ""}]),
                (["city"], [{"city": "bad"}]),
                (["city"], []),
            ):
                self.write_csv(path, fields, rows)
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    priority.load_priority_city_ids(config, "city")

    def test_filters_priority_ids_and_refuses_missing_cities(self):
        points = MagicMock()
        points.aggregate_array.return_value.distinct.return_value.getInfo.return_value = [7, 42, 99]
        selected = points.filter.return_value
        selected.size.return_value.getInfo.return_value = 3
        with patch.object(priority.ee.Filter, "inList", return_value="filter") as make_filter:
            self.assertEqual(
                priority.filter_priority_sample_points(points, "city", [7, 42]),
                (selected, 3)
            )
            make_filter.assert_called_once_with("city", [7, 42])
            with self.assertRaisesRegex(ValueError, "no sample points"):
                priority.filter_priority_sample_points(points, "city", [7, 100])
            make_filter.assert_called_once()

    def test_export_json_count_idempotence_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cities.csv"
            output = Path(directory) / "ids.json"
            self.write_csv(source, ["city"], [
                {"city": "42"}, {"city": "7"}, {"city": "42"}
            ])
            config = {"cities": {
                "id_field": "city", "priority_city_csv": str(source),
                "priority_city_ids_json": str(output), "priority_city_expected_count": 2
            }}
            path, count, digest, created = priority.export_priority_city_ids_json(config)
            self.assertEqual((path, count, created), (output.resolve(), 2, True))
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), [7, 42])
            self.assertEqual(priority.export_priority_city_ids_json(config), (path, count, digest, False))
            output.write_text("[99]\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                priority.export_priority_city_ids_json(config)
            self.assertEqual(output.read_text(encoding="utf-8"), "[99]\n")
            output.write_text("[7.0, 42]\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                priority.export_priority_city_ids_json(config)

    def test_export_json_rejects_unexpected_count_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cities.csv"
            output = Path(directory) / "ids.json"
            self.write_csv(source, ["city"], [{"city": "7"}])
            config = {"cities": {
                "id_field": "city", "priority_city_csv": str(source),
                "priority_city_ids_json": str(output), "priority_city_expected_count": 2
            }}
            with self.assertRaisesRegex(ValueError, "Expected 2 priority cities, found 1"):
                priority.export_priority_city_ids_json(config)
            self.assertFalse(output.exists())

    def test_load_json_snapshot_validates_ids_and_preserves_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cities.csv"
            snapshot = root / "ids.json"
            self.write_csv(source, ["city"], [{"city": "42"}, {"city": "7"}])
            config = {"cities": {
                "id_field": "city", "priority_city_csv": str(source),
                "priority_city_ids_json": str(snapshot), "priority_city_expected_count": 2
            }}
            priority.export_priority_city_ids_json(config)
            ids, digest, path = priority.load_priority_city_ids_json(config)
            self.assertEqual((ids, path), ([7, 42], snapshot.resolve()))
            self.assertEqual(digest, priority.load_priority_city_ids(config, "city")[1])
            for value in ([7, 7], [7, True], [7], ["7", 42]):
                snapshot.write_text(json.dumps(value), encoding="utf-8")
                with self.subTest(value=value), self.assertRaises(ValueError):
                    priority.load_priority_city_ids_json(config)


if __name__ == "__main__":
    unittest.main()
