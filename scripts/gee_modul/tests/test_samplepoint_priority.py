import csv
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


if __name__ == "__main__":
    unittest.main()
