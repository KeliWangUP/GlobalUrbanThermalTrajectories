import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src import samplepoint_lst_spatial_exports as exports
from src import samplepoint_spatial as spatial


class SpatialPartitionTests(unittest.TestCase):
    def test_half_open_tile_filters_partition_negative_and_boundary_points(self):
        class FakeFilters:
            @staticmethod
            def gte(field, value):
                return lambda point: point[field] >= value

            @staticmethod
            def lt(field, value):
                return lambda point: point[field] < value

            @staticmethod
            def And(*filters):
                return lambda point: all(predicate(point) for predicate in filters)

        class FakePoints:
            def __init__(self, points):
                self.points = points

            def filter(self, predicate):
                return [point for point in self.points if predicate(point)]

        points = FakePoints([
            {"modis_x": x, "modis_y": y}
            for x in (-641, -640, -1, 0, 639, 640)
            for y in (-640, -1, 0, 639, 640)
        ])
        with patch.object(spatial.ee, "Filter", FakeFilters):
            assigned = []
            for column in range(-2, 2):
                for row in range(-1, 2):
                    tile = spatial.filter_tile(points, column, row, 640)
                    for point in tile:
                        self.assertEqual(point["modis_x"] // 640, column)
                        self.assertEqual(point["modis_y"] // 640, row)
                    assigned.extend(tile)
        self.assertEqual(len(assigned), len(points.points))
        self.assertEqual(len({(p["modis_x"], p["modis_y"]) for p in assigned}), len(points.points))

    def test_test_city_coordinate_uses_floor_for_negative_values(self):
        points = MagicMock()
        points.filter.return_value.size.return_value.getInfo.return_value = 1
        feature = MagicMock()
        feature.toDictionary.return_value.getInfo.return_value = {
            "modis_x": -1, "modis_y": -640
        }
        with (
            patch.object(spatial.ee.Filter, "eq", return_value="filter"),
            patch.object(spatial.ee, "Feature", return_value=feature)
        ):
            self.assertEqual(spatial.tile_for_test_city(points, "city", 42, 640), (-1, -1))

    def test_occupied_tiles_are_sorted_and_validate_coverage(self):
        points = MagicMock()
        points.filter.return_value.size.return_value.getInfo.return_value = 4
        info = points.map.return_value.aggregate_histogram.return_value.getInfo
        info.return_value = {"2,-1": 1, "-1,0": 3}
        with patch.object(spatial.ee.Filter, "notNull", return_value="filter", create=True):
            self.assertEqual(spatial.get_occupied_tiles(points, 4, 640), [(-1, 0), (2, -1)])
            info.return_value = {"2,-1": 1}
            with self.assertRaisesRegex(ValueError, "every sample point"):
                spatial.get_occupied_tiles(points, 4, 640)
            points.filter.return_value.size.return_value.getInfo.return_value = 3
            with self.assertRaisesRegex(ValueError, "missing MODIS"):
                spatial.get_occupied_tiles(points, 4, 640)

    def test_one_tile_exports_entire_configured_period_and_logs_submission(self):
        config = {
            "samplepoint_lst_spatial_export": {
                "tile_pixels": 640, "selection": "test_city",
                "city_scope": "all",
                "tile_column": None, "tile_row": None,
                "tile_offset": 0, "max_tiles_per_run": 1
            },
            "study": {"start_year": 2019, "end_year": 2020},
            "cities": {"id_field": "city"},
            "test": {"city_id": 42},
            "project": {"version": "v0.3"},
            "export": {
                "asset_root": "points",
                "samplepoint_lst_dataset_name": "sample_lst",
                "folder": "folder"
            }
        }
        tile_points = MagicMock()
        tile_points.size.return_value.getInfo.return_value = 3
        task = SimpleNamespace(id="task1")
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "tasks.jsonl"
            with (
                patch.object(exports, "TASK_LOG_FILE", log_path),
                patch.object(exports, "load_sample_points", return_value=("points", 5)),
                patch.object(exports, "tile_for_test_city", return_value=(-1, 2)),
                patch.object(exports, "filter_tile", return_value=tile_points) as select,
                patch.object(exports, "build_samplepoint_lst_selectors", return_value=["pixel_id", "city"]),
                patch.object(exports, "build_study_period_samplepoint_lst_statistics", return_value="collection") as study,
                patch.object(exports, "export_table_to_drive", return_value=task) as submit
            ):
                tasks = exports.export_samplepoint_lst_spatial(config, "projection", 926)
            self.assertEqual(tasks, [task])
            select.assert_called_once_with("points", -1, 2, 640)
            study.assert_called_once_with(
                points=tile_points, config=config, projection="projection",
                scale=926, id_field="city"
            )
            self.assertEqual(submit.call_args.kwargs["collection"], "collection")
            self.assertEqual(submit.call_args.kwargs["selectors"], ["pixel_id", "city"])
            self.assertEqual(
                submit.call_args.kwargs["description"],
                "sample_lst_v0.3_tile_640_xm1_yp2_2019_2020"
            )
            record = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(
                (record["task_id"], record["tile_column"], record["tile_row"], record["point_count"]),
                ("task1", -1, 2, 3)
            )

    def test_invalid_configuration_rejected_before_loading_points(self):
        config = {
            "samplepoint_lst_spatial_export": {
                "tile_pixels": 0, "selection": "test_city",
                "city_scope": "all",
                "tile_offset": 0, "max_tiles_per_run": 1
            }
        }
        with patch.object(exports, "load_sample_points") as loader:
            with self.assertRaises(ValueError):
                exports.export_samplepoint_lst_spatial(config, "projection", 926)
            loader.assert_not_called()

    def test_all_selection_submits_only_requested_tile_window(self):
        config = {
            "samplepoint_lst_spatial_export": {
                "tile_pixels": 640, "selection": "all",
                "city_scope": "all",
                "tile_offset": 1, "max_tiles_per_run": 1
            },
            "study": {"start_year": 2020, "end_year": 2020},
            "cities": {"id_field": "city"},
            "project": {"version": "v0.3"},
            "export": {
                "asset_root": "points",
                "samplepoint_lst_dataset_name": "sample_lst",
                "folder": "folder"
            }
        }
        tile_points = MagicMock()
        tile_points.size.return_value.getInfo.return_value = 1
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(exports, "TASK_LOG_FILE", Path(directory) / "tasks.jsonl"),
                patch.object(exports, "load_sample_points", return_value=("points", 3)),
                patch.object(exports, "get_occupied_tiles", return_value=[(-1, 0), (0, 0), (1, 0)]),
                patch.object(exports, "filter_tile", return_value=tile_points) as select,
                patch.object(exports, "build_samplepoint_lst_selectors", return_value=["pixel_id"]),
                patch.object(exports, "build_study_period_samplepoint_lst_statistics", return_value="collection"),
                patch.object(exports, "export_table_to_drive", return_value=SimpleNamespace(id="task1"))
            ):
                exports.export_samplepoint_lst_spatial(config, "projection", 926)
                select.assert_called_once_with("points", 0, 0, 640)
                config["samplepoint_lst_spatial_export"]["tile_offset"] = 3
                with self.assertRaisesRegex(ValueError, "outside"):
                    exports.export_samplepoint_lst_spatial(config, "projection", 926)

    def test_priority_scope_filters_before_selecting_tile_and_marks_output(self):
        config = {
            "samplepoint_lst_spatial_export": {
                "tile_pixels": 640, "selection": "test_city",
                "city_scope": "priority", "tile_offset": 0, "max_tiles_per_run": 1
            },
            "study": {"start_year": 2020, "end_year": 2020},
            "cities": {"id_field": "city"},
            "test": {"city_id": 42},
            "project": {"version": "v0.3"},
            "export": {
                "asset_root": "points",
                "samplepoint_lst_dataset_name": "sample_lst",
                "folder": "folder"
            }
        }
        tile_points = MagicMock()
        tile_points.size.return_value.getInfo.return_value = 2
        digest = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "tasks.jsonl"
            with (
                patch.object(exports, "TASK_LOG_FILE", log_path),
                patch.object(exports, "load_sample_points", return_value=("all_points", 10)),
                patch.object(exports, "load_priority_city_ids", return_value=([42, 43], digest, Path("priority.csv"))),
                patch.object(exports, "filter_priority_sample_points", return_value=("priority_points", 3)) as filter_priority,
                patch.object(exports, "tile_for_test_city", return_value=(1, 2)) as tile_for_city,
                patch.object(exports, "filter_tile", return_value=tile_points) as tile_filter,
                patch.object(exports, "build_samplepoint_lst_selectors", return_value=["pixel_id"]),
                patch.object(exports, "build_study_period_samplepoint_lst_statistics", return_value="collection") as study,
                patch.object(exports, "export_table_to_drive", return_value=SimpleNamespace(id="task")) as submit
            ):
                exports.export_samplepoint_lst_spatial(config, "projection", 926)
            filter_priority.assert_called_once_with("all_points", "city", [42, 43])
            tile_for_city.assert_called_once_with("priority_points", "city", 42, 640)
            tile_filter.assert_called_once_with("priority_points", 1, 2, 640)
            self.assertIs(study.call_args.kwargs["points"], tile_points)
            self.assertEqual(
                submit.call_args.kwargs["description"],
                "sample_lst_v0.3_priority_aaaaaaaaaaaa_tile_640_xp1_yp2_2020_2020"
            )
            record = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(record["city_scope"], "priority")
            self.assertEqual(record["priority_city_ids_sha256"], digest)


if __name__ == "__main__":
    unittest.main()
