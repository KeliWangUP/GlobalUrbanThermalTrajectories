import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src import samplepoint_attribute_exports as exports
from src import samplepoint_recovery as recovery
from src import samplepoint_utils
import export_samplepoint_lst as annual_lst
import export_samplepoint_lst_month as monthly_lst
import run_export


class SamplepointRefactorTests(unittest.TestCase):
    def test_existing_failure_list(self):
        batches = recovery.read_failed_batches(recovery.FAILED_BATCH_FILE)
        self.assertTrue(batches)
        self.assertEqual(len({batch[0] for batch in batches}), len(batches))
        self.assertEqual(len({batch[1] for batch in batches}), 1)

    def test_parser_validates_legacy_and_multiple_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed.json"
            path.write_text(
                json.dumps({"batch": "test_city_batch_0002_of_0003"}),
                encoding="utf-8"
            )
            self.assertEqual(recovery.read_failed_batch(path)[:2], (2, 3))
            for names in (
                ["test_city_batch_0002_of_0003"] * 2,
                ["test_city_batch_0002_of_0003", "test_city_batch_0001_of_0004"],
                ["test_city_batch_0004_of_0003"],
                ["not_a_batch"],
            ):
                path.write_text(json.dumps({"batches": names}), encoding="utf-8")
                with self.subTest(names=names), self.assertRaises(ValueError):
                    recovery.read_failed_batches(path)

    def test_failed_city_mapping_preserves_original_sorted_order(self):
        points = MagicMock()
        points.aggregate_array.return_value.distinct.return_value.sort.return_value.getInfo.return_value = list(range(1, 8))
        config = {
            "export": {"asset_root": "points"},
            "cities": {"id_field": "city"},
            "sample_points": {"city_batch_size": 3}
        }
        with patch.object(recovery.ee, "FeatureCollection", return_value=points):
            ids = recovery.get_failed_city_ids(
                config, [(3, 3, "last"), (1, 3, "first")]
            )
            self.assertEqual(ids, [1, 2, 3, 7])
            with self.assertRaises(ValueError):
                recovery.get_failed_city_ids(config, [(1, 2, "wrong total")])

    def test_shared_point_loader_checks_input(self):
        points = MagicMock()
        points.size.return_value.getInfo.return_value = 2
        feature = MagicMock()
        feature.propertyNames.return_value.getInfo.return_value = ["pixel_id", "city"]
        config = {"export": {"asset_root": "points"}}
        with (
            patch.object(samplepoint_utils, "asset_exists", return_value=True),
            patch.object(samplepoint_utils.ee, "FeatureCollection", return_value=points),
            patch.object(samplepoint_utils.ee, "Feature", return_value=feature)
        ):
            self.assertEqual(
                samplepoint_utils.load_sample_points(config, ["pixel_id", "city"]),
                (points, 2)
            )
            with self.assertRaisesRegex(ValueError, "missing required"):
                samplepoint_utils.load_sample_points(config, ["modis_x"])
            points.size.return_value.getInfo.return_value = 0
            with self.assertRaisesRegex(ValueError, "empty"):
                samplepoint_utils.load_sample_points(config, ["pixel_id"])
        with patch.object(samplepoint_utils, "asset_exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                samplepoint_utils.load_sample_points(config, ["pixel_id"])

    def test_city_exports_keep_schema_names_and_callback(self):
        config = {
            "export": {
                "asset_root": "points",
                "samplepoint_attributes_dataset_name": "sample",
                "folder": "folder"
            },
            "project": {"version": "v0.3"},
            "cities": {"id_field": "city"},
            "sample_points": {"city_batch_size": 2, "include_lonlat": True}
        }
        points = MagicMock()
        points.aggregate_array.return_value.distinct.return_value.sort.return_value.getInfo.return_value = [10, 20, 30]
        submitted = []

        def capture_task(**kwargs):
            submitted.append(kwargs)
            return SimpleNamespace(id=f"task{len(submitted)}")

        callback = MagicMock()
        with (
            patch.object(exports, "load_sample_points", return_value=(points, 3)) as loader,
            patch.object(exports, "build_samplepoint_selectors", return_value=["pixel_id", "city"]) as selectors,
            patch.object(exports, "prepare_samplepoint_sources", return_value=SimpleNamespace(isa_band="isa")),
            patch.object(exports.ee.Filter, "inList", side_effect=lambda *args: args),
            patch.object(exports, "build_samplepoint_attribute_collection", return_value="collection"),
            patch.object(exports, "export_table_to_drive", side_effect=capture_task)
        ):
            tasks = exports.export_samplepoint_attributes(
                config, "projection", 1, expected_batch_count=2,
                selected_city_ids=[10, 30], on_city_task_submitted=callback
            )
            self.assertEqual(len(tasks), 2)
            self.assertEqual(
                [item["description"] for item in submitted],
                ["sample_v0.3_city_10_retry", "sample_v0.3_city_30_retry"]
            )
            self.assertTrue(all(item["selectors"] == ["pixel_id", "city"] for item in submitted))
            self.assertEqual([call.args[0] for call in callback.call_args_list], [10, 30])
            loader.assert_called_once_with(
                config, ["pixel_id", "city", "modis_x", "modis_y", "lon", "lat"]
            )
            selectors.assert_called_once_with(config)
            submitted.clear()
            exports.export_samplepoint_attributes(
                config, "projection", 1, selected_batch_index=2,
                expected_batch_count=2
            )
            self.assertEqual(
                submitted[0]["description"],
                "sample_v0.3_city_batch_0002_of_0002_retry"
            )
            self.assertEqual(points.filter.call_args.args[0], ("city", [30]))
            submitted.clear()
            exports.export_samplepoint_attributes(
                config, "projection", 1, selected_city_id=20
            )
            self.assertEqual(
                submitted[0]["description"], "sample_v0.3_city_20_probe"
            )
            self.assertEqual(points.filter.call_args.args[0], ("city", [20]))

    def test_archived_scripts_import_without_submitting_tasks(self):
        for name in (
            "export_samplepoint_attributes_failed_batch.py",
            "export_samplepoint_attributes_failed_city_probe.py"
        ):
            with self.subTest(name=name):
                namespace = runpy.run_path(str(MODULE_DIR / "archive" / name))
                self.assertTrue(callable(namespace["main"]))

    def test_run_export_uses_reusable_attribute_exporter(self):
        self.assertIs(
            run_export.EXPORTERS["samplepoint_attributes"],
            exports.export_samplepoint_attributes
        )

    def test_annual_and_monthly_lst_keep_output_contract(self):
        config = {
            "export": {
                "asset_root": "points",
                "samplepoint_lst_dataset_name": "sample_lst",
                "folder": "folder"
            },
            "project": {"version": "v0.3"},
            "cities": {"id_field": "city"},
            "study": {"start_year": 2020, "end_year": 2020},
            "samplepoint_lst_export": {"year": 2020, "month": 1}
        }
        points = MagicMock()
        points.aggregate_array.return_value.distinct.return_value.size.return_value.getInfo.return_value = 1
        selectors = ["pixel_id", "city", "year", "month", "terra_day_mean"]
        for module, function, expected_name in (
            (annual_lst, annual_lst.export_samplepoint_lst, "sample_lst_v0.3_2020"),
            (monthly_lst, monthly_lst.export_samplepoint_lst_month, "sample_lst_v0.3_2020_01"),
        ):
            with (
                self.subTest(module=module.__name__),
                patch.object(module, "load_sample_points", return_value=(points, 2)) as loader,
                patch.object(module, "build_samplepoint_lst_selectors", return_value=selectors),
                patch.object(module, "export_table_to_drive", return_value="task") as submit,
                patch.object(module, "build_yearly_samplepoint_lst_statistics", return_value="collection")
                if module is annual_lst else
                patch.object(module, "build_monthly_samplepoint_lst_statistics", return_value="collection")
            ):
                function(config, "projection", 1)
                loader.assert_called_once_with(config, ["pixel_id", "city"])
                self.assertEqual(submit.call_args.kwargs["description"], expected_name)
                self.assertEqual(submit.call_args.kwargs["selectors"], selectors)


if __name__ == "__main__":
    unittest.main()
