import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import run_export
from src import export_utils
from src import samplepoint_lst_city_exports as exports


def make_config():
    return {
        "samplepoint_lst_city_export": {
            "priority_only": True, "selection": "test_city",
            "city_offset": 0, "max_cities_per_run": 1
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


class SamplepointCityExportTests(unittest.TestCase):
    def test_study_period_helper_reuses_annual_monthly_pipeline(self):
        config = make_config()
        flattened = MagicMock()
        flattened.flatten.return_value = "result"
        with (
            patch.object(export_utils, "build_yearly_samplepoint_lst_statistics", side_effect=["2019", "2020"]) as yearly,
            patch.object(export_utils.ee, "FeatureCollection", return_value=flattened) as collection
        ):
            result = export_utils.build_study_period_samplepoint_lst_statistics(
                points="points", config=config, projection="grid",
                scale=926, id_field="city"
            )
        self.assertEqual(result, "result")
        self.assertEqual([call.kwargs["year"] for call in yearly.call_args_list], [2019, 2020])
        collection.assert_called_once_with(["2019", "2020"])

    def test_priority_test_city_is_one_city_task_for_full_period(self):
        config = make_config()
        all_points = MagicMock()
        city_points = all_points.filter.return_value
        city_points.size.return_value.getInfo.return_value = 3
        digest = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "tasks.jsonl"
            with (
                patch.object(exports, "TASK_LOG_FILE", log_path),
                patch.object(exports, "load_sample_points", return_value=(all_points, 20)),
                patch.object(exports, "load_priority_city_ids", return_value=([7, 42, 99], digest, Path("priority.csv"))),
                patch.object(exports, "filter_priority_sample_points", return_value=("priority_points", 10)) as priority_filter,
                patch.object(exports.ee.Filter, "eq", side_effect=lambda *args: args),
                patch.object(exports, "build_samplepoint_lst_selectors", return_value=["pixel_id", "city"]) as selectors,
                patch.object(exports, "build_study_period_samplepoint_lst_statistics", return_value="result") as build,
                patch.object(exports, "export_table_to_drive", return_value=SimpleNamespace(id="task")) as submit
            ):
                tasks = exports.export_samplepoint_lst_city(config, "grid", 926)
            self.assertEqual(len(tasks), 1)
            priority_filter.assert_called_once_with(all_points, "city", [7, 42, 99])
            all_points.filter.assert_called_once_with(("city", 42))
            build.assert_called_once_with(
                points=city_points, config=config, projection="grid",
                scale=926, id_field="city"
            )
            selectors.assert_called_once_with("city")
            self.assertEqual(
                submit.call_args.kwargs["description"],
                "sample_lst_v0.3_priority_aaaaaaaaaaaa_city_42_2019_2020"
            )
            self.assertEqual(submit.call_args.kwargs["selectors"], ["pixel_id", "city"])
            record = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(
                (record["city_id"], record["city_offset"], record["point_count"]),
                (42, 1, 3)
            )
            self.assertEqual(record["priority_city_ids_sha256"], digest)

    def test_all_cities_switch_and_offset_limit(self):
        config = make_config()
        config["samplepoint_lst_city_export"].update(
            priority_only=False, selection="all", city_offset=1, max_cities_per_run=2
        )
        points = MagicMock()
        points.aggregate_array.return_value.distinct.return_value.sort.return_value.getInfo.return_value = [7, 42, 99, 123]
        points.filter.return_value.size.return_value.getInfo.return_value = 5
        submitted = []
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "tasks.jsonl"
            with (
                patch.object(exports, "TASK_LOG_FILE", log_path),
                patch.object(exports, "load_sample_points", return_value=(points, 5)),
                patch.object(exports, "load_priority_city_ids") as load_priority,
                patch.object(exports.ee.Filter, "notNull", return_value="notnull", create=True),
                patch.object(exports.ee.Filter, "eq", side_effect=lambda *args: args),
                patch.object(exports, "build_samplepoint_lst_selectors", return_value=["pixel_id"]),
                patch.object(exports, "build_study_period_samplepoint_lst_statistics", return_value="result"),
                patch.object(exports, "export_table_to_drive", side_effect=lambda **kw: (submitted.append(kw) or SimpleNamespace(id=f"task{len(submitted)}")))
            ):
                tasks = exports.export_samplepoint_lst_city(config, "grid", 926)
            self.assertEqual(len(tasks), 2)
            load_priority.assert_not_called()
            self.assertEqual(
                [item["description"] for item in submitted],
                ["sample_lst_v0.3_all_city_42_2019_2020", "sample_lst_v0.3_all_city_99_2019_2020"]
            )
            records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["city_offset"] for record in records], [1, 2])
            self.assertEqual([record["city_id"] for record in records], [42, 99])
            self.assertEqual(records[0]["city_queue_sha256"], records[1]["city_queue_sha256"])

    def test_invalid_configuration_stops_before_loading(self):
        config = make_config()
        for field, value in (
            ("priority_only", "true"), ("selection", "invalid"),
            ("city_offset", -1), ("max_cities_per_run", 0)
        ):
            bad = copy.deepcopy(config)
            bad["samplepoint_lst_city_export"][field] = value
            with self.subTest(field=field), patch.object(exports, "load_sample_points") as loader:
                with self.assertRaises(ValueError):
                    exports.export_samplepoint_lst_city(bad, "grid", 926)
                loader.assert_not_called()

    def test_test_city_outside_priority_scope_never_submits(self):
        config = make_config()
        with (
            patch.object(exports, "load_sample_points", return_value=("points", 5)),
            patch.object(exports, "load_priority_city_ids", return_value=([7, 99], "a" * 64, Path("priority.csv"))),
            patch.object(exports, "filter_priority_sample_points", return_value=("filtered", 3)),
            patch.object(exports, "export_table_to_drive") as submit
        ):
            with self.assertRaisesRegex(ValueError, "not present in the selected city scope"):
                exports.export_samplepoint_lst_city(config, "grid", 926)
            submit.assert_not_called()

    def test_router_exposes_city_target(self):
        self.assertIs(run_export.EXPORTERS["samplepoint_lst_city"], exports.export_samplepoint_lst_city)


if __name__ == "__main__":
    unittest.main()
