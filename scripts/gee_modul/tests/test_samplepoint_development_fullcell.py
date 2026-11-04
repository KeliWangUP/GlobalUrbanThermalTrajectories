import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import run_export
from src import samplepoint_attributes as attributes
from src import samplepoint_development as development
from src import samplepoint_development_checks as checks
from src import samplepoint_development_exports as exports


def make_config(path: Path) -> dict:
    return {
        "samplepoint_development_export": {
            "dataset_name": "samplepoint_development_fullcell",
            "version": "v0.4",
            "selection": "test_city",
            "city_offset": 0,
            "max_cities_per_run": 1,
            "test_point_limit": 5,
            "test_pixel_id": None,
            "task_log": str(path)
        },
        "samplepoint_attributes": {
            "isa_asset": "isa_asset",
            "isa_years": list(range(1985, 2022)),
            "max_pixels_per_cell": 4096
        },
        "sample_points": {"include_lonlat": True},
        "processing": {"tile_scale": 4},
        "cities": {"id_field": "city"},
        "test": {"city_id": 20},
        "ghsl_built": {"years": [2000, 2005, 2010, 2015, 2020]},
        "export": {"asset_root": "points", "folder": "drive"}
    }


def example_properties() -> dict:
    return {
        "isa_urban_area_m2_1985": 900.0,
        "isa_urban_area_m2_2020": 1800.0,
        "isa_urban_area_m2_2021": 2700.0,
        "isa_urban_fraction_1985": 0.25,
        "isa_urban_fraction_2020": 0.5,
        "isa_urban_fraction_2021": 0.75,
        "water_area_m2": 900.0,
        "water_valid_area_m2": 2700.0,
        "water_fraction": 0.25,
        "built_surface_2020_m2_mean": 1000.0,
        "built_surface_2020_m2_count": 5,
        "built_volume_2020_m3_mean": 2000.0,
        "built_volume_2020_m3_count": 5
    }


class SamplepointFullCellDevelopmentTests(unittest.TestCase):
    def test_aligned_grid_reference_water_area(self):
        water_area = valid_area = 0
        for y in range(2):
            for x in range(2):
                valid = not (x % 2 == 1 and y == 1)
                water = x % 2 == 1 and y == 0
                valid_area += 900 if valid else 0
                water_area += 900 if water and valid else 0
        self.assertEqual((water_area, valid_area, water_area / 3600), (900, 2700, 0.25))

    def test_water_builds_area_numerators_and_independent_coverage(self):
        water = MagicMock()
        pixel_area = MagicMock()
        source_pixel_area = pixel_area.setDefaultProjection.return_value
        masked_water_area = MagicMock()
        masked_valid_area = MagicMock()
        water_area = MagicMock()
        valid_area = MagicMock()
        source_pixel_area.updateMask.side_effect = [
            masked_water_area,
            masked_valid_area
        ]
        masked_water_area.unmask.return_value.rename.return_value = water_area
        masked_valid_area.unmask.return_value.rename.return_value = valid_area
        with (
            patch.object(development.ee, "Image") as image,
        ):
            image.pixelArea.return_value = pixel_area
            image.cat.return_value = "area-image"
            result = development.build_fullcell_water_image(water)
        pixel_area.setDefaultProjection.assert_called_once_with(
            water.projection.return_value
        )
        source_pixel_area.updateMask.assert_has_calls([
            call(water),
            call(water.mask.return_value)
        ])
        masked_water_area.unmask.assert_called_once_with(0)
        masked_water_area.unmask.return_value.rename.assert_called_once_with(
            "water_area_num_m2"
        )
        masked_valid_area.unmask.assert_called_once_with(0)
        masked_valid_area.unmask.return_value.rename.assert_called_once_with(
            "water_valid_area_num_m2"
        )
        image.cat.assert_called_once_with([
            water_area,
            valid_area
        ])
        self.assertEqual(result, "area-image")

    def test_legacy_and_outer_join_keep_their_distinct_modes(self):
        with (
            patch.object(attributes.ee, "Join") as join,
            patch.object(attributes.ee, "Filter"),
            patch.object(attributes.ee, "FeatureCollection")
        ):
            attributes._join_by_point_and_city("a", "b", "city", "match")
            attributes._join_by_point_and_city("a", "b", "city", "match", outer=True)
        self.assertEqual(join.saveFirst.call_args_list, [
            call("match"), call("match", outer=True)
        ])

    def test_water_area_and_ghsl_reference_values(self):
        values = example_properties()
        years = (1985, 2020, 2021)
        checks._validate_development_values(values, years, [2020], 3600, "test")
        cases = [
            {"water_area_m2": 225},
            {"water_valid_area_m2": 4000},
            {"water_fraction": 0.75},
            {"built_surface_2020_m2_count": 0},
            {"built_volume_2020_m3_mean": -1},
        ]
        for change in cases:
            bad = {**values, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                checks._validate_development_values(bad, years, [2020], 3600, "test")
        no_ghsl = {
            **values,
            "built_surface_2020_m2_mean": None,
            "built_surface_2020_m2_count": 0,
            "built_volume_2020_m3_mean": None,
            "built_volume_2020_m3_count": 0
        }
        checks._validate_development_values(no_ghsl, years, [2020], 3600, "test")

    def test_one_city_then_limited_priority_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "tasks.jsonl"
            config = make_config(log)
            digest = "a" * 64
            points = MagicMock()
            city_points = points.filter.return_value
            city_points.size.return_value.getInfo.return_value = 3
            priority_points = MagicMock()
            priority_points.filter.return_value.size.return_value.getInfo.return_value = 10
            submitted = []

            def submit(**kwargs):
                submitted.append(kwargs)
                return SimpleNamespace(id=f"task{len(submitted)}")

            sources = SimpleNamespace(isa_band="isa", isa_years=tuple(range(1985, 2022)))
            with (
                patch.object(exports, "load_priority_city_ids_json", return_value=([10, 20, 30], digest, Path("ids.json"))),
                patch.object(exports, "load_sample_points", return_value=(points, 50)),
                patch.object(exports, "filter_priority_sample_points", return_value=(priority_points, 10)),
                patch.object(exports.ee.Filter, "notNull", return_value="notnull", create=True),
                patch.object(exports.ee.Filter, "eq", side_effect=lambda *args: args) as eq,
                patch.object(exports, "prepare_samplepoint_sources", return_value=sources),
                patch.object(exports, "prepare_fullcell_development_images", return_value="images"),
                patch.object(exports, "sample_fullcell_development", return_value="collection") as sample,
                patch.object(exports, "export_table_to_drive", side_effect=submit)
            ):
                tasks = exports.export_samplepoint_development(config, "projection", 926)
                self.assertEqual(len(tasks), 1)
                eq.assert_called_with("city", 20)
                sample.assert_called_with(city_points, config, "projection", 926, "images")
                self.assertEqual(
                    submitted[0]["description"],
                    "samplepoint_development_fullcell_v0.4_priority_aaaaaaaaaaaa_city_20"
                )
                self.assertEqual(len(submitted[0]["selectors"]), 103)
                self.assertIn("isa_urban_area_m2_2021", submitted[0]["selectors"])
                record = json.loads(log.read_text(encoding="utf-8"))
                self.assertEqual((record["city_id"], record["city_offset"]), (20, 1))
                self.assertEqual(record["water_denominator"], "full_modis_cell_masked_as_zero")
                config["samplepoint_development_export"].update(
                    selection="all", city_offset=0, max_cities_per_run=1
                )
                exports.export_samplepoint_development(config, "projection", 926)
                eq.assert_called_with("city", 10)
                second = json.loads(log.read_text(encoding="utf-8").splitlines()[1])
                self.assertEqual((second["city_id"], second["city_offset"]), (10, 0))
                config["samplepoint_development_export"]["city_offset"] = 3
                with self.assertRaisesRegex(ValueError, "outside"):
                    exports.export_samplepoint_development(config, "projection", 926)
                self.assertEqual(len(submitted), 2)

    def test_missing_priority_properties_stop_before_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory) / "tasks.jsonl")
            priority_points = MagicMock()
            priority_points.filter.return_value.size.return_value.getInfo.return_value = 9
            with (
                patch.object(exports, "load_priority_city_ids_json", return_value=([20], "a" * 64, Path("ids.json"))),
                patch.object(exports, "load_sample_points", return_value=("points", 50)),
                patch.object(exports, "filter_priority_sample_points", return_value=(priority_points, 10)),
                patch.object(exports.ee.Filter, "notNull", return_value="notnull", create=True),
                patch.object(exports, "export_table_to_drive") as submit
            ):
                with self.assertRaisesRegex(ValueError, "missing required"):
                    exports.export_samplepoint_development(config, "projection", 926)
                submit.assert_not_called()

    def test_city_calculation_checks_values_without_submitting_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory) / "tasks.jsonl")
            config["ghsl_built"]["years"] = [2020]
            points = MagicMock()
            city_points = points.filter.return_value
            city_points.size.return_value.getInfo.return_value = 1
            selected = city_points.sort.return_value.limit.return_value
            selected.aggregate_array.return_value.getInfo.return_value = ["MODIS_1_2"]
            values = {**example_properties(), "pixel_id": "MODIS_1_2", "city": 20}
            result = MagicMock()
            result.getInfo.return_value = {"features": [{"properties": values}]}
            area_result = MagicMock()
            area_result.getInfo.return_value = {"features": [{
                "properties": {
                    "pixel_id": "MODIS_1_2",
                    "city": 20,
                    "modis_area_m2": 3600
                }
            }]}
            sources = SimpleNamespace(isa_years=(1985, 2020, 2021))
            with (
                patch.object(checks, "load_priority_city_ids_json", return_value=([20], "digest", Path("ids.json"))),
                patch.object(checks, "check_synthetic_development") as development_check,
                patch.object(checks, "load_sample_points", return_value=(points, 1)),
                patch.object(checks.ee.Filter, "eq", return_value="city_filter"),
                patch.object(checks, "prepare_samplepoint_sources", return_value=sources),
                patch.object(checks, "prepare_fullcell_development_images", return_value="images"),
                patch.object(checks, "sample_fullcell_development", return_value=result),
                patch.object(checks, "sample_modis_pixel_area", return_value=area_result)
            ):
                checks.check_samplepoint_development_city(config, "projection", 926)
                development_check.assert_not_called()
                checks.sample_modis_pixel_area.assert_called_once_with(
                    selected, ["pixel_id", "city"], "projection", 926, 4
                )
                result.getInfo.return_value = {"features": []}
                with self.assertRaisesRegex(ValueError, "did not preserve"):
                    checks.check_samplepoint_development_city(config, "projection", 926)

    def test_router_exposes_separate_development_target(self):
        self.assertIs(
            run_export.EXPORTERS["samplepoint_development"],
            exports.export_samplepoint_development
        )
        self.assertIn("samplepoint_attributes", run_export.EXPORTERS)


if __name__ == "__main__":
    unittest.main()
