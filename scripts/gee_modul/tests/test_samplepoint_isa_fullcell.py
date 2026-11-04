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
from src import samplepoint_isa as isa
from src import samplepoint_isa_checks as checks
from src import samplepoint_isa_exports as exports
from src import samplepoint_attributes as attributes
from src.samplepoint_utils import partition_samplepoint_cities


def make_config(log_path: Path) -> dict:
    return {
        "samplepoint_isa_export": {
            "dataset_name": "samplepoint_isa_fullcell",
            "version": "v0.4",
            "selection": "test_city",
            "batch_offset": 0,
            "max_batches_per_run": 1,
            "test_point_limit": 5,
            "test_pixel_id": None,
            "task_log": str(log_path)
        },
        "samplepoint_attributes": {
            "isa_asset": "isa_asset",
            "max_pixels_per_cell": 4096
        },
        "sample_points": {"city_batch_size": 2, "include_lonlat": True},
        "processing": {"tile_scale": 4},
        "cities": {"id_field": "city"},
        "test": {"city_id": 20},
        "export": {"asset_root": "points", "folder": "drive"}
    }


class SamplepointFullCellISATests(unittest.TestCase):
    def test_synthetic_grid_uses_supported_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory) / "tasks.jsonl")
            with (
                patch.object(checks.ee, "Projection") as projection,
                patch.object(checks.ee, "Image"),
                patch.object(checks.ee, "Geometry") as geometry,
                patch.object(checks.ee, "Feature"),
                patch.object(checks.ee, "FeatureCollection")
            ):
                checks.synthetic_fullcell_grid(config)
            self.assertEqual(projection.call_args_list, [
                call("EPSG:3857", [30, 0, 0, 0, -30, 0]),
                call("EPSG:3857", [60, 0, 0, 0, -60, 0]),
                call("EPSG:3857")
            ])
            geometry.Point.assert_called_once_with([30, -30], proj=projection.return_value)

    def test_aligned_grid_expected_area_fractions(self):
        source_pixel_area = 30 * 30
        weighted_areas = {1985: 0, 2020: 0, 2021: 0}
        for y in range(2):
            for x in range(2):
                category = (
                    38 if (x, y) == (0, 0)
                    else 3 if (x, y) == (1, 0)
                    else 2 if (x, y) == (0, 1)
                    else 0
                )
                if (x, y) == (1, 1):
                    category = None
                for year in weighted_areas:
                    if category is not None and category >= 2023 - year:
                        weighted_areas[year] += source_pixel_area
        self.assertEqual(
            {year: area / (4 * source_pixel_area) for year, area in weighted_areas.items()},
            {1985: 0.25, 2020: 0.5, 2021: 0.75}
        )

    def test_masked_annual_indicators_become_area_numerators(self):
        source = MagicMock()
        indicators = [MagicMock(), MagicMock(), MagicMock()]
        source.gte.side_effect = indicators
        pixel_area = MagicMock()
        with (
            patch.object(isa.ee, "Image") as image,
        ):
            image.pixelArea.return_value = pixel_area
            image.cat.return_value = "area-image"
            result = isa.build_fullcell_isa_image(
                source, (1985, 2020, 2021), 4096
            )
        self.assertEqual(source.gte.call_args_list, [call(38), call(3), call(2)])
        self.assertEqual(source.lte.call_args_list, [call(38), call(38), call(38)])
        for indicator, year in zip(indicators, (1985, 2020, 2021)):
            indicator.And.assert_called_once()
            indicator.And.return_value.unmask.assert_called_once_with(
                0, sameFootprint=False
            )
            indicator.And.return_value.unmask.return_value.multiply.assert_called_once_with(
                pixel_area
            )
            indicator.And.return_value.unmask.return_value.multiply.return_value.rename.assert_called_once_with(
                f"isa_area_m2_{year}"
            )
        self.assertEqual(result, "area-image")

    def test_fraction_reference_cases_and_invalid_inputs(self):
        checks._validate_fractions({
            "isa_urban_fraction_1985": 0.25,
            "isa_urban_fraction_2020": 0.5,
            "isa_urban_fraction_2021": 0.75
        }, (1985, 2020, 2021), "example")
        for values in (
            (0.25, None, 0.75), (0.5, 0.4, 0.75),
            (0.25, float("nan"), 0.75), (0.25, 0.5, 1.1)
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                checks._validate_fractions(
                    dict(zip(
                        ["isa_urban_fraction_1985", "isa_urban_fraction_2020", "isa_urban_fraction_2021"],
                        values
                    )),
                    (1985, 2020, 2021),
                    "example"
                )
        for years in ((), (2020, 1985), (1985, 1985), (1984,), (2022,)):
            with self.subTest(years=years), self.assertRaises(ValueError):
                isa.build_fullcell_isa_image(MagicMock(), years, 4096)
        with self.assertRaises(ValueError):
            isa.build_fullcell_isa_image(MagicMock(), (2020,), 0)

    def test_selectors_and_shared_city_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory) / "tasks.jsonl")
            fields = isa.build_samplepoint_isa_selectors(config, (1985, 2021))
            self.assertEqual(fields, [
                "pixel_id", "city", "modis_x", "modis_y", "lon", "lat",
                "isa_urban_fraction_1985", "isa_urban_fraction_2021"
            ])
            config["sample_points"]["include_lonlat"] = False
            self.assertNotIn("lon", isa.build_samplepoint_isa_selectors(config, (2021,)))
            full = isa.build_samplepoint_isa_selectors(config, tuple(range(1985, 2022)))
            self.assertEqual(len(full), 41)
            self.assertEqual(full[4], "isa_urban_fraction_1985")
            self.assertEqual(full[-1], "isa_urban_fraction_2021")
            self.assertEqual(
                partition_samplepoint_cities([10, 20, 30, 40, 50], 2),
                [[10, 20], [30, 40], [50]]
            )
            for bad_size in (0, True, 2.5):
                with self.subTest(size=bad_size), self.assertRaises(ValueError):
                    partition_samplepoint_cities([10], bad_size)

    def test_prepare_isa_source_does_not_load_ghsl_or_water(self):
        config = {
            "samplepoint_attributes": {
                "isa_asset": "isa_asset", "isa_band": None,
                "isa_years": list(range(1985, 2022))
            }
        }
        image = MagicMock()
        image.bandNames.return_value.getInfo.return_value = ["isa_band"]
        with patch.object(attributes.ee, "Image", return_value=image) as create:
            selected, band, years = attributes.prepare_isa_source(config)
        create.assert_called_once_with("isa_asset")
        image.select.assert_called_once_with("isa_band")
        self.assertEqual((selected, band, years), (
            image.select.return_value, "isa_band", tuple(range(1985, 2022))
        ))

    def test_one_city_task_and_batch_window(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "tasks.jsonl"
            config = make_config(log_path)
            points = MagicMock()
            points.aggregate_array.return_value.distinct.return_value.sort.return_value.getInfo.return_value = [10, 20, 30, 40, 50]
            valid = MagicMock()
            valid.size.return_value.getInfo.return_value = 10
            chosen = MagicMock()
            chosen.size.return_value.getInfo.return_value = 3
            points.filter.side_effect = lambda value: valid if value == "notnull" else chosen
            submitted = []

            def submit(**kwargs):
                submitted.append(kwargs)
                return SimpleNamespace(id=f"task{len(submitted)}")

            with (
                patch.object(exports, "load_sample_points", return_value=(points, 10)),
                patch.object(exports.ee.Filter, "notNull", return_value="notnull", create=True),
                patch.object(exports.ee.Filter, "inList", side_effect=lambda *args: args) as in_list,
                patch.object(exports, "prepare_isa_source", return_value=("isa", "band", (1985, 2020, 2021))),
                patch.object(exports, "build_fullcell_isa_image", return_value="image") as build,
                patch.object(exports, "sample_fullcell_isa", return_value="collection") as sample,
                patch.object(exports, "export_table_to_drive", side_effect=submit)
            ):
                tasks = exports.export_samplepoint_isa(config, "projection", 926)
                self.assertEqual(len(tasks), 1)
                self.assertEqual(
                    submitted[0]["description"],
                    "samplepoint_isa_fullcell_v0.4_city_20_test"
                )
                in_list.assert_called_with("city", [20])
                build.assert_called_with("isa", (1985, 2020, 2021), 4096)
                sample.assert_called_with(
                    "image", chosen, config, "projection", 926, (1985, 2020, 2021)
                )
                self.assertEqual(submitted[0]["selectors"][-3:], [
                    "isa_urban_fraction_1985", "isa_urban_fraction_2020", "isa_urban_fraction_2021"
                ])
                record = json.loads(log_path.read_text(encoding="utf-8"))
                self.assertEqual((record["task_id"], record["city_ids"]), ("task1", [20]))
                self.assertEqual(record["denominator"], "full_modis_cell_masked_as_zero")
                config["samplepoint_isa_export"].update(
                    selection="batches", batch_offset=1, max_batches_per_run=1
                )
                exports.export_samplepoint_isa(config, "projection", 926)
                self.assertEqual(
                    submitted[1]["description"],
                    "samplepoint_isa_fullcell_v0.4_city_batch_0002_of_0003"
                )
                in_list.assert_called_with("city", [30, 40])
                record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[1])
                self.assertEqual((record["batch_index"], record["batch_count"]), (2, 3))
                config["samplepoint_isa_export"]["batch_offset"] = 3
                with self.assertRaisesRegex(ValueError, "outside"):
                    exports.export_samplepoint_isa(config, "projection", 926)
                self.assertEqual(len(submitted), 2)

    def test_missing_point_fields_stop_before_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory) / "tasks.jsonl")
            points = MagicMock()
            points.filter.return_value.size.return_value.getInfo.return_value = 9
            with (
                patch.object(exports, "load_sample_points", return_value=(points, 10)),
                patch.object(exports.ee.Filter, "notNull", return_value="notnull", create=True),
                patch.object(exports, "export_table_to_drive") as submit
            ):
                with self.assertRaisesRegex(ValueError, "missing required"):
                    exports.export_samplepoint_isa(config, "projection", 926)
                submit.assert_not_called()

    def test_city_check_preserves_points_and_validates_actual_values(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory) / "tasks.jsonl")
            points = MagicMock()
            city_points = points.filter.return_value
            city_points.size.return_value.getInfo.return_value = 1
            selected = city_points.sort.return_value.limit.return_value
            selected.aggregate_array.return_value.getInfo.return_value = ["MODIS_1_2"]
            values = {
                "pixel_id": "MODIS_1_2", "city": 20,
                "isa_urban_fraction_1985": 0.0,
                "isa_urban_fraction_2000": 0.1,
                "isa_urban_fraction_2020": 0.2,
                "isa_urban_fraction_2021": 0.3
            }
            sample = MagicMock()
            sample.getInfo.return_value = {"features": [{"properties": values}]}
            with (
                patch.object(checks, "check_synthetic_fullcell") as synthetic,
                patch.object(checks, "load_sample_points", return_value=(points, 1)),
                patch.object(checks.ee.Filter, "eq", return_value="city_filter"),
                patch.object(checks, "prepare_isa_source", return_value=("isa", "band", (1985, 2000, 2020, 2021))),
                patch.object(checks, "build_fullcell_isa_image", return_value="image"),
                patch.object(checks, "sample_fullcell_isa", return_value=sample)
            ):
                checks.check_samplepoint_isa_city(config, "projection", 926)
                synthetic.assert_called_once_with(config)
                sample.getInfo.return_value = {"features": []}
                with self.assertRaisesRegex(ValueError, "did not preserve"):
                    checks.check_samplepoint_isa_city(config, "projection", 926)

    def test_router_exposes_isa_without_changing_original_target(self):
        self.assertIs(run_export.EXPORTERS["samplepoint_isa"], exports.export_samplepoint_isa)
        self.assertIn("samplepoint_attributes", run_export.EXPORTERS)


if __name__ == "__main__":
    unittest.main()
