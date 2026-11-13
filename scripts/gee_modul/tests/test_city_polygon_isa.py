import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src import city_polygon_isa as city_isa
from src import city_polygon_isa_exports as exports


def make_config(task_log: Path) -> dict:
    return {
        "city_polygon_isa": {
            "years": list(range(2000, 2021)),
            "scale_m": 30,
            "area_max_error_m": 10,
            "dataset_name": "city_isa_polygon",
            "version": "v0.1",
            "task_log": str(task_log)
        },
        "cities": {
            "asset": "city_boundaries",
            "id_field": "ID_UC_G0"
        },
        "samplepoint_attributes": {
            "isa_asset": "gaia_isa",
            "isa_band": "b1",
            "isa_years": list(range(1985, 2022))
        },
        "processing": {"tile_scale": 4},
        "export": {"folder": "drive"}
    }


class CityPolygonISATests(unittest.TestCase):
    def test_selectors_include_polygon_area_and_annual_area_fractions(self):
        config = make_config(Path("unused.jsonl"))
        years, selectors = city_isa.validate_city_polygon_isa_config(config)

        self.assertEqual(years, tuple(range(2000, 2021)))
        self.assertEqual(selectors[:2], [
            "ID_UC_G0", "city_polygon_area_m2"
        ])
        self.assertEqual(selectors[2], "isa_urban_area_m2_2000")
        self.assertEqual(selectors[22], "isa_urban_area_m2_2020")
        self.assertEqual(selectors[23], "isa_urban_fraction_2000")
        self.assertEqual(selectors[-1], "isa_urban_fraction_2020")
        self.assertEqual(len(selectors), 44)

    def test_rejects_nonannual_or_incomplete_year_ranges(self):
        for years in (
            list(range(2000, 2020)),
            [2000, 2001, 2003],
            list(range(1985, 2022))
        ):
            config = make_config(Path("unused.jsonl"))
            config["city_polygon_isa"]["years"] = years
            with self.subTest(years=years), self.assertRaisesRegex(
                ValueError, "2000 through 2020"
            ):
                city_isa.validate_city_polygon_isa_config(config)

    def test_area_bands_use_gaia_thresholds_and_zero_masked_pixels(self):
        source = MagicMock()
        indicators = [MagicMock() for _ in range(21)]
        source.gte.side_effect = indicators
        pixel_area = MagicMock()

        with patch.object(city_isa.ee, "Image") as image:
            image.pixelArea.return_value = pixel_area
            image.cat.return_value = "area-image"
            result = city_isa.build_city_polygon_isa_area_image(
                source, tuple(range(2000, 2021))
            )

        self.assertEqual(result, "area-image")
        self.assertEqual(
            source.gte.call_args_list,
            [call(2023 - year) for year in range(2000, 2021)]
        )
        self.assertEqual(source.lte.call_args_list, [call(38)] * 21)
        bands = []
        for indicator, year in zip(indicators, range(2000, 2021)):
            indicator.And.assert_called_once()
            masked = indicator.And.return_value.unmask
            masked.assert_called_once_with(0, sameFootprint=False)
            area_band = masked.return_value.multiply.return_value.rename
            area_band.assert_called_once_with(f"isa_urban_area_m2_{year}")
            bands.append(area_band.return_value)
        image.pixelArea.assert_called_once_with()
        image.cat.assert_called_once_with(bands)

    def test_fraction_denominator_is_whole_city_polygon_area(self):
        class FakeNumber:
            def __init__(self, value):
                self.value = value

            def divide(self, denominator):
                return FakeNumber(self.value / denominator.value)

        class FakeGeometry:
            @staticmethod
            def area(maxError):
                return 1000.0

        class FakeFeature:
            def __init__(self, properties=None):
                self.properties = properties or {}

            def geometry(self):
                return FakeGeometry()

            def get(self, name):
                return self.properties[name]

            def set(self, key, value=None):
                if isinstance(key, dict):
                    self.properties.update(key)
                else:
                    self.properties[key] = value
                return self

        cities = MagicMock()
        area_image = MagicMock()
        years = tuple(range(2000, 2021))

        with (
            patch.object(city_isa.ee, "Feature", side_effect=lambda value: value),
            patch.object(city_isa.ee, "Number", side_effect=FakeNumber),
            patch.object(city_isa.ee, "ErrorMargin", return_value="margin"),
            patch.object(city_isa.ee, "Reducer")
        ):
            city_isa.build_city_polygon_isa_statistics(
                cities, area_image, years, 30, 10, 4
            )
            area_feature = FakeFeature()
            city_mapper = cities.map.call_args.args[0]
            city_mapper(area_feature)
            self.assertEqual(
                area_feature.properties["city_polygon_area_m2"], 1000.0
            )

            properties = {"city_polygon_area_m2": 1000.0}
            properties.update({
                f"isa_urban_area_m2_{year}": (
                    250.0 if year == 2000 else
                    750.0 if year == 2020 else
                    500.0
                )
                for year in years
            })
            area_feature = FakeFeature(properties)
            fraction_mapper = area_image.reduceRegions.return_value.map.call_args.args[0]
            fraction_mapper(area_feature)

        self.assertEqual(
            area_feature.properties["isa_urban_fraction_2000"].value, 0.25
        )
        self.assertEqual(
            area_feature.properties["isa_urban_fraction_2020"].value, 0.75
        )

    def test_export_submits_single_city_polygon_table_and_logs_task(self):
        with tempfile.TemporaryDirectory() as directory:
            task_log = Path(directory) / "city_isa_tasks.jsonl"
            config = make_config(task_log)
            cities = MagicMock()
            cities.size.return_value.getInfo.return_value = 11422
            task = SimpleNamespace(id="task-123")

            with (
                patch.object(exports.ee, "FeatureCollection", return_value=cities),
                patch.object(
                    exports, "prepare_isa_source",
                    return_value=("isa-image", "b1", tuple(range(1985, 2022)))
                ) as prepare_source,
                patch.object(
                    exports, "build_city_polygon_isa_area_image",
                    return_value="area-image"
                ) as build_area,
                patch.object(
                    exports, "build_city_polygon_isa_statistics",
                    return_value="city-stats"
                ) as build_stats,
                patch.object(
                    exports, "export_table_to_drive", return_value=task
                ) as submit
            ):
                result = exports.export_city_polygon_isa(config)

            self.assertIs(result, task)
            prepare_source.assert_called_once_with(config)
            build_area.assert_called_once_with(
                "isa-image", tuple(range(2000, 2021))
            )
            build_stats.assert_called_once()
            submitted = submit.call_args.kwargs
            self.assertEqual(submitted["collection"], "city-stats")
            self.assertEqual(submitted["description"], "city_isa_polygon_v0.1")
            self.assertEqual(len(submitted["selectors"]), 44)
            record = json.loads(task_log.read_text(encoding="utf-8"))
            self.assertEqual(record["city_count"], 11422)
            self.assertEqual(record["fraction_denominator"], "city_polygon_area_m2")
            self.assertEqual(record["masked_gaia_pixels"], "zero_contribution")


if __name__ == "__main__":
    unittest.main()
