import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src.era5_land_exports import (
    build_yearly_era5_rows,
    expected_month_labels,
    study_date_range,
    summarize_sample_rows,
    validate_monthly_source_coverage
)


class Era5LandExportTests(unittest.TestCase):
    def test_project_config_has_export_settings(self):
        with (MODULE_DIR / "config.json").open(encoding="utf-8") as file:
            config = json.load(file)
        self.assertTrue({
            "collection", "temperature_band", "kelvin_to_celsius_offset"
        } <= config["era5_land"].keys())
        self.assertTrue({
            "mode", "sample_city_count", "sample_year", "dataset_name",
            "folder", "max_error_m", "tile_scale"
        } <= config["era5_land_export"].keys())

    def test_study_date_range_includes_final_configured_year(self):
        self.assertEqual(
            study_date_range({
                "study": {"start_year": 2000, "end_year": 2024}
            }),
            ("2000-01-01", "2025-01-01")
        )

    def test_month_coverage_rejects_missing_or_duplicate_source_months(self):
        expected = ["2000-01", "2000-02"]
        validate_monthly_source_coverage(
            [946684800000, 949363200000],
            expected
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_monthly_source_coverage([946684800000], expected)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_monthly_source_coverage(
                [946684800000, 946684800000, 949363200000],
                expected
            )

    def test_sample_summary_reports_missing_values_keys_and_duplicates(self):
        rows = [
            {"properties": {
                "city_id": 7, "month": "2000-01", "temperature_2m_c": 10
            }},
            {"properties": {
                "city_id": 7, "month": "2000-01", "temperature_2m_c": 10
            }},
            {"properties": {
                "city_id": 9, "month": "2000-02", "temperature_2m_c": None
            }}
        ]
        result = summarize_sample_rows(
            rows, [7, 9], ["2000-01", "2000-02"]
        )
        self.assertEqual(result, {
            "city_count": 2,
            "row_count": 3,
            "missing_values": 1,
            "missing_keys": 2,
            "duplicate_keys": 1
        })

    def test_expected_month_labels_are_calendar_months(self):
        self.assertEqual(
            expected_month_labels(2024, 2024),
            [f"2024-{month:02d}" for month in range(1, 13)]
        )

    def test_yearly_builder_uses_all_calendar_months_and_native_projection(self):
        source = MagicMock()
        monthly_source = source.filterDate.return_value
        monthly_source.first.return_value = "monthly-image"
        image = MagicMock()
        image.select.return_value = image
        image.add.return_value = image
        image.rename.return_value = image
        reduced = MagicMock()
        image.reduceRegions.return_value = reduced
        reduced.map.return_value = "monthly-rows"
        flattened = MagicMock()
        flattened.flatten.return_value = "yearly-rows"
        with (
            patch("src.era5_land_exports.ee.Date") as ee_date,
            patch("src.era5_land_exports.ee.Image", return_value=image) as ee_image,
            patch("src.era5_land_exports.ee.Reducer.mean", return_value="mean"),
            patch(
                "src.era5_land_exports.ee.FeatureCollection",
                return_value=flattened
            ) as feature_collection
        ):
            result = build_yearly_era5_rows(
                source, "city-bboxes", 2024, "temperature_2m",
                -273.15, "native-projection", 4
            )
        self.assertEqual(result, "yearly-rows")
        self.assertEqual(source.filterDate.call_count, 12)
        self.assertEqual(ee_date.call_count, 12)
        self.assertEqual(ee_image.call_count, 12)
        self.assertEqual(image.reduceRegions.call_count, 12)
        image.reduceRegions.assert_called_with(
            collection="city-bboxes",
            reducer="mean",
            crs="native-projection",
            tileScale=4
        )
        image.add.assert_called_with(-273.15)
        feature_collection.assert_called_once()

        format_row = reduced.map.call_args_list[0].args[0]
        source_row = MagicMock()
        source_row.get.side_effect = {
            "city_id": 1,
            "mean": 24.85
        }.get
        with (
            patch("src.era5_land_exports.ee.Feature", side_effect=[source_row, "formatted"]) as feature,
            patch("src.era5_land_exports.ee.Dictionary", side_effect=lambda value: value)
        ):
            self.assertEqual(format_row("reduced-row"), "formatted")
        self.assertEqual(
            feature.call_args.args[1]["temperature_2m_c"],
            24.85
        )


if __name__ == "__main__":
    unittest.main()
