import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src import samplepoint_grid as grid


class SamplepointGridTests(unittest.TestCase):
    def test_builds_cell_rectangle_from_pixel_indices_and_affine_transform(self):
        points = MagicMock()
        projection = MagicMock()
        projection.getInfo.return_value = {
            "transform": [60, 0, 100, 0, -60, 200]
        }
        region = MagicMock()
        feature = MagicMock()
        feature.get.side_effect = lambda name: 0

        class Number:
            def __init__(self, value):
                self.value = value

            def multiply(self, value):
                return Number(self.value * value)

            def add(self, value):
                return Number(self.value + value)

        with (
            patch.object(grid.ee, "Number", side_effect=Number),
            patch.object(grid.ee, "Geometry") as geometry,
            patch.object(grid.ee, "Feature", return_value=feature)
        ):
            geometry.Rectangle.return_value = region
            result = grid.build_modis_cell_regions(points, projection, 60)
            map_function = points.map.call_args.args[0]
            map_function("point")
        self.assertEqual(result, points.map.return_value)
        self.assertEqual(len(geometry.Rectangle.call_args.args), 1)
        coordinates = geometry.Rectangle.call_args.args[0]
        self.assertEqual([value.value for value in coordinates], [100, 140, 160, 200])
        self.assertEqual(geometry.Rectangle.call_args.kwargs, {
            "proj": projection, "geodesic": False
        })
        feature.setGeometry.assert_called_once_with(region)
        feature.setGeometry.return_value.set.assert_called_once_with(
            "_modis_area_m2", 3600
        )
        feature.geometry.assert_not_called()
        region.area.assert_not_called()

    def test_reduces_regions_on_native_image_grid(self):
        image = MagicMock()
        regions = MagicMock()
        reducer = object()
        with patch.object(grid, "validate_max_pixels") as validate:
            result = grid.reduce_over_modis_cells(image, regions, reducer, 4096, 4)
        validate.assert_called_once_with(4096)
        image.reduceRegions.assert_called_once_with(
            collection=regions,
            reducer=reducer,
            tileScale=4,
            maxPixelsPerRegion=4096
        )
        self.assertEqual(result, image.reduceRegions.return_value)

    def test_rejects_invalid_scale_or_pixel_limit(self):
        points = MagicMock()
        for scale in (0, -1, float("nan"), True):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                grid.build_modis_cell_regions(points, object(), scale)
        for max_pixels in (0, -1, True, 2.5):
            with self.subTest(max_pixels=max_pixels), self.assertRaises(ValueError):
                grid.validate_max_pixels(max_pixels)


if __name__ == "__main__":
    unittest.main()
