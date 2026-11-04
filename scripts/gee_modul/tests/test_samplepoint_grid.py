import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src import samplepoint_grid as grid


class SamplepointGridTests(unittest.TestCase):
    def test_builds_square_cell_regions_from_pixel_centers(self):
        points = MagicMock()
        projection = object()
        margin = object()
        region = MagicMock()
        feature = MagicMock()
        geometry = feature.geometry.return_value
        buffered = geometry.buffer.return_value
        buffered.bounds.return_value = region
        with (
            patch.object(grid.ee, "ErrorMargin", return_value=margin),
            patch.object(grid.ee, "Feature", return_value=feature)
        ):
            result = grid.build_modis_cell_regions(points, projection, 60)
            map_function = points.map.call_args.args[0]
            map_function("point")
        self.assertEqual(result, points.map.return_value)
        geometry.buffer.assert_called_once_with(
            30, maxError=margin, proj=projection
        )
        buffered.bounds.assert_called_once_with(
            maxError=margin, proj=projection
        )
        feature.setGeometry.assert_called_once_with(region)
        feature.setGeometry.return_value.set.assert_called_once_with(
            "_modis_area_m2", region.area.return_value
        )
        region.area.assert_called_once_with(maxError=margin, proj=projection)

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
