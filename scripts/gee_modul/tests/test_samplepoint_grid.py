import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src import samplepoint_grid as grid


class SamplepointGridTests(unittest.TestCase):
    def test_samples_pixel_area_on_reference_grid(self):
        points = MagicMock()
        projection = MagicMock()
        image = MagicMock()
        pixel_area = image.pixelArea.return_value
        with patch.object(grid.ee, "Image", image):
            result = grid.sample_modis_pixel_area(
                points, ["pixel_id", "city"], projection, 926, 4
            )
        pixel_area.rename.assert_called_once_with("modis_area_m2")
        pixel_area.rename.return_value.sampleRegions.assert_called_once_with(
            collection=points,
            properties=["pixel_id", "city"],
            projection=projection,
            scale=926,
            tileScale=4,
            geometries=False
        )
        self.assertEqual(
            result, pixel_area.rename.return_value.sampleRegions.return_value
        )

    def test_rejects_invalid_pixel_limit(self):
        for max_pixels in (0, -1, True, 2.5):
            with self.subTest(max_pixels=max_pixels), self.assertRaises(ValueError):
                grid.validate_max_pixels(max_pixels)

    def test_accepts_positive_integer_pixel_limit(self):
        grid.validate_max_pixels(4096)


if __name__ == "__main__":
    unittest.main()
