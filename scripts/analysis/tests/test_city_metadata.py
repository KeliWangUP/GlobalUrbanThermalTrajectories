import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from shapely.ops import transform
from pyproj import Transformer

from src.city_metadata import build_city_metadata, merge_city_metadata


class CityMetadataTests(unittest.TestCase):
    def test_builds_unique_wgs84_centroid_latitudes(self) -> None:
        to_mollweide = Transformer.from_crs(
            "EPSG:4326",
            "ESRI:54009",
            always_xy=True
        ).transform
        geometries = [
            transform(to_mollweide, box(-2, 10, 2, 14)),
            transform(to_mollweide, box(-2, -14, 2, -10))
        ]
        boundaries = gpd.GeoDataFrame(
            {"ID_UC_G0": [10, 20]},
            geometry=geometries,
            crs="ESRI:54009"
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            shapefile = Path(temporary_dir) / "cities.shp"
            boundaries.to_file(shapefile)
            metadata, repaired_count = build_city_metadata(shapefile)

        self.assertEqual(repaired_count, 0)
        self.assertEqual(metadata.columns.tolist(), ["ID_UC_G0", "centroid_lat"])
        self.assertEqual(metadata["ID_UC_G0"].tolist(), [10, 20])
        self.assertGreater(metadata.loc[0, "centroid_lat"], 0)
        self.assertLess(metadata.loc[1, "centroid_lat"], 0)

    def test_repairs_invalid_polygon_before_centroid(self) -> None:
        invalid_polygon = gpd.GeoSeries.from_wkt(
            ["POLYGON ((0 0, 2 2, 0 2, 2 0, 0 0))"],
            crs="ESRI:54009"
        ).iloc[0]
        boundaries = gpd.GeoDataFrame(
            {"ID_UC_G0": [10]},
            geometry=[invalid_polygon],
            crs="ESRI:54009"
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            shapefile = Path(temporary_dir) / "invalid_city.shp"
            boundaries.to_file(shapefile)
            metadata, repaired_count = build_city_metadata(shapefile)

        self.assertEqual(repaired_count, 1)
        self.assertTrue(metadata["centroid_lat"].between(-90, 90).all())

    def test_merges_metadata_for_all_panel_rows(self) -> None:
        panel = pd.DataFrame({
            "ID_UC_G0": [10, 10, 20],
            "year": [2000, 2001, 2000]
        })
        metadata = pd.DataFrame({
            "ID_UC_G0": [10, 20],
            "centroid_lat": [12.0, -12.0]
        })

        merged = merge_city_metadata(panel, metadata)

        self.assertEqual(merged["centroid_lat"].tolist(), [12.0, 12.0, -12.0])
        self.assertEqual(len(merged), len(panel))

    def test_replaces_existing_centroid_lat_when_remerging(self) -> None:
        panel = pd.DataFrame({
            "ID_UC_G0": [10],
            "centroid_lat": [-55.0]
        })
        metadata = pd.DataFrame({
            "ID_UC_G0": [10],
            "centroid_lat": [12.0]
        })

        merged = merge_city_metadata(panel, metadata)

        self.assertEqual(merged["centroid_lat"].tolist(), [12.0])

    def test_rejects_unmatched_panel_city_ids(self) -> None:
        panel = pd.DataFrame({"ID_UC_G0": [99], "year": [2000]})
        metadata = pd.DataFrame({
            "ID_UC_G0": [10],
            "centroid_lat": [12.0]
        })

        with self.assertRaisesRegex(ValueError, "missing from the boundary"):
            merge_city_metadata(panel, metadata)


if __name__ == "__main__":
    unittest.main()
