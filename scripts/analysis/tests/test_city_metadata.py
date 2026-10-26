import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from shapely.ops import transform
from pyproj import Transformer

from src.city_metadata import (
    build_city_metadata,
    merge_city_metadata
)
from src.config import load_analysis_config

CZ_MACRO_GROUPS = {
    "Tropical": (1, 2, 3),
    "Arid": (4, 5, 6, 7),
    "Temperate": (9, 10, 11, 12, 13, 14, 15, 16),
    "Continental": (18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28),
    "Hot-summer Dry": (8, 17),
    "Polar": (29, 30)
}


class CityMetadataTests(unittest.TestCase):
    def test_configured_macro_groups_cover_all_raster_codes_once(self) -> None:
        configured_groups = load_analysis_config().cz_macro_groups

        self.assertEqual(configured_groups, CZ_MACRO_GROUPS)
        assigned_codes = [
            code
            for codes in configured_groups.values()
            for code in codes
        ]
        self.assertEqual(sorted(assigned_codes), list(range(1, 31)))

    @staticmethod
    def _write_climate_raster(path: Path, no_data: bool = False) -> None:
        values = np.zeros((180, 360), dtype="uint8")
        if not no_data:
            values[:90, :] = 1
            values[90:, :] = 29
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=180,
            width=360,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_origin(-180, 90, 1, 1)
        ) as raster:
            raster.write(values, 1)

    def test_builds_centroids_and_samples_climate_classes(self) -> None:
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
            raster = Path(temporary_dir) / "climate.tif"
            boundaries.to_file(shapefile)
            self._write_climate_raster(raster)
            metadata, repaired_count = build_city_metadata(
                shapefile,
                raster,
                CZ_MACRO_GROUPS
            )

        self.assertEqual(repaired_count, 0)
        self.assertEqual(
            metadata.columns.tolist(),
            [
                "ID_UC_G0",
                "centroid_lat",
                "koppen_geiger_code",
                "koppen_geiger_class",
                "koppen_geiger_macro_group"
            ]
        )
        self.assertEqual(metadata["ID_UC_G0"].tolist(), [10, 20])
        self.assertGreater(metadata.loc[0, "centroid_lat"], 0)
        self.assertLess(metadata.loc[1, "centroid_lat"], 0)
        self.assertEqual(
            metadata["koppen_geiger_code"].tolist(),
            [1, 29]
        )
        self.assertEqual(
            metadata["koppen_geiger_class"].tolist(),
            ["Af", "ET"]
        )
        self.assertEqual(
            metadata["koppen_geiger_macro_group"].tolist(),
            ["Tropical", "Polar"]
        )

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
            raster = Path(temporary_dir) / "climate.tif"
            boundaries.to_file(shapefile)
            self._write_climate_raster(raster)
            metadata, repaired_count = build_city_metadata(
                shapefile,
                raster,
                CZ_MACRO_GROUPS
            )

        self.assertEqual(repaired_count, 1)
        self.assertTrue(metadata["centroid_lat"].between(-90, 90).all())

    def test_zero_raster_class_is_stored_as_missing_climate(self) -> None:
        to_mollweide = Transformer.from_crs(
            "EPSG:4326",
            "ESRI:54009",
            always_xy=True
        ).transform
        boundary = gpd.GeoDataFrame(
            {"ID_UC_G0": [10]},
            geometry=[transform(to_mollweide, box(-2, 10, 2, 14))],
            crs="ESRI:54009"
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            shapefile = Path(temporary_dir) / "city.shp"
            raster = Path(temporary_dir) / "no_data.tif"
            boundary.to_file(shapefile)
            self._write_climate_raster(raster, no_data=True)
            metadata, _ = build_city_metadata(
                shapefile,
                raster,
                CZ_MACRO_GROUPS
            )

        self.assertTrue(pd.isna(metadata.loc[0, "koppen_geiger_code"]))
        self.assertTrue(pd.isna(metadata.loc[0, "koppen_geiger_class"]))
        self.assertTrue(pd.isna(metadata.loc[0, "koppen_geiger_macro_group"]))

    def test_merges_metadata_for_all_panel_rows(self) -> None:
        panel = pd.DataFrame({
            "ID_UC_G0": [10, 10, 20],
            "year": [2000, 2001, 2000]
        })
        metadata = pd.DataFrame({
            "ID_UC_G0": [10, 20],
            "centroid_lat": [12.0, -12.0],
            "koppen_geiger_code": pd.array([1, None], dtype="Int64"),
            "koppen_geiger_class": ["Af", None],
            "koppen_geiger_macro_group": ["Tropical", None]
        })

        merged = merge_city_metadata(panel, metadata, CZ_MACRO_GROUPS)

        self.assertEqual(merged["centroid_lat"].tolist(), [12.0, 12.0, -12.0])
        self.assertEqual(
            merged["koppen_geiger_class"].tolist(),
            ["Af", "Af", None]
        )
        self.assertEqual(
            merged["koppen_geiger_macro_group"].tolist(),
            ["Tropical", "Tropical", None]
        )
        self.assertEqual(len(merged), len(panel))

    def test_replaces_existing_city_metadata_when_remerging(self) -> None:
        panel = pd.DataFrame({
            "ID_UC_G0": [10],
            "centroid_lat": [-55.0],
            "koppen_geiger_code": [4],
            "koppen_geiger_class": ["BWh"],
            "koppen_geiger_macro_group": ["Arid"]
        })
        metadata = pd.DataFrame({
            "ID_UC_G0": [10],
            "centroid_lat": [12.0],
            "koppen_geiger_code": pd.array([1], dtype="Int64"),
            "koppen_geiger_class": ["Af"],
            "koppen_geiger_macro_group": ["Tropical"]
        })

        merged = merge_city_metadata(panel, metadata, CZ_MACRO_GROUPS)

        self.assertEqual(merged["centroid_lat"].tolist(), [12.0])
        self.assertEqual(merged["koppen_geiger_code"].tolist(), [1])
        self.assertEqual(merged["koppen_geiger_class"].tolist(), ["Af"])
        self.assertEqual(
            merged["koppen_geiger_macro_group"].tolist(),
            ["Tropical"]
        )

    def test_rejects_unmatched_panel_city_ids(self) -> None:
        panel = pd.DataFrame({"ID_UC_G0": [99], "year": [2000]})
        metadata = pd.DataFrame({
            "ID_UC_G0": [10],
            "centroid_lat": [12.0],
            "koppen_geiger_code": pd.array([1], dtype="Int64"),
            "koppen_geiger_class": ["Af"],
            "koppen_geiger_macro_group": ["Tropical"]
        })

        with self.assertRaisesRegex(ValueError, "missing from the boundary"):
            merge_city_metadata(panel, metadata, CZ_MACRO_GROUPS)


if __name__ == "__main__":
    unittest.main()
