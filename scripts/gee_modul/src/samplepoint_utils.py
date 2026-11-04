from typing import Any

import ee

from src.export_utils import asset_exists


def samplepoint_base_fields(config: dict[str, Any]) -> list[str]:
    fields = ["pixel_id", config["cities"]["id_field"], "modis_x", "modis_y"]
    if config["sample_points"]["include_lonlat"]:
        fields.extend(["lon", "lat"])
    return fields


def get_samplepoint_city_ids(
    points: ee.FeatureCollection,
    id_field: str
) -> list[Any]:
    city_ids = points.aggregate_array(id_field).distinct().sort().getInfo()
    if not city_ids or any(city_id is None for city_id in city_ids):
        raise ValueError(f"Sample-point asset has no valid city IDs in '{id_field}'.")
    return city_ids


def partition_samplepoint_cities(
    city_ids: list[Any],
    city_batch_size: int
) -> list[list[Any]]:
    if type(city_batch_size) is not int or city_batch_size < 1:
        raise ValueError("sample_points.city_batch_size must be a positive integer.")
    if not city_ids:
        raise ValueError("Cannot partition an empty city list.")
    return [
        city_ids[start:start + city_batch_size]
        for start in range(0, len(city_ids), city_batch_size)
    ]


def load_sample_points(
    config: dict[str, Any],
    required_fields: list[str]
) -> tuple[ee.FeatureCollection, int]:
    """Load a non-empty point asset and validate its first feature's fields."""
    point_asset = config["export"]["asset_root"]
    if not asset_exists(point_asset):
        raise FileNotFoundError(
            f"Sample-point asset not found: {point_asset}. "
            "Build or upload the MODIS sample-point asset first."
        )

    points = ee.FeatureCollection(point_asset)
    point_count = points.size().getInfo()
    if point_count == 0:
        raise ValueError(f"Sample-point asset is empty: {point_asset}")

    available_fields = ee.Feature(points.first()).propertyNames().getInfo()
    missing_fields = [
        field for field in required_fields
        if field not in available_fields
    ]
    if missing_fields:
        raise ValueError(
            "Sample-point asset is missing required properties: "
            f"{missing_fields}"
        )
    return points, point_count
