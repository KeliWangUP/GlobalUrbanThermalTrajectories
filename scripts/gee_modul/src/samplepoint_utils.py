from typing import Any

import ee

from src.export_utils import asset_exists


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
