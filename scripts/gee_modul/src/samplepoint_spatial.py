from typing import Any

import ee


TILE_PROPERTY = "_samplepoint_tile"


def filter_tile(
    points: ee.FeatureCollection,
    column: int,
    row: int,
    tile_pixels: int
) -> ee.FeatureCollection:
    """Select a disjoint, half-open block of MODIS pixel coordinates."""
    x_min = column * tile_pixels
    y_min = row * tile_pixels
    return points.filter(ee.Filter.And(
        ee.Filter.gte("modis_x", x_min),
        ee.Filter.lt("modis_x", x_min + tile_pixels),
        ee.Filter.gte("modis_y", y_min),
        ee.Filter.lt("modis_y", y_min + tile_pixels)
    ))


def tile_for_test_city(
    points: ee.FeatureCollection,
    id_field: str,
    city_id: Any,
    tile_pixels: int
) -> tuple[int, int]:
    city_points = points.filter(ee.Filter.eq(id_field, city_id))
    if city_points.size().getInfo() == 0:
        raise ValueError(f"No sample points found for test city {city_id}.")
    feature = ee.Feature(city_points.sort("pixel_id").first())
    coordinates = feature.toDictionary(["modis_x", "modis_y"]).getInfo()
    x = coordinates.get("modis_x")
    y = coordinates.get("modis_y")
    if (
        type(x) not in (int, float)
        or type(y) not in (int, float)
        or int(x) != x
        or int(y) != y
    ):
        raise ValueError("Test city sample point has invalid MODIS coordinates.")
    return int(x) // tile_pixels, int(y) // tile_pixels


def get_occupied_tiles(
    points: ee.FeatureCollection,
    point_count: int,
    tile_pixels: int
) -> list[tuple[int, int]]:
    """Aggregate occupied tiles without transferring individual points."""
    valid_count = points.filter(
        ee.Filter.notNull(["modis_x", "modis_y"])
    ).size().getInfo()
    if valid_count != point_count:
        raise ValueError(
            f"{point_count - valid_count} sample points have missing MODIS "
            "coordinates; cannot partition all points."
        )

    def add_tile(feature: ee.Feature) -> ee.Feature:
        feature = ee.Feature(feature)
        column = ee.Number(feature.get("modis_x")).divide(tile_pixels).floor()
        row = ee.Number(feature.get("modis_y")).divide(tile_pixels).floor()
        tile_id = column.format("%d").cat(",").cat(row.format("%d"))
        return feature.set(TILE_PROPERTY, tile_id)

    histogram = points.map(add_tile).aggregate_histogram(TILE_PROPERTY).getInfo()
    if not histogram or sum(histogram.values()) != point_count:
        raise ValueError("Tile counts do not account for every sample point.")
    tiles = []
    for tile_id in histogram:
        parts = tile_id.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid MODIS tile ID: {tile_id}")
        tiles.append((int(parts[0]), int(parts[1])))
    return sorted(tiles)
