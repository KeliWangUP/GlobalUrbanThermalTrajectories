from typing import Any

import ee

TILE_ID_PROPERTY = "modis_tile_id"
SUBTILE_ID_PROPERTY = "modis_subtile_id"


def _grid_transform(
    projection: ee.Projection
) -> tuple[list[float], str]:
    projection_info = projection.getInfo()
    transform = projection_info["transform"]
    if (
        len(transform) != 6
        or transform[0] <= 0
        or transform[4] >= 0
        or transform[1] != 0
        or transform[3] != 0
    ):
        raise ValueError(
            "MODIS projection must provide an axis-aligned affine transform "
            "with positive x scale and negative y scale."
        )
    return transform, projection_info["crs"]


def _validate_tile_grid(config: dict[str, Any]) -> tuple[int, int, int]:
    batch_config = config["batches"]
    tile_pixels = batch_config["tile_pixels"]
    columns = batch_config["tile_columns"]
    rows = batch_config["tile_rows"]
    if (
        any(type(value) is not int or value < 1 for value in (
            tile_pixels,
            columns,
            rows
        ))
        or columns > 99
        or rows > 99
    ):
        raise ValueError(
            "batches.tile_pixels, tile_columns, and tile_rows must be "
            "positive integers; tile_columns and tile_rows must be "
            "at most 99."
        )
    return tile_pixels, columns, rows


def build_modis_tile_grid(
    config: dict[str, Any],
    projection: ee.Projection
) -> ee.FeatureCollection:
    """Build the configured MODIS sinusoidal tile grid."""
    tile_pixels, columns, rows = _validate_tile_grid(config)
    transform, crs = _grid_transform(projection)
    # Geometry coordinates are map units; the raster affine transform is
    # used to calculate bounds, not attached to the geometry CRS.
    geometry_projection = ee.Projection(crs)
    tile_width = transform[0] * tile_pixels
    tile_height = -transform[4] * tile_pixels
    origin_x = transform[2]
    origin_y = transform[5]
    tiles = []

    for horizontal in range(columns):
        x_min = origin_x + horizontal * tile_width
        x_max = x_min + tile_width
        for vertical in range(rows):
            y_max = origin_y - vertical * tile_height
            y_min = y_max - tile_height
            tile_id = f"h{horizontal:02d}v{vertical:02d}"
            geometry = ee.Geometry.Rectangle(
                [x_min, y_min, x_max, y_max],
                proj=geometry_projection,
                geodesic=False
            )
            tiles.append(
                ee.Feature(geometry, {"tile_id": tile_id})
            )

    return ee.FeatureCollection(tiles)


def assign_cities_to_modis_tiles(
    config: dict[str, Any],
    projection: ee.Projection
) -> ee.FeatureCollection:
    """Assign each city to its first intersecting MODIS tile."""
    city_config = config["cities"]
    cities = ee.FeatureCollection(city_config["asset"])
    tiles = build_modis_tile_grid(config, projection)
    matches = ee.Join.saveFirst(
        matchKey="_matched_modis_tile",
        ordering="tile_id",
        ascending=True
    ).apply(
        primary=cities,
        secondary=tiles,
        condition=ee.Filter.intersects(
            leftField=".geo",
            rightField=".geo",
            maxError=config["rural"]["max_error_m"]
        )
    )

    def _set_tile_id(feature: ee.Feature) -> ee.Feature:
        feature = ee.Feature(feature)
        tile = ee.Feature(feature.get("_matched_modis_tile"))
        assigned = feature.set(
            TILE_ID_PROPERTY,
            tile.get("tile_id")
        )
        properties = assigned.propertyNames().remove(
            "_matched_modis_tile"
        )
        return assigned.select(properties, None, True)

    return ee.FeatureCollection(matches).map(_set_tile_id)


def get_tile_city_counts(
    city_assignments: ee.FeatureCollection
) -> dict[str, int]:
    """Return feature counts keyed by non-empty MODIS tile ID."""
    histogram = city_assignments.aggregate_histogram(
        TILE_ID_PROPERTY
    ).getInfo()
    if not histogram:
        return {}
    return {
        tile_id: int(count)
        for tile_id, count in sorted(histogram.items())
    }


def build_modis_subtile_grid(
    config: dict[str, Any],
    projection: ee.Projection
) -> ee.FeatureCollection:
    """Build a server-side grid of MODIS-aligned processing subtiles."""
    tile_pixels, tile_columns, tile_rows = _validate_tile_grid(config)
    transform, crs = _grid_transform(projection)
    geometry_projection = ee.Projection(crs)
    subtile_pixels = config["rural"]["static_assets"]["subtile_pixels"]
    if type(subtile_pixels) is not int or subtile_pixels < 1:
        raise ValueError(
            "rural.static_assets.subtile_pixels must be a positive integer."
        )

    total_columns = tile_pixels * tile_columns
    total_rows = tile_pixels * tile_rows
    column_count = (total_columns + subtile_pixels - 1) // subtile_pixels
    row_count = (total_rows + subtile_pixels - 1) // subtile_pixels
    if column_count > 1000 or row_count > 1000:
        raise ValueError(
            "The configured MODIS subtile grid exceeds the supported "
            "three-digit subtile IDs."
        )
    origin_x = transform[2]
    origin_y = transform[5]
    scale_x = transform[0]
    scale_y = -transform[4]

    def _make_subtile(column: Any, row: Any) -> ee.Feature:
        column = ee.Number(column)
        row = ee.Number(row)
        first_column = column.multiply(subtile_pixels)
        first_row = row.multiply(subtile_pixels)
        last_column = first_column.add(subtile_pixels).min(total_columns)
        last_row = first_row.add(subtile_pixels).min(total_rows)
        x_min = ee.Number(origin_x).add(
            first_column.multiply(scale_x)
        )
        x_max = ee.Number(origin_x).add(
            last_column.multiply(scale_x)
        )
        y_max = ee.Number(origin_y).subtract(
            first_row.multiply(scale_y)
        )
        y_min = ee.Number(origin_y).subtract(
            last_row.multiply(scale_y)
        )
        subtile_id = (
            ee.String("x")
            .cat(column.format("%03d"))
            .cat("y")
            .cat(row.format("%03d"))
        )
        geometry = ee.Geometry.Rectangle(
            [x_min, y_min, x_max, y_max],
            proj=geometry_projection,
            geodesic=False
        )
        return ee.Feature(
            geometry,
            {SUBTILE_ID_PROPERTY: subtile_id}
        )

    nested = ee.List.sequence(0, column_count - 1).map(
        lambda column: ee.List.sequence(0, row_count - 1).map(
            lambda row: _make_subtile(column, row)
        )
    )
    return ee.FeatureCollection(ee.List(nested).flatten())


def get_occupied_subtile_ids(
    config: dict[str, Any],
    projection: ee.Projection,
    rural_halos: ee.FeatureCollection
) -> list[str]:
    """Find MODIS subtiles intersecting at least one non-empty rural halo."""
    subtiles = build_modis_subtile_grid(config, projection)
    matches = ee.Join.saveFirst(
        matchKey="_matched_rural_halo"
    ).apply(
        primary=subtiles,
        secondary=rural_halos,
        condition=ee.Filter.intersects(
            leftField=".geo",
            rightField=".geo",
            maxError=config["rural"]["max_error_m"]
        )
    )
    occupied = ee.FeatureCollection(matches).filter(
        ee.Filter.notNull(["_matched_rural_halo"])
    ).select(
        [SUBTILE_ID_PROPERTY],
        None,
        False
    )
    return (
        occupied
        .aggregate_array(SUBTILE_ID_PROPERTY)
        .distinct()
        .sort()
        .getInfo()
    )


def get_subtile_geometry(
    config: dict[str, Any],
    projection: ee.Projection,
    subtile_id: str
) -> ee.Geometry:
    """Return a configured MODIS-aligned subtile geometry."""
    if (
        len(subtile_id) != 8
        or subtile_id[0] != "x"
        or subtile_id[4] != "y"
        or not subtile_id[1:4].isdigit()
        or not subtile_id[5:8].isdigit()
    ):
        raise ValueError(f"Invalid MODIS subtile ID: {subtile_id}")

    tile_pixels, tile_columns, tile_rows = _validate_tile_grid(config)
    transform, crs = _grid_transform(projection)
    geometry_projection = ee.Projection(crs)
    subtile_pixels = config["rural"]["static_assets"]["subtile_pixels"]
    total_columns = tile_pixels * tile_columns
    total_rows = tile_pixels * tile_rows
    column = int(subtile_id[1:4])
    row = int(subtile_id[5:8])
    first_column = column * subtile_pixels
    first_row = row * subtile_pixels
    if first_column >= total_columns or first_row >= total_rows:
        raise ValueError(f"MODIS subtile is outside the grid: {subtile_id}")

    last_column = min(first_column + subtile_pixels, total_columns)
    last_row = min(first_row + subtile_pixels, total_rows)
    scale_x = transform[0]
    scale_y = -transform[4]
    origin_x = transform[2]
    origin_y = transform[5]
    return ee.Geometry.Rectangle(
        [
            origin_x + first_column * scale_x,
            origin_y - last_row * scale_y,
            origin_x + last_column * scale_x,
            origin_y - first_row * scale_y
        ],
        proj=geometry_projection,
        geodesic=False
    )


def resolve_batch_indices(
    config: dict[str, Any],
    tile_ids: list[str]
) -> list[int]:
    """Return selected indices in the sorted, non-empty tile list."""
    selected = config["export"].get("rural_batch_indices")
    batch_count = len(tile_ids)
    if not batch_count:
        raise ValueError("No cities intersect the MODIS tile grid.")
    if selected is None:
        return list(range(batch_count))

    if (
        not isinstance(selected, list)
        or not selected
        or any(type(index) is not int for index in selected)
        or len(set(selected)) != len(selected)
        or any(not 0 <= index < batch_count for index in selected)
    ):
        raise ValueError(
            "export.rural_batch_indices must be null or a non-empty list "
            f"of unique integers from 0 to {batch_count - 1}."
        )
    return sorted(selected)
