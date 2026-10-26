from typing import Any

import ee

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


def filter_nonempty_geometries(
    features: ee.FeatureCollection,
    max_error_m: int
) -> ee.FeatureCollection:
    """Remove features whose geometry has no area."""
    with_area = ee.FeatureCollection(features).map(
        lambda feature: ee.Feature(feature).set(
            "_geometry_area_m2",
            ee.Feature(feature).geometry().area(max_error_m)
        )
    )
    return with_area.filter(ee.Filter.gt("_geometry_area_m2", 0))


def get_occupied_subtile_ids(
    config: dict[str, Any],
    projection: ee.Projection,
    rural_halos: ee.FeatureCollection,
    subtile_pixels: int
) -> list[str]:
    """Find MODIS subtiles intersecting at least one non-empty rural halo."""
    tile_pixels, tile_columns, tile_rows = _validate_tile_grid(config)
    transform, crs = _grid_transform(projection)
    if type(subtile_pixels) is not int or subtile_pixels < 1:
        raise ValueError(
            "subtile_pixels must be a positive integer."
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

    subtile_width = transform[0] * subtile_pixels
    subtile_height = -transform[4] * subtile_pixels
    origin_x = transform[2]
    origin_y = transform[5]
    subtile_projection = ee.Projection(
        crs,
        [
            subtile_width,
            0,
            origin_x,
            0,
            -subtile_height,
            origin_y
        ]
    )
    geometry_projection = ee.Projection(crs)
    max_error_m = config["rural"]["max_error_m"]

    def _add_subtile_id(cell: ee.Feature) -> ee.Feature:
        cell = ee.Feature(cell)
        center = ee.List(
            cell.geometry()
            .centroid(max_error_m, geometry_projection)
            .coordinates()
        )
        column = (
            ee.Number(center.get(0))
            .subtract(origin_x)
            .divide(subtile_width)
            .floor()
            .toInt()
        )
        row = (
            ee.Number(origin_y)
            .subtract(ee.Number(center.get(1)))
            .divide(subtile_height)
            .floor()
            .toInt()
        )
        subtile_id = (
            ee.String("x")
            .cat(column.format("%03d"))
            .cat("y")
            .cat(row.format("%03d"))
        )
        return cell.set({
            SUBTILE_ID_PROPERTY: subtile_id,
            "_subtile_column": column,
            "_subtile_row": row
        })

    def _ids_for_halo(feature: ee.Feature) -> ee.List:
        cells = ee.FeatureCollection(
            ee.Feature(feature)
            .geometry()
            .coveringGrid(subtile_projection)
        )
        indexed_cells = cells.map(_add_subtile_id).filter(
            ee.Filter.gte("_subtile_column", 0)
        ).filter(
            ee.Filter.lt("_subtile_column", column_count)
        ).filter(
            ee.Filter.gte("_subtile_row", 0)
        ).filter(
            ee.Filter.lt("_subtile_row", row_count)
        )
        return indexed_cells.aggregate_array(SUBTILE_ID_PROPERTY)

    halos_with_subtiles = ee.FeatureCollection(rural_halos).map(
        lambda feature: ee.Feature(feature).set(
            "_occupied_subtile_ids",
            _ids_for_halo(ee.Feature(feature))
        )
    )
    return (
        halos_with_subtiles
        .aggregate_array("_occupied_subtile_ids")
        .flatten()
        .distinct()
        .sort()
        .getInfo()
    )


def get_subtile_geometry(
    config: dict[str, Any],
    projection: ee.Projection,
    subtile_id: str,
    subtile_pixels: int
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
    if type(subtile_pixels) is not int or subtile_pixels < 1:
        raise ValueError(
            "subtile_pixels must be a positive integer."
        )
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
