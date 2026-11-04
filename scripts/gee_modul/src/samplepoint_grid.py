import math
from typing import Any

import ee


_TRANSFORM_CACHE: dict[int, tuple[ee.Projection, tuple[float, ...]]] = {}


def validate_max_pixels(max_pixels: int) -> None:
    if type(max_pixels) is not int or max_pixels < 1:
        raise ValueError("samplepoint_attributes.max_pixels_per_cell must be positive.")


def _projection_transform(
    projection: ee.Projection,
    scale: float
) -> tuple[tuple[float, ...], float]:
    cached = _TRANSFORM_CACHE.get(id(projection))
    if cached is not None and cached[0] is projection:
        transform = cached[1]
    else:
        info: dict[str, Any] = projection.getInfo()
        raw_transform = info.get("transform")
        if (
            not isinstance(raw_transform, (list, tuple))
            or len(raw_transform) != 6
            or any(type(value) not in (int, float) for value in raw_transform)
        ):
            raise ValueError("MODIS projection must provide a six-value affine transform.")
        transform = tuple(float(value) for value in raw_transform)
        _TRANSFORM_CACHE[id(projection)] = (projection, transform)

    x_scale, x_shear, _, y_shear, y_scale, _ = transform
    if not math.isclose(x_shear, 0, abs_tol=1e-12) or not math.isclose(
        y_shear, 0, abs_tol=1e-12
    ):
        raise ValueError("MODIS cell geometry requires an axis-aligned projection grid.")
    if not math.isclose(abs(x_scale), scale, rel_tol=1e-6) or not math.isclose(
        abs(y_scale), scale, rel_tol=1e-6
    ):
        raise ValueError(
            "MODIS scale does not match the pixel dimensions in its projection transform."
        )
    pixel_area = abs(x_scale * y_scale - x_shear * y_shear)
    if not math.isfinite(pixel_area) or pixel_area <= 0:
        raise ValueError("MODIS projection transform has an invalid pixel area.")
    return transform, pixel_area


def build_modis_cell_regions(
    points: ee.FeatureCollection,
    projection: ee.Projection,
    scale: float
) -> ee.FeatureCollection:
    """Build exact MODIS cell rectangles from pixel indices and the affine grid."""
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("MODIS pixel scale must be a positive finite number.")
    transform, pixel_area = _projection_transform(projection, scale)
    x_scale, _, x_origin, _, y_scale, y_origin = transform

    def to_cell(feature: ee.Feature) -> ee.Feature:
        feature = ee.Feature(feature)
        x = ee.Number(feature.get("modis_x"))
        y = ee.Number(feature.get("modis_y"))
        x0 = x.multiply(x_scale).add(x_origin)
        x1 = x.add(1).multiply(x_scale).add(x_origin)
        y0 = y.multiply(y_scale).add(y_origin)
        y1 = y.add(1).multiply(y_scale).add(y_origin)
        x_min, x_max = (x0, x1) if x_scale > 0 else (x1, x0)
        y_min, y_max = (y0, y1) if y_scale > 0 else (y1, y0)
        cell = ee.Geometry.Rectangle(
            [x_min, y_min, x_max, y_max],
            proj=projection,
            geodesic=False
        )
        return feature.setGeometry(cell).set("_modis_area_m2", pixel_area)

    return points.map(to_cell)


def reduce_over_modis_cells(
    image: ee.Image,
    regions: ee.FeatureCollection,
    reducer: ee.Reducer,
    max_pixels: int,
    tile_scale: int
) -> ee.FeatureCollection:
    """Reduce regions on the image's native grid, without rescaling its projection."""
    validate_max_pixels(max_pixels)
    return image.reduceRegions(
        collection=regions,
        reducer=reducer,
        tileScale=tile_scale,
        maxPixelsPerRegion=max_pixels
    )
